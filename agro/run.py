"""
AGRO pipeline orchestrator — one command, both engines, every stage timed (mirrors grid run.py).

  python -m agro.run --data data/agro_1m --engine cpu --out out/agro_cpu
  python -m agro.run --data data/agro_1m --engine gpu --out out/agro_gpu

`--engine gpu` activates NVIDIA RAPIDS cudf.pandas + XGBoost(device=cuda) + CuPy allocator.
Writes the same artifact shape as the grid pack so the API/dashboard treat both uniformly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from pipeline.engine import activate, array_module, xgb_device
from pipeline.run import Timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--alloc-candidates", type=int, default=20000)
    ap.add_argument("--budget-ml", type=float, default=None)
    args = ap.parse_args()
    out_dir = args.out or f"out/agro_{args.engine}"
    os.makedirs(out_dir, exist_ok=True)

    env = activate(args.engine)
    from agro import allocate, models, scoring, stages  # noqa: PLC0415
    import pandas as pd  # noqa: F401,PLC0415  (cudf.pandas proxy on GPU)

    print(f"VAJRA AGRO | engine={args.engine} | {env.get('gpu_name') or 'CPU'}")
    T = Timer()
    t_all = time.time()

    with T.stage("1_ingest"):
        d = stages.ingest(args.data)
        rows_in = len(d["ndvi"])
    with T.stage("2_clean_gapfill"):
        ndvi, quality = stages.clean(d["ndvi"], d["plots"])
    with T.stage("3_join_enrich"):
        enriched = stages.join_enrich(ndvi, d["plots"], d["villages"])
    with T.stage("4_aggregate"):
        plot_season, vil_date, block_w = stages.aggregate(enriched, d["weather"])
        del enriched, ndvi, d["ndvi"]
    with T.stage("5_features"):
        vf = stages.village_features(plot_season, vil_date, block_w, d["villages"])
    with T.stage("6_yield_models"):
        clf, reg, auc = models.train_yield_models(d["yield_history"],
                                                  xgb_device(args.engine), args.fast)
        scored = models.score_villages(clf, reg, vf)
    with T.stage("7_priority_trigger"):
        f = scoring.score_villages_priority(scored, d["yield_history"])
        summary = scoring.system_summary(f, vil_date, d["weather"], quality, auc)
    with T.stage("8_allocate_bench"):
        state = village_state(f)
        budget = args.budget_ml or max(round(summary["water_need_total_ml"] * 0.35, 1), 5.0)
        al = allocate.allocate(state, budget, args.alloc_candidates,
                               xp=array_module(args.engine), seed=11)
    with T.stage("9_write_outputs"):
        write_outputs(out_dir, f, vil_date, summary, al, state)

    total = round(time.time() - t_all, 2)
    timings = {"engine": args.engine, "domain": "agro", "env": env, "rows_in": int(rows_in),
               "rows_clean": quality["rows_clean"], "stages": T.stages,
               "total_seconds": total, "dispatch_block_seconds": 900,
               "fits_in_block": total < 900, "blocks_needed": max(1, int(-(-total // 900))),
               "whatif_plans_per_second": al.get("plans_per_second", 0)}
    with open(os.path.join(out_dir, "timings.json"), "w") as fjs:
        json.dump(timings, fjs, indent=2)
    print(f"TOTAL {total}s | rows_in={rows_in:,} | engine={args.engine} "
          f"| fits in refresh window: {timings['fits_in_block']}")


def village_state(f) -> list[dict]:
    cols = ["village_id", "name", "yield_saveable_t", "water_need_ml",
            "past_support_index", "canal_reach", "p_loss"]
    return [{k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in cols}
            for _, r in f[cols].iterrows()]


def write_outputs(out_dir, f, vil_date, summary, al, state):
    import pandas as pd  # noqa: PLC0415
    spark = {vid: [round(float(v), 3) for v in grp.sort_values("date")["ndvi"]]
             for vid, grp in vil_date.groupby("village_id", observed=True)}

    villages_out = []
    for i, r in f.iterrows():
        vid = r["village_id"]
        villages_out.append({
            "rank": i + 1, "village_id": vid, "name": r["name"],
            "district_id": r["district_id"], "block_id": r["block_id"],
            "canal_reach": r["canal_reach"], "lat": float(r["lat"]), "lon": float(r["lon"]),
            "vapi": float(r["vapi"]),
            "components": {k: float(r[f"c_{k}"]) for k in
                           ["loss_risk", "yield_shortfall", "ndvi_decline", "exposure"]},
            "reason_codes": list(r["reason_codes_full"]),
            "p_loss": round(float(r["p_loss"]), 4),
            "yield_pred": round(float(r["yield_pred"]), 2),
            "normal_yield": round(float(r["normal_yield"]), 2),
            "yield_shortfall_frac": round(float(r["yield_shortfall_frac"]), 3),
            "insurance_trigger": int(r["insurance_trigger"]),
            "trigger_score": float(r["trigger_score"]),
            "water_deficit": round(float(r["water_deficit"]), 3),
            "area_ha": round(float(r["area_ha"]), 1), "plots": int(r["plots"]),
            "rainfed_frac": round(float(r["rainfed_frac"]), 2),
            "past_support_index": float(r["past_support_index"]),
            "yield_saveable_t": float(r["yield_saveable_t"]),
            "water_need_ml": float(r["water_need_ml"]),
            "ndvi_trend": round(float(r["ndvi_trend"]), 3),
            "spark_ndvi": spark.get(vid, []),
            "shap": list(r["shap_detail"]),
        })

    artifacts = {
        "villages.json": villages_out,
        "system.json": summary,
        "allocate_bench.json": al,
        "village_state.json": state,
        # trigger watchlist (insurance) as its own artifact for the API/dashboard
        "triggers.json": [v for v in villages_out if v["insurance_trigger"]][:300],
    }
    for name, obj in artifacts.items():
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(obj, fh)
    print(f"  wrote {len(artifacts)} artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
