"""
AEGIS pipeline orchestrator — one command, both engines, every stage timed.

  python -m pipeline.run --data data/city_1m --engine cpu --out out/cpu_run
  python -m pipeline.run --data data/city_1m --engine gpu --out out/gpu_run

The --engine flag is the ONLY difference between CPU and GPU runs:
`gpu` activates NVIDIA RAPIDS cudf.pandas + CuPy + XGBoost(device=cuda).
Per-stage wall-clock timings are written to timings.json — the raw material for the
runtime / scale / freshness / decision acceleration proofs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from pipeline.engine import activate, array_module, xgb_device


class Timer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    def stage(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()
                print(f"  [{name}] ...", flush=True)
                return self

            def __exit__(self, *a):
                dt = time.time() - self.t0
                timer.stages[name] = round(dt, 3)
                print(f"  [{name}] {dt:.2f}s", flush=True)
        return _Ctx()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true", help="smaller models for quick runs")
    ap.add_argument("--whatif-candidates", type=int, default=20000)
    ap.add_argument("--target-mw", type=float, default=None,
                    help="what-if benchmark target; default = system deficit")
    args = ap.parse_args()
    out_dir = args.out or f"out/{args.engine}_run"
    os.makedirs(out_dir, exist_ok=True)

    env = activate(args.engine)  # MUST precede pandas-importing modules
    from pipeline import models, scoring, stages, whatif  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415  (proxied by cudf.pandas on GPU)

    print(f"AEGIS pipeline | engine={args.engine} | {env.get('gpu_name') or 'CPU'}")
    T = Timer()
    t_all = time.time()

    with T.stage("1_ingest"):
        d = stages.ingest(args.data)
        rows_in = len(d["readings"])

    with T.stage("2_clean"):
        readings, quality = stages.clean(d["readings"], d["meters"])

    with T.stage("3_join_enrich"):
        enriched = stages.join_enrich(readings, d["meters"], d["transformers"], d["feeders"])

    with T.stage("4_aggregate"):
        tx_int, fd_int = stages.aggregate(enriched)
        del enriched, readings, d["readings"]

    with T.stage("5_features"):
        txf = stages.tx_features(tx_int)
        fdf = stages.feeder_features(fd_int)

    with T.stage("6_risk_model"):
        clf, auc = models.train_risk_model(d["tx_history"], xgb_device(args.engine), args.fast)
        txs = models.score_transformers(clf, txf, d["transformers"])

    with T.stage("7_forecast"):
        forecast, fc_ts, fc_temp = models.train_and_forecast(
            fd_int, d["weather"], 16, xgb_device(args.engine), args.fast)

    with T.stage("8_score_rank"):
        f = scoring.score_feeders(fdf, txs, d["feeders"], d["meters"],
                                  d["transformers"], d["shed_history"], forecast)
        summary = scoring.system_summary(f, fd_int, forecast, d["weather"], quality, auc)

    with T.stage("9_whatif_bench"):
        state = feeder_state(f)
        target = args.target_mw or max(summary["deficit_mw"], 5.0)
        wi = whatif.evaluate(state, target, args.whatif_candidates,
                             xp=array_module(args.engine), seed=11)

    with T.stage("10_write_outputs"):
        write_outputs(out_dir, f, txs, forecast, fc_ts, fc_temp, fd_int,
                      summary, wi, state)

    total = round(time.time() - t_all, 2)
    timings = {
        "engine": args.engine, "env": env, "rows_in": int(rows_in),
        "rows_clean": quality["rows_clean"], "stages": T.stages, "total_seconds": total,
        "dispatch_block_seconds": 900,
        "fits_in_block": total < 900,
        "blocks_needed": max(1, int(-(-total // 900))),
        "whatif_plans_per_second": wi.get("plans_per_second", 0),
    }
    with open(os.path.join(out_dir, "timings.json"), "w") as fjs:
        json.dump(timings, fjs, indent=2)
    print(f"TOTAL {total}s | rows_in={rows_in:,} | engine={args.engine} "
          f"| fits in 15-min block: {timings['fits_in_block']}")


def feeder_state(f) -> list[dict]:
    cols = ["feeder_id", "name", "sheddable_mw", "pain_score", "shed_hours_30d",
            "is_protected", "critical_type"]
    return [
        {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in cols}
        for _, row in f[cols].iterrows()
    ]


def write_outputs(out_dir, f, txs, forecast, fc_ts, fc_temp, fd_int, summary, wi, state):
    import pandas as pd  # noqa: PLC0415

    # feeder cards for the dashboard (spark = last 24h, fc = next 4h)
    last_ts = fd_int["ts"].max()
    w24 = fd_int[fd_int["ts"] > last_ts - pd.Timedelta(hours=24)]
    spark = {fid: [round(float(v) / 1000.0, 3) for v in grp.sort_values("ts")["kw"]][::4]
             for fid, grp in w24.groupby("feeder_id", observed=True)}

    feeders_out = []
    for i, r in f.iterrows():
        fid = r["feeder_id"]
        feeders_out.append({
            "rank": i + 1, "feeder_id": fid, "name": r["name"],
            "substation_id": r["substation_id"], "lat": float(r["lat"]), "lon": float(r["lon"]),
            "fci": float(r["fci"]),
            "components": {k: float(r[f"c_{k}"]) for k in
                           ["forecast_stress", "tx_risk", "utilization", "volatility"]},
            "reason_codes": list(r["reason_codes"]),
            "current_mw": round(float(r["kw_now"]) / 1000.0, 3),
            "forecast_peak_mw": round(float(r["forecast_peak_kw"]) / 1000.0, 3),
            "capacity_mw": round(float(r["capacity_kw"]) / 1000.0, 3),
            "headroom_mw": float(r["headroom_mw"]), "sheddable_mw": float(r["sheddable_mw"]),
            "pain_score": float(r["pain_score"]), "is_protected": bool(r["is_protected"]),
            "critical_type": str(r["critical_type"]),
            "shed_hours_30d": float(r["shed_hours_30d"]),
            "complaints_30d": int(r["complaints_30d"]),
            "customers": int(r["customers"]), "tx_high_risk": int(r["tx_high_risk"]),
            "utilization_now": round(float(r["utilization_now"]), 3),
            "spark_mw": spark.get(fid, []),
            "forecast_mw": [round(float(v) / 1000.0, 3) for v in forecast.loc[fid]]
            if fid in forecast.index else [],
        })

    tx_out = []
    for _, r in txs.head(250).iterrows():
        tx_out.append({
            "transformer_id": r["transformer_id"], "feeder_id": r["feeder_id"],
            "p_fail_72h": round(float(r["p_fail_72h"]), 4),
            "loading_max": round(float(r["loading_max"]), 3),
            "loading_mean": round(float(r["loading_mean"]), 3),
            "overload_minutes": round(float(r["overload_minutes"]), 1),
            "sag_count": int(r["sag_count"]), "age_years": int(r["age_years"]),
            "capacity_kva": int(r["capacity_kva"]),
            "lat": float(r["lat"]), "lon": float(r["lon"]),
            "reason_codes": list(r["reason_codes"]), "shap": list(r["shap_detail"]),
        })

    artifacts = {
        "feeders.json": feeders_out, "transformers.json": tx_out,
        "system.json": {**summary, "forecast_ts": fc_ts, "forecast_temp": fc_temp},
        "whatif_bench.json": wi, "feeder_state.json": state,
    }
    for name, obj in artifacts.items():
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(obj, fh)
    print(f"  wrote {len(artifacts)} artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
