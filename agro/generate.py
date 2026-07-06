"""
VAJRA AGRO synthetic district generator — agronomy-informed satellite/weather data
with realistic mess. Monsoon (kharif) rice-belt scenario for an APAC smallholder region.

Hierarchy:  districts -> blocks -> villages -> plots  (villages are the DECISION unit,
the farming analogue of grid feeders; plots are the meter analogue).

Emits:
  * plot-level satellite NDVI readings (plot x date, ~5-day cadence) with monsoon CLOUD
    GAPS, sensor noise, duplicate passes, orphan plots, and NDVI*10000 integer-unit errors
    (all of which really occur in Sentinel-2 pipelines);
  * daily weather per block (rainfall with dry spells, tmax/tmin) driving GDD + soil water;
  * village x season history (5 seasons) with realized yields + drivers => training labels
    for the yield / crop-loss model (same idea as the grid's transformer-failure history).

Physics: NDVI follows a GDD-driven double-logistic phenology curve whose peak is suppressed
by water deficit during the reproductive stage; yield is a function of peak/integrated NDVI,
reproductive-stage water stress, GDD adequacy and pest pressure. See docs/RESEARCH.md.

Scales linearly like the grid generator:
  python -m agro.generate --plots 40000 --out data/agro_1m   # ~1M NDVI rows (40k x 24 dates)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from pipeline import io_util

CADENCE_DAYS = 5           # Sentinel-2 ~5-day revisit
SEASON_DAYS = 120          # kharif window
N_DATES = SEASON_DAYS // CADENCE_DAYS
GDD_BASE = 10.0            # rice base temperature (agronomic standard)

DISTRICTS = ["Warangal", "Guntur", "Nalgonda", "Karimnagar", "Nizamabad", "Khammam"]
REGION_LAT, REGION_LON = 17.9700, 79.6000

CROPS = ["rice", "cotton", "maize", "soybean"]
CROP_SHARE = [0.55, 0.22, 0.14, 0.09]
# per-crop: peak NDVI, GDD to maturity, reproductive window (frac of season), water sensitivity
CROP = {
    "rice":    {"ndvi_peak": 0.86, "gdd_mat": 1650, "repro": (0.45, 0.70), "wsens": 1.00, "base_yield": 4.2},
    "cotton":  {"ndvi_peak": 0.78, "gdd_mat": 2100, "repro": (0.50, 0.78), "wsens": 0.70, "base_yield": 2.1},
    "maize":   {"ndvi_peak": 0.82, "gdd_mat": 1500, "repro": (0.48, 0.68), "wsens": 0.85, "base_yield": 5.6},
    "soybean": {"ndvi_peak": 0.80, "gdd_mat": 1400, "repro": (0.52, 0.75), "wsens": 0.80, "base_yield": 2.6},
}
REACH = ["head", "mid", "tail"]      # canal position — the fairness dimension (tail chronically underserved)


def build_topology(n_plots: int, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    n_dist = max(3, n_plots // 20000)
    blocks_per_dist = rng.integers(5, 9, n_dist)
    n_block = int(blocks_per_dist.sum())

    d_ang = np.linspace(0, 2 * np.pi, n_dist, endpoint=False) + rng.normal(0, 0.1, n_dist)
    d_rad = rng.uniform(0.05, 0.22, n_dist)
    dist = pd.DataFrame({
        "district_id": [f"D-{i:02d}" for i in range(n_dist)],
        "name": [DISTRICTS[i % len(DISTRICTS)] for i in range(n_dist)],
        "lat": REGION_LAT + d_rad * np.sin(d_ang), "lon": REGION_LON + d_rad * np.cos(d_ang),
    })

    b_dist_idx = np.repeat(np.arange(n_dist), blocks_per_dist)
    b_ang = d_ang[b_dist_idx] + rng.normal(0, 0.4, n_block)
    b_rad = rng.uniform(0.02, 0.08, n_block)
    blocks = pd.DataFrame({
        "block_id": [f"B-{i:03d}" for i in range(n_block)],
        "district_id": dist["district_id"].values[b_dist_idx],
        "lat": dist["lat"].values[b_dist_idx] + b_rad * np.sin(b_ang),
        "lon": dist["lon"].values[b_dist_idx] + b_rad * np.cos(b_ang),
    })

    vil_per_block = np.maximum(4, rng.poisson(max(4, n_plots // (40 * n_block)), n_block))
    n_vil = int(vil_per_block.sum())
    v_block_idx = np.repeat(np.arange(n_block), vil_per_block)
    v_ang = b_ang[v_block_idx] + rng.normal(0, 0.5, n_vil)
    v_rad = rng.uniform(0.006, 0.03, n_vil)
    reach = rng.choice(REACH, n_vil, p=[0.34, 0.36, 0.30])
    villages = pd.DataFrame({
        "village_id": [f"V-{i:05d}" for i in range(n_vil)],
        "block_id": blocks["block_id"].values[v_block_idx],
        "district_id": blocks["district_id"].values[v_block_idx],
        "name": [f"{DISTRICTS[rng.integers(0, len(DISTRICTS))]}-{i}" for i in range(n_vil)],
        "canal_reach": reach,
        "lat": blocks["lat"].values[v_block_idx] + v_rad * np.sin(v_ang),
        "lon": blocks["lon"].values[v_block_idx] + v_rad * np.cos(v_ang),
        # tail-reach villages historically receive less irrigation support (fairness input)
        "past_support_index": np.clip(
            np.where(reach == "head", rng.uniform(0.6, 1.0, n_vil),
             np.where(reach == "mid", rng.uniform(0.35, 0.75, n_vil),
                      rng.uniform(0.05, 0.4, n_vil))), 0, 1).round(3),
    })

    # plots
    p_vil_idx = rng.integers(0, n_vil, n_plots)
    crop = rng.choice(CROPS, n_plots, p=CROP_SHARE)
    irr = rng.choice(["rainfed", "canal", "borewell"], n_plots, p=[0.5, 0.32, 0.18])
    plots = pd.DataFrame({
        "plot_id": [f"P-{i:07d}" for i in range(n_plots)],
        "village_id": villages["village_id"].values[p_vil_idx],
        "crop": crop,
        "irrigation": irr,
        "area_ha": np.round(rng.gamma(2.0, 0.6, n_plots) + 0.2, 2),   # smallholder: mostly <2 ha
        "soil_awc": rng.uniform(80, 180, n_plots).round(1),          # available water capacity mm
        "lat": villages["lat"].values[p_vil_idx] + rng.normal(0, 0.004, n_plots),
        "lon": villages["lon"].values[p_vil_idx] + rng.normal(0, 0.004, n_plots),
    })
    plots["_vidx"] = p_vil_idx
    return {"districts": dist, "blocks": blocks, "villages": villages, "plots": plots}


def build_weather(n_block: int, rng: np.random.Generator, start: pd.Timestamp,
                  block_ids) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily rainfall (monsoon onset + 1-2 dry spells) and temperature per block."""
    days = SEASON_DAYS
    t = np.arange(days)
    # baseline monsoon rainfall envelope (mm/day), onset ~day 5, active spells
    envelope = 9 * np.exp(-((t - 40) ** 2) / 1500) + 6 * np.exp(-((t - 85) ** 2) / 900) + 1.5
    rain = np.zeros((n_block, days), dtype=np.float32)
    dryspell_repro = np.zeros(n_block, dtype=np.float32)
    for b in range(n_block):
        wet = rng.random(days) < np.clip(envelope / 12.0, 0.05, 0.8)
        amt = rng.gamma(2.0, 6.0, days) * envelope / envelope.mean()
        series = np.where(wet, amt, 0.0)
        # inject a dry spell of 8-20 days somewhere, biased into the reproductive window
        if rng.random() < 0.75:
            start_d = int(rng.integers(45, 80))
            length = int(rng.integers(8, 21))
            series[start_d:start_d + length] *= rng.uniform(0.0, 0.15)
            # count dry days inside the reproductive window (~day 54-84)
            dryspell_repro[b] = float(np.sum(series[54:84] < 1.0))
        rain[b] = series
    tmax = 30 + 4 * np.sin(2 * np.pi * (t - 20) / 120) + rng.normal(0, 1.2, (n_block, days))
    tmax = tmax.astype(np.float32)
    tmin = (tmax - rng.uniform(7, 11, (n_block, 1))).astype(np.float32)

    rows = []
    for b in range(n_block):
        for di in range(days):
            rows.append((block_ids[b], start + pd.Timedelta(days=di),
                         round(float(rain[b, di]), 2), round(float(tmax[b, di]), 2),
                         round(float(tmin[b, di]), 2)))
    wdf = pd.DataFrame(rows, columns=["block_id", "date", "rain_mm", "tmax", "tmin"])
    return wdf, dryspell_repro


def _phenology(gdd_frac: np.ndarray, peak: float) -> np.ndarray:
    """Double-logistic NDVI vs fraction-of-thermal-maturity (green-up then senescence)."""
    up = 1.0 / (1.0 + np.exp(-12 * (gdd_frac - 0.25)))
    down = 1.0 / (1.0 + np.exp(12 * (gdd_frac - 0.80)))
    shape = up * down
    return 0.15 + (peak - 0.15) * shape / shape.max()


def generate_ndvi(topo, weather, dryspell_repro, out_dir, rng, start,
                  chunk_plots=4000, mess=True) -> tuple[int, pd.DataFrame]:
    """Per-plot NDVI time series from phenology x water-stress, + mess. Also returns the
    per-plot season summary used to synthesize yields (kept out of the 'clean' pipeline)."""
    plots = topo["plots"]
    villages = topo["villages"]
    blocks = topo["blocks"]
    block_index = {b: i for i, b in enumerate(blocks["block_id"])}
    # village -> block, and block-level GDD & dry-spell
    vil_block = villages.set_index("village_id")["block_id"]
    # precompute cumulative GDD per block over dates
    wpv = weather.pivot_table(index="block_id", columns="date", values=["rain_mm", "tmax", "tmin"])
    dates = pd.DatetimeIndex(sorted(weather["date"].unique()))
    sample_dates = dates[::CADENCE_DAYS][:N_DATES]

    tmean = (wpv["tmax"] + wpv["tmin"]) / 2.0
    gdd_daily = np.clip(tmean.values - GDD_BASE, 0, None)          # block x day
    gdd_cum = np.cumsum(gdd_daily, axis=1)                          # block x day
    rain_cum = np.cumsum(wpv["rain_mm"].values, axis=1)
    day_idx = {d: i for i, d in enumerate(pd.DatetimeIndex(tmean.columns))}
    samp_cols = [day_idx[d] for d in sample_dates]

    total_rows, part = 0, 0
    summaries = []
    t0 = time.time()
    for s in range(0, len(plots), chunk_plots):
        pc = plots.iloc[s:s + chunk_plots]
        P = len(pc)
        b_of_plot = np.array([block_index[vil_block[v]] for v in pc["village_id"]])
        crop_mat = np.array([CROP[c]["gdd_mat"] for c in pc["crop"]])
        crop_peak = np.array([CROP[c]["ndvi_peak"] for c in pc["crop"]])
        crop_ws = np.array([CROP[c]["wsens"] for c in pc["crop"]])

        gdd_frac = gdd_cum[b_of_plot][:, samp_cols] / crop_mat[:, None]      # P x N_DATES
        ndvi = np.stack([_phenology(gdd_frac[i], crop_peak[i]) for i in range(P)])

        # water stress: reproductive-window dry days for this plot's block, softened by
        # irrigation access and soil water capacity
        dsr = dryspell_repro[b_of_plot]
        irr_relief = np.select(
            [pc["irrigation"].values == "canal", pc["irrigation"].values == "borewell"],
            [0.55, 0.75], default=0.0)
        awc_relief = (pc["soil_awc"].values - 80) / 100 * 0.25
        stress = np.clip((dsr / 22.0) * crop_ws * (1 - irr_relief - awc_relief), 0, 0.85)
        # suppress NDVI in & after the reproductive window when the plot is water-stressed
        repro_mask = (gdd_frac > 0.45) & (gdd_frac < 0.95)
        ndvi = np.where(repro_mask, ndvi * (1 - stress[:, None] * 0.9), ndvi)
        ndvi = np.clip(ndvi + rng.normal(0, 0.02, ndvi.shape), 0.02, 0.98)

        # per-plot season summary (ground truth drivers -> yields; NOT fed to clean pipeline)
        peak_ndvi = ndvi.max(1)
        integ_ndvi = ndvi.sum(1)
        summaries.append(pd.DataFrame({
            "plot_id": pc["plot_id"].values, "village_id": pc["village_id"].values,
            "crop": pc["crop"].values, "area_ha": pc["area_ha"].values,
            "peak_ndvi": peak_ndvi, "integ_ndvi": integ_ndvi,
            "water_stress": stress, "gdd_frac_end": gdd_frac[:, -1],
        }))

        # build long readings table
        df = pd.DataFrame({
            "plot_id": np.repeat(pc["plot_id"].values, N_DATES),
            "date": np.tile(sample_dates.values, P),
            "ndvi": ndvi.ravel().astype(np.float32),
        })

        if mess:
            n = len(df)
            r = rng.random(n)
            # monsoon cloud cover: ~22% of passes unusable (NDVI garbage / missing)
            cloud = r < 0.22
            df.loc[cloud & (r < 0.11), "ndvi"] = np.nan
            cg = cloud & (r >= 0.11)
            df.loc[cg, "ndvi"] = rng.uniform(-0.1, 0.2, int(cg.sum())).astype(np.float32)
            # NDVI reported as scaled integer (x10000) — classic real-data unit error
            unit = (r >= 0.22) & (r < 0.25)
            df.loc[unit, "ndvi"] = df.loc[unit, "ndvi"] * 10000.0
            df["cloud_flag"] = cloud.astype("int8")
            # duplicate passes + orphan plots
            dups = df.sample(frac=0.02, random_state=int(rng.integers(1e9)))
            orph = df.sample(frac=0.003, random_state=int(rng.integers(1e9))).copy()
            orph["plot_id"] = "P-9999999"
            df = pd.concat([df, dups, orph], ignore_index=True)
        else:
            df["cloud_flag"] = 0

        io_util.save(df, out_dir, f"ndvi_{part:04d}")
        total_rows += len(df); part += 1
    dt = time.time() - t0
    print(f"  ndvi: {total_rows:,} rows in {dt:.1f}s ({total_rows/max(dt,1e-9):,.0f} rows/s)")
    return total_rows, pd.concat(summaries, ignore_index=True)


def build_history(topo, summary, dryspell_repro, blocks, rng, seasons=5) -> pd.DataFrame:
    """village x season yields with drivers — training labels for the yield/loss model.

    Realized yield ~ f(peak/integrated NDVI, reproductive water stress, GDD adequacy, pest).
    normal_yield = 5-season mean; a loss event = yield < 0.8 * normal (PMFBY-style threshold).
    """
    villages = topo["villages"]
    # aggregate this season's plot summaries to village means (drivers for THIS season)
    vgrp = summary.groupby("village_id").agg(
        peak_ndvi=("peak_ndvi", "mean"), integ_ndvi=("integ_ndvi", "mean"),
        water_stress=("water_stress", "mean"), gdd_frac=("gdd_frac_end", "mean"),
        area_ha=("area_ha", "sum")).reset_index()
    vgrp = vgrp.merge(villages[["village_id", "block_id", "canal_reach"]], on="village_id")

    rows = []
    for si in range(seasons):
        # historical seasons: perturb drivers around this season's structure
        jitter_ndvi = rng.normal(0, 0.05, len(vgrp))
        jitter_stress = np.clip(rng.normal(0, 0.12, len(vgrp)), -0.3, 0.5)
        peak = np.clip(vgrp["peak_ndvi"].values + jitter_ndvi, 0.3, 0.95)
        integ = np.clip(vgrp["integ_ndvi"].values * (1 + jitter_ndvi), 4, None)
        stress = np.clip(vgrp["water_stress"].values + jitter_stress, 0, 0.9)
        gddf = np.clip(vgrp["gdd_frac"].values + rng.normal(0, 0.06, len(vgrp)), 0.7, 1.2)
        pest = np.clip(rng.gamma(1.4, 0.12, len(vgrp)), 0, 0.8)
        base = 4.0
        yield_t = (base * (0.4 + 0.7 * (peak - 0.3)) * (1 - 0.75 * stress)
                   * np.clip(gddf, 0.6, 1.05) * (1 - 0.5 * pest)
                   + rng.normal(0, 0.25, len(vgrp))).clip(0.2, None)
        rows.append(pd.DataFrame({
            "village_id": vgrp["village_id"], "season": si,
            "peak_ndvi": peak.astype(np.float32), "integ_ndvi": integ.astype(np.float32),
            "water_stress": stress.astype(np.float32), "gdd_frac": gddf.astype(np.float32),
            "pest_index": pest.astype(np.float32), "yield_t_ha": yield_t.astype(np.float32),
        }))
    hist = pd.concat(rows, ignore_index=True)
    normal = hist.groupby("village_id")["yield_t_ha"].transform("mean")
    hist["normal_yield"] = normal.astype(np.float32)
    hist["loss_event"] = (hist["yield_t_ha"] < 0.8 * normal).astype(np.int8)  # PMFBY-style
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", type=int, default=40000)
    ap.add_argument("--seasons", type=int, default=5)
    ap.add_argument("--out", default="data/agro")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-plots", type=int, default=4000)
    ap.add_argument("--start", default="2026-06-10")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print(f"VAJRA AGRO generator: {args.plots:,} plots x {N_DATES} passes "
          f"(~{args.plots * N_DATES:,} NDVI rows)")

    topo = build_topology(args.plots, rng)
    weather, dsr = build_weather(len(topo["blocks"]), rng, pd.Timestamp(args.start),
                                 topo["blocks"]["block_id"].values)
    for k in ["districts", "blocks", "villages"]:
        io_util.save(topo[k], args.out, k)
    io_util.save(topo["plots"].drop(columns=["_vidx"]), args.out, "plots")
    io_util.save(weather, args.out, "weather")

    n_rows, summary = generate_ndvi(topo, weather, dsr, args.out, rng,
                                    pd.Timestamp(args.start), args.chunk_plots)
    hist = build_history(topo, summary, dsr, topo["blocks"], rng, args.seasons)
    io_util.save(hist, args.out, "yield_history")

    meta = {
        "domain": "agro", "plots": args.plots, "ndvi_rows": int(n_rows),
        "villages": len(topo["villages"]), "blocks": len(topo["blocks"]),
        "districts": len(topo["districts"]), "dates": N_DATES, "seasons": args.seasons,
        "start": args.start, "seed": args.seed, "scenario": "kharif_monsoon_dryspell",
        "generated_in_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  topology: {meta['districts']} districts / {meta['blocks']} blocks / "
          f"{meta['villages']} villages; yield_history: {len(hist):,} rows")
    print(f"done in {meta['generated_in_s']}s -> {args.out}")


if __name__ == "__main__":
    main()
