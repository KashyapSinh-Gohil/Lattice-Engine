"""
AGRO pipeline stages 1-5: ingest -> clean (NDVI gap-fill) -> join -> aggregate -> features.

Plain pandas API → runs on NVIDIA RAPIDS cudf.pandas under --engine gpu. The NDVI cleaning
(per-plot interpolation/smoothing over millions of plot×date cells) and the plot→village
geospatial rollups are the GPU-heavy stages, exactly like the grid pack's clean/aggregate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import io_util

NDVI_UNIT_ERR = 1.5      # any NDVI > 1.5 is a scaled-integer (×10000) unit error
GDD_BASE = 10.0


def ingest(data_dir: str) -> dict:
    d = {"ndvi": io_util.load_glob(data_dir, "ndvi_*")}
    for name in ["plots", "villages", "blocks", "districts", "weather", "yield_history"]:
        d[name] = io_util.load(data_dir, name)
    return d


def clean(ndvi: pd.DataFrame, plots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Raw satellite passes -> a trustworthy per-plot NDVI series. Every fix counted."""
    q = {"rows_in": int(len(ndvi))}
    ndvi["date"] = ndvi["date"].dt.floor("D")

    before = len(ndvi)
    ndvi = ndvi.drop_duplicates(subset=["plot_id", "date"], keep="first")
    q["duplicate_passes_removed"] = int(before - len(ndvi))

    before = len(ndvi)
    ndvi = ndvi.merge(plots[["plot_id"]], on="plot_id", how="inner")
    q["orphan_plots_removed"] = int(before - len(ndvi))

    # scaled-integer unit errors (NDVI ×10000)
    unit = ndvi["ndvi"] > NDVI_UNIT_ERR
    q["unit_errors_fixed"] = int(unit.sum())
    ndvi.loc[unit, "ndvi"] = ndvi.loc[unit, "ndvi"] / 10000.0

    # cloud-contaminated / physically-impossible values -> NaN for gap-fill
    bad = (ndvi["cloud_flag"] == 1) | (ndvi["ndvi"] < 0.05) | (ndvi["ndvi"] > 1.0)
    q["cloud_masked"] = int(bad.sum())
    ndvi.loc[bad, "ndvi"] = np.nan

    # gap-fill: per-plot time-ordered interpolation then smoothing (heavy groupby → GPU)
    ndvi = ndvi.sort_values(["plot_id", "date"])
    q["gaps_filled"] = int(ndvi["ndvi"].isna().sum())
    g = ndvi.groupby("plot_id")["ndvi"]
    ndvi["ndvi"] = g.transform(lambda s: s.interpolate(limit_direction="both"))
    # residual all-NaN plots -> global median; then a light rolling smooth
    ndvi["ndvi"] = ndvi["ndvi"].fillna(ndvi["ndvi"].median())
    ndvi["ndvi_smooth"] = (ndvi.groupby("plot_id")["ndvi"]
                           .transform(lambda s: s.rolling(3, min_periods=1, center=True).mean()))
    q["rows_clean"] = int(len(ndvi))
    return ndvi, q


def join_enrich(ndvi, plots, villages) -> pd.DataFrame:
    r = ndvi.merge(plots[["plot_id", "village_id", "crop", "area_ha", "irrigation"]],
                   on="plot_id", how="left")
    r = r.merge(villages[["village_id", "block_id", "district_id", "canal_reach"]],
                on="village_id", how="left")
    return r


def _gdd_by_block(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["gdd"] = ((w["tmax"] + w["tmin"]) / 2.0 - GDD_BASE).clip(lower=0)
    agg = (w.groupby("block_id")
           .agg(gdd_total=("gdd", "sum"), rain_total=("rain_mm", "sum"),
                rain_repro=("rain_mm", lambda s: s.iloc[54:84].sum() if len(s) >= 84 else s.sum()),
                dry_days_repro=("rain_mm", lambda s: int((s.iloc[54:84] < 1.0).sum())
                                if len(s) >= 84 else int((s < 1.0).sum())))
           .reset_index())
    return agg


def aggregate(enriched, weather) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """plot×date -> plot-season summary and village×date NDVI, plus block weather features."""
    plot_season = (enriched.groupby(["plot_id", "village_id", "crop", "area_ha",
                                     "irrigation", "canal_reach", "block_id"], observed=True)
                   .agg(peak_ndvi=("ndvi_smooth", "max"),
                        mean_ndvi=("ndvi_smooth", "mean"),
                        integ_ndvi=("ndvi_smooth", "sum"),
                        last_ndvi=("ndvi_smooth", "last"))
                   .reset_index())
    vil_date = (enriched.groupby(["village_id", "date"], observed=True)
                .agg(ndvi=("ndvi_smooth", "mean")).reset_index())
    block_w = _gdd_by_block(weather)
    return plot_season, vil_date, block_w


def village_features(plot_season, vil_date, block_w, villages) -> pd.DataFrame:
    """Per-village features for the yield/risk models + advisory scoring."""
    vf = (plot_season.groupby(["village_id", "block_id", "canal_reach"], observed=True)
          .agg(peak_ndvi=("peak_ndvi", "mean"), mean_ndvi=("mean_ndvi", "mean"),
               integ_ndvi=("integ_ndvi", "mean"), last_ndvi=("last_ndvi", "mean"),
               area_ha=("area_ha", "sum"), plots=("plot_id", "count"),
               rainfed_frac=("irrigation", lambda s: float((s == "rainfed").mean())))
          .reset_index())

    # NDVI trend over the last 3 passes (decline = distress) via village×date series
    vil_date = vil_date.sort_values(["village_id", "date"])
    last3 = (vil_date.groupby("village_id", observed=True)["ndvi"]
             .apply(lambda s: s.tail(3).values))
    trend = last3.apply(lambda a: float(a[-1] - a[0]) if len(a) >= 2 else 0.0)
    vf = vf.merge(trend.rename("ndvi_trend").reset_index(), on="village_id", how="left")

    vf = vf.merge(block_w, on="block_id", how="left")
    vf = vf.merge(villages[["village_id", "district_id", "name", "canal_reach",
                            "past_support_index", "lat", "lon"]],
                  on="village_id", how="left", suffixes=("", "_v"))
    vf["gdd_frac"] = (vf["gdd_total"] / 1650.0).clip(0.4, 1.3)         # vs rice maturity
    vf["water_deficit"] = (vf["dry_days_repro"] / 30.0).clip(0, 1)     # reproductive dry-day frac
    return vf
