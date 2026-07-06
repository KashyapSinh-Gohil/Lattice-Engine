"""
AEGIS pipeline stages 1-5: ingest -> clean -> join/enrich -> aggregate -> features.

Written in plain pandas API. Under `--engine gpu` the same code runs on NVIDIA RAPIDS
cudf.pandas — merges, groupbys, rolling windows and string ops execute on the GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import io_util

V_SAG = 207.0        # <0.9 pu on 230V
UNIT_ERR_KWH = 100.0  # a 15-min residential/commercial reading >100 kWh is a Wh-unit error
DIVERSITY_PF = 0.9    # kVA -> usable kW on a DT


def ingest(data_dir: str) -> dict:
    """Stage 1 — read raw parquet from the landing zone (GCS-mounted or local)."""
    d = {"readings": io_util.load_glob(data_dir, "readings_*")}
    for name in ["meters", "transformers", "feeders", "substations",
                 "shed_history", "weather", "tx_history"]:
        d[name] = io_util.load(data_dir, name)
    return d


def clean(readings: pd.DataFrame, meters: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stage 2 — raw AMI telemetry -> trustworthy telemetry. Every fix is counted."""
    q = {"rows_in": int(len(readings))}

    # clock skew: snap to the 15-min dispatch grid
    readings["ts"] = readings["ts"].dt.round("15min")

    # duplicate suppression (meter re-transmissions)
    before = len(readings)
    readings = readings.drop_duplicates(subset=["meter_id", "ts"], keep="first")
    q["duplicates_removed"] = int(before - len(readings))

    # orphan meters (not in master data)
    before = len(readings)
    readings = readings.merge(meters[["meter_id"]], on="meter_id", how="inner")
    q["orphans_removed"] = int(before - len(readings))

    # unit errors: Wh reported instead of kWh
    unit_mask = readings["kwh"] > UNIT_ERR_KWH
    q["unit_errors_fixed"] = int(unit_mask.sum())
    readings.loc[unit_mask, "kwh"] = readings.loc[unit_mask, "kwh"] / 1000.0

    # physically impossible negatives -> missing
    neg_mask = readings["kwh"] < 0
    q["negatives_nulled"] = int(neg_mask.sum())
    readings.loc[neg_mask, "kwh"] = np.nan

    # impute missing with per-meter median (heavy groupby-transform -> GPU shines)
    q["nulls_imputed"] = int(readings["kwh"].isna().sum())
    med = readings.groupby("meter_id")["kwh"].transform("median")
    readings["kwh"] = readings["kwh"].fillna(med).fillna(0.0)

    readings["is_sag"] = (readings["voltage"] < V_SAG).astype("int8")
    q["rows_clean"] = int(len(readings))
    return readings, q


def join_enrich(readings, meters, transformers, feeders) -> pd.DataFrame:
    """Stage 3 — the expensive joins: 100M readings x topology (GPU gold)."""
    r = readings.merge(
        meters[["meter_id", "transformer_id", "customer_class"]], on="meter_id", how="left")
    r = r.merge(
        transformers[["transformer_id", "feeder_id", "capacity_kva"]],
        on="transformer_id", how="left")
    r = r.merge(feeders[["feeder_id", "substation_id"]], on="feeder_id", how="left")
    r["kw"] = r["kwh"] * 4.0  # 15-min energy -> average power
    return r


def aggregate(enriched) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 4 — meter-level -> transformer-interval and feeder-interval load."""
    tx_int = (enriched.groupby(["transformer_id", "feeder_id", "capacity_kva", "ts"],
                               observed=True)
              .agg(kw=("kw", "sum"), sags=("is_sag", "sum"), n_meters=("meter_id", "count"))
              .reset_index())
    tx_int["loading"] = tx_int["kw"] / (tx_int["capacity_kva"] * DIVERSITY_PF)

    fd_int = (tx_int.groupby(["feeder_id", "ts"], observed=True)
              .agg(kw=("kw", "sum"), sags=("sags", "sum"),
                   capacity_kw=("capacity_kva", "sum"))
              .reset_index())
    fd_int["capacity_kw"] = fd_int["capacity_kw"] * DIVERSITY_PF
    return tx_int, fd_int


def tx_features(tx_int) -> pd.DataFrame:
    """Stage 5a — per-transformer 24h stress features (same names as training history)."""
    last_ts = tx_int["ts"].max()
    w = tx_int[tx_int["ts"] > last_ts - pd.Timedelta(hours=24)].copy()
    w["overload"] = (w["loading"] > 1.0).astype("int8") * 15.0
    w["thermal"] = (w["loading"] - 0.8).clip(lower=0) ** 2
    f = (w.groupby(["transformer_id", "feeder_id"], observed=True)
         .agg(loading_mean=("loading", "mean"), loading_max=("loading", "max"),
              overload_minutes=("overload", "sum"), sag_count=("sags", "sum"),
              thermal_stress=("thermal", "sum"), kw_now=("kw", "last"))
         .reset_index())
    return f


def feeder_features(fd_int) -> pd.DataFrame:
    """Stage 5b — per-feeder profile features incl. rolling stats (GPU rolling windows)."""
    fd_int = fd_int.sort_values(["feeder_id", "ts"])
    g = fd_int.groupby("feeder_id", observed=True)["kw"]
    fd_int["roll_mean_2h"] = g.transform(lambda s: s.rolling(8, min_periods=1).mean())
    last_ts = fd_int["ts"].max()
    w24 = fd_int[fd_int["ts"] > last_ts - pd.Timedelta(hours=24)]
    f = (w24.groupby("feeder_id", observed=True)
         .agg(peak_kw_24h=("kw", "max"), mean_kw_24h=("kw", "mean"),
              std_kw_24h=("kw", "std"), sag_rate=("sags", "mean"),
              capacity_kw=("capacity_kw", "max"), kw_now=("kw", "last"))
         .reset_index())
    f["load_factor"] = f["mean_kw_24h"] / f["peak_kw_24h"].clip(lower=1e-6)
    f["volatility"] = (f["std_kw_24h"] / f["mean_kw_24h"].clip(lower=1e-6)).fillna(0)
    f["utilization_now"] = f["kw_now"] / f["capacity_kw"].clip(lower=1e-6)
    return f
