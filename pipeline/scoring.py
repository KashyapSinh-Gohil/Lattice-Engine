"""
AEGIS stage 8 — Feeder Criticality Index (FCI) + shed-pain scoring, fully explainable.

FCI answers: "which feeders need operator attention first?"
Pain answers: "if I shed this feeder, how much does it hurt?"
Every number ships with its component breakdown and reason codes — no black boxes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FCI_WEIGHTS = {"forecast_stress": 0.35, "tx_risk": 0.30, "utilization": 0.20,
               "volatility": 0.15}
CLASS_PAIN = {"residential": 0.35, "commercial": 0.50, "industrial": 0.65}


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 1e-9 else s * 0.0


def score_feeders(fdf: pd.DataFrame, txs: pd.DataFrame, feeders: pd.DataFrame,
                  meters: pd.DataFrame, transformers: pd.DataFrame,
                  shed_hist: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    # transformer risk rolled up to feeder
    tx_risk = (txs.groupby("feeder_id", observed=True)
               .agg(tx_risk_max=("p_fail_72h", "max"), tx_risk_mean=("p_fail_72h", "mean"),
                    tx_high_risk=("p_fail_72h", lambda s: int((s > 0.5).sum())))
               .reset_index())

    # forecast peak per feeder (next 4h)
    fc_peak = forecast.max(axis=1).rename("forecast_peak_kw").reset_index()

    # customer mix per feeder for pain scoring
    mtx = meters.merge(transformers[["transformer_id", "feeder_id"]],
                       on="transformer_id", how="left")
    mix = (mtx.assign(one=1)
           .pivot_table(index="feeder_id", columns="customer_class", values="one",
                        aggfunc="sum", fill_value=0).reset_index())
    for c in CLASS_PAIN:
        if c not in mix.columns:
            mix[c] = 0
    mix["customers"] = mix[list(CLASS_PAIN)].sum(axis=1)

    f = (fdf.merge(tx_risk, on="feeder_id", how="left")
            .merge(fc_peak, on="feeder_id", how="left")
            .merge(mix, on="feeder_id", how="left")
            .merge(shed_hist, on="feeder_id", how="left")
            .merge(feeders[["feeder_id", "name", "substation_id", "critical_type",
                            "is_protected", "lat", "lon"]], on="feeder_id", how="left"))
    f = f.fillna({"tx_risk_max": 0, "tx_risk_mean": 0, "tx_high_risk": 0,
                  "shed_hours_30d": 0, "complaints_30d": 0})

    # ---- FCI components (each normalized 0-1 across the fleet) ----
    f["forecast_stress"] = (f["forecast_peak_kw"] / f["capacity_kw"].clip(lower=1e-6))
    comp = pd.DataFrame({
        "forecast_stress": _norm(f["forecast_stress"]),
        "tx_risk": _norm(0.6 * f["tx_risk_max"] + 0.4 * f["tx_risk_mean"]),
        "utilization": _norm(f["utilization_now"]),
        "volatility": _norm(f["volatility"]),
    })
    f["fci"] = sum(FCI_WEIGHTS[k] * comp[k] for k in FCI_WEIGHTS).round(4)
    for k in comp:
        f[f"c_{k}"] = comp[k].round(4)

    # ---- shed pain (what it costs to shed this feeder) ----
    class_pain = sum(CLASS_PAIN[c] * f[c] for c in CLASS_PAIN) / f["customers"].clip(lower=1)
    fairness = _norm(f["shed_hours_30d"])         # recently-shed feeders hurt more to cut again
    complaints = _norm(f["complaints_30d"])
    f["pain_score"] = (0.45 * class_pain + 0.35 * fairness + 0.20 * complaints).round(4)
    f.loc[f["is_protected"], "pain_score"] = 9.99  # hospitals/water/transit: never shed
    f["sheddable_mw"] = (f["kw_now"] / 1000.0).round(3)
    f["headroom_mw"] = ((f["capacity_kw"] - f["forecast_peak_kw"]) / 1000.0).round(3)

    # ---- reason codes (explainability) ----
    codes = []
    for _, r in f.iterrows():
        c = []
        if r["forecast_stress"] > 0.95: c.append("OVERLOAD-4H")
        if r["tx_high_risk"] >= 1: c.append(f"TX-RISK×{int(r['tx_high_risk'])}")
        if r["utilization_now"] > 0.9: c.append("AT-LIMIT")
        if r["c_volatility"] > 0.7: c.append("VOLATILE")
        if r["is_protected"]: c.append(f"PROTECTED-{str(r['critical_type']).upper()}")
        elif r["shed_hours_30d"] > 8: c.append("RECENT-SHED")
        else: c.append("FAIR-OK")
        if r["sag_rate"] > 0.5: c.append("SAGS")
        codes.append(c)
    f["reason_codes"] = codes
    return f.sort_values("fci", ascending=False).reset_index(drop=True)


def system_summary(f: pd.DataFrame, fd_int: pd.DataFrame, forecast: pd.DataFrame,
                   weather: pd.DataFrame, quality: dict, risk_auc: float) -> dict:
    sys_ts = (fd_int.groupby("ts", observed=True)["kw"].sum() / 1000.0)
    sys_ts = sys_ts.sort_index()
    fc_sys = (forecast.sum(axis=0) / 1000.0)
    current_mw = float(sys_ts.iloc[-1])
    forecast_peak = float(fc_sys.max())
    capacity_mw = float(f["capacity_kw"].sum() / 1000.0)
    # emergency scenario: supply allocation is 88% of the forecast system peak
    supply_cap = round(0.88 * forecast_peak, 1)
    deficit = max(0.0, round(forecast_peak - supply_cap, 1))
    wshort = weather.tail(96 * 2)
    return {
        "current_mw": round(current_mw, 1), "forecast_peak_mw": round(forecast_peak, 1),
        "capacity_mw": round(capacity_mw, 1), "supply_cap_mw": supply_cap,
        "deficit_mw": deficit, "temp_now_c": float(weather["temp_c"].iloc[-1]),
        "feeders": int(len(f)), "protected_feeders": int(f["is_protected"].sum()),
        "tx_watchlist": int((f["tx_high_risk"] > 0).sum()), "risk_model_auc": round(risk_auc, 4),
        "data_quality": quality,
        "system_load": [{"ts": str(t), "mw": round(float(v), 2)}
                        for t, v in sys_ts.tail(96).items()],
        "system_forecast": [{"ts": str(t), "mw": round(float(v), 2)}
                            for t, v in fc_sys.items()],
        "temperature": [{"ts": str(r.ts), "c": float(r.temp_c)}
                        for r in wshort.itertuples()][::4],
    }
