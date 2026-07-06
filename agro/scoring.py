"""
AGRO stage 8 — Village Advisory Priority Index (VAPI) + insurance-trigger score.

VAPI answers: "which villages get the scarce extension visit / advisory this week?"
Insurance trigger answers: "which villages crossed a payout threshold — start relief now?"
Both fully explainable (component breakdown + reason codes), mirroring the grid FCI/pain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VAPI_WEIGHTS = {"loss_risk": 0.35, "yield_shortfall": 0.30, "ndvi_decline": 0.20,
                "exposure": 0.15}


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 1e-9 else s * 0.0


def score_villages_priority(scored: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    f = scored.copy()
    normal = (hist.groupby("village_id")["normal_yield"].mean()
              .rename("normal_yield").reset_index())
    f = f.merge(normal, on="village_id", how="left")
    f["normal_yield"] = f["normal_yield"].fillna(f["yield_pred"].median())
    f["yield_shortfall_frac"] = ((f["normal_yield"] - f["yield_pred"])
                                 / f["normal_yield"].clip(lower=1e-6)).clip(-0.5, 1.0)

    comp = pd.DataFrame({
        "loss_risk": _norm(f["p_loss"]),
        "yield_shortfall": _norm(f["yield_shortfall_frac"].clip(lower=0)),
        "ndvi_decline": _norm((-f["ndvi_trend"]).clip(lower=0)),  # decline = distress
        "exposure": _norm(f["rainfed_frac"] * np.log1p(f["area_ha"])),
    })
    f["vapi"] = sum(VAPI_WEIGHTS[k] * comp[k] for k in VAPI_WEIGHTS).round(4)
    for k in comp:
        f[f"c_{k}"] = comp[k].round(4)

    # insurance trigger: PMFBY-style — yield < 0.8x normal OR severe reproductive dry spell
    f["yield_trigger"] = (f["yield_pred"] < 0.8 * f["normal_yield"]).astype(int)
    f["rain_trigger"] = (f["water_deficit"] > 0.5).astype(int)
    f["insurance_trigger"] = ((f["yield_trigger"] == 1) | (f["rain_trigger"] == 1)).astype(int)
    f["trigger_score"] = (0.6 * f["yield_shortfall_frac"].clip(lower=0)
                          + 0.4 * f["water_deficit"]).round(4)

    # how much yield a timely intervention can rescue (benefit for the allocator)
    f["yield_saveable_t"] = (f["p_loss"] * f["water_deficit"].clip(lower=0.05)
                             * f["yield_pred"] * f["area_ha"] * 0.35).round(3)
    f["water_need_ml"] = (f["area_ha"] * 0.06 * (0.5 + f["water_deficit"])).round(3)  # megalitres

    codes = []
    for _, r in f.iterrows():
        c = list(r["reason_codes"])  # model SHAP codes
        if r["insurance_trigger"]: c.append("TRIGGER-HIT")
        if r["yield_shortfall_frac"] > 0.25: c.append("YIELD-SHORT")
        if r["ndvi_trend"] < -0.05: c.append("NDVI-DECLINE")
        if r["canal_reach"] == "tail" and r["past_support_index"] < 0.4: c.append("TAIL-UNDERSERVED")
        elif r["past_support_index"] > 0.7: c.append("WELL-SERVED")
        else: c.append("FAIR-OK")
        codes.append(c)
    f["reason_codes_full"] = codes
    return f.sort_values("vapi", ascending=False).reset_index(drop=True)


def system_summary(f: pd.DataFrame, vil_date: pd.DataFrame, weather: pd.DataFrame,
                   quality: dict, auc: float) -> dict:
    ndvi_ts = vil_date.groupby("date")["ndvi"].mean().sort_index()
    rain_ts = weather.groupby("date")["rain_mm"].mean().sort_index()
    total_area = float(f["area_ha"].sum())
    at_risk_area = float(f.loc[f["p_loss"] > 0.5, "area_ha"].sum())
    return {
        "domain": "agro",
        "villages": int(len(f)), "plots_total": int(f["plots"].sum()),
        "total_area_ha": round(total_area, 1),
        "at_risk_area_ha": round(at_risk_area, 1),
        "at_risk_pct": round(100 * at_risk_area / max(total_area, 1e-6), 1),
        "insurance_triggers": int(f["insurance_trigger"].sum()),
        "mean_yield_pred": round(float(f["yield_pred"].mean()), 2),
        "mean_normal_yield": round(float(f["normal_yield"].mean()), 2),
        "yield_saveable_total_t": round(float(f["yield_saveable_t"].sum()), 1),
        "water_need_total_ml": round(float(f["water_need_ml"].sum()), 1),
        "tail_villages": int((f["canal_reach"] == "tail").sum()),
        "risk_model_auc": round(auc, 4),
        "data_quality": quality,
        "ndvi_curve": [{"date": str(d)[:10], "ndvi": round(float(v), 3)}
                       for d, v in ndvi_ts.items()],
        "rain_curve": [{"date": str(d)[:10], "rain": round(float(v), 2)}
                       for d, v in rain_ts.items()][::3],
    }
