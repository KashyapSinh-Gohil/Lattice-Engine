"""
AGRO ML layer — stage 6/7 (mirrors grid models.py).

* Crop-loss risk: XGBoost classifier (device=cuda on GPU) trained on 5 seasons of village
  yield history (loss_event = yield < 0.8 x village normal, a PMFBY-style threshold), scored
  on this season's live village features, with native SHAP reason codes.
* Season-end yield forecast: XGBoost regressor on the same features → t/ha per village, and a
  shortfall-vs-normal that feeds the insurance-trigger score.

Falls back to a numpy logistic / ridge model when XGBoost isn't installed, so the product runs
anywhere (same design as the grid pack).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.models import HAS_XGB, NumpyLogit, _auc  # reuse shared fallback + metric

if HAS_XGB:
    import xgboost as xgb

# training-history feature names; live village features are mapped into this order below
RISK_FEATURES = ["peak_ndvi", "integ_ndvi", "water_stress", "gdd_frac", "pest_index"]
REASON_LABELS = {
    "peak_ndvi": "LOW-CANOPY", "integ_ndvi": "LOW-BIOMASS",
    "water_stress": "DRY-SPELL-REPRO", "gdd_frac": "HEAT-STRESS", "pest_index": "PEST-PRESSURE",
}


def train_yield_models(hist: pd.DataFrame, device: str = "cpu", fast: bool = False):
    """Returns (loss_clf, yield_reg, auc)."""
    Xc = hist[RISK_FEATURES].to_numpy(dtype=np.float32)
    yc = hist["loss_event"].to_numpy(dtype=np.int8)
    yr = hist["yield_t_ha"].to_numpy(dtype=np.float32)
    n = len(yc)
    idx = np.random.default_rng(7).permutation(n)
    cut = int(n * 0.85); tr, va = idx[:cut], idx[cut:]

    if HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=120 if fast else 350, max_depth=5, learning_rate=0.09,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist", device=device,
            eval_metric="auc", n_jobs=-1)
        clf.fit(Xc[tr], yc[tr], eval_set=[(Xc[va], yc[va])], verbose=False)
        auc = float(clf.evals_result()["validation_0"]["auc"][-1])
        reg = xgb.XGBRegressor(
            n_estimators=150 if fast else 400, max_depth=5, learning_rate=0.09,
            subsample=0.9, tree_method="hist", device=device, n_jobs=-1)
        reg.fit(Xc[tr], yr[tr], verbose=False)
    else:
        clf = NumpyLogit().fit(Xc[tr].astype(np.float64), yc[tr].astype(np.float64))
        auc = _auc(yc[va], clf.predict_proba(Xc[va].astype(np.float64))[:, 1])
        reg = _NumpyRidge().fit(Xc[tr].astype(np.float64), yr[tr].astype(np.float64))
    return clf, reg, auc


class _NumpyRidge:
    def fit(self, X, y, lam=1.0):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        A = Z.T @ Z + lam * np.eye(Z.shape[1]); A[0, 0] -= lam
        self.w = np.linalg.solve(A, Z.T @ y)
        return self

    def predict(self, X):
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return Z @ self.w


def score_villages(clf, reg, vf: pd.DataFrame) -> pd.DataFrame:
    """Score live village features: crop-loss probability (+SHAP), yield forecast, shortfall."""
    f = vf.copy()
    f["pest_index"] = 0.15  # live pest proxy (no live scouting feed in this demo)
    # map live features into the training feature order (water_deficit ↔ water_stress)
    X = np.column_stack([
        f["peak_ndvi"], f["integ_ndvi"], f["water_deficit"],
        f["gdd_frac"], f["pest_index"]]).astype(np.float32)

    f["p_loss"] = clf.predict_proba(X)[:, 1]
    f["yield_pred"] = np.clip(reg.predict(X.astype(np.float64) if not HAS_XGB else X), 0.2, None)

    # SHAP / signed contributions -> reason codes
    if HAS_XGB and isinstance(clf, xgb.XGBClassifier):
        booster = clf.get_booster(); booster.set_param({"device": "cpu"})
        contribs = booster.predict(xgb.DMatrix(X, feature_names=RISK_FEATURES),
                                   pred_contribs=True)[:, :-1]
    else:
        contribs = clf.contribs(X.astype(np.float64))
    order = np.argsort(-np.abs(contribs), axis=1)[:, :3]
    names = np.array(RISK_FEATURES)
    reasons, details = [], []
    for i in range(len(f)):
        codes, det = [], []
        for j in order[i]:
            if contribs[i, j] > 0.01:
                codes.append(REASON_LABELS[names[j]])
                det.append({"feature": str(names[j]), "shap": round(float(contribs[i, j]), 3),
                            "value": round(float(X[i, j]), 3)})
        reasons.append(codes); details.append(det)
    f["reason_codes"] = reasons
    f["shap_detail"] = details
    return f
