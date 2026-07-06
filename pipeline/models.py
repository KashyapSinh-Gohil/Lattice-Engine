"""
AEGIS ML layer — stage 6/7.

* Transformer 72-hour failure risk: XGBoost classifier (device=cuda on GPU) trained on
  90 days of transformer history, scored on live 24h stress features, with native SHAP
  (pred_contribs) turned into human reason codes. Explainable by construction.
* Feeder demand forecast: XGBoost regressor, vectorized iterative multi-step prediction
  for the next 4 hours (16 x 15-min intervals) across all feeders at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:  # pragma: no cover — numpy fallback keeps everything runnable
    HAS_XGB = False

RISK_FEATURES = ["age_years", "capacity_kva", "loading_mean", "loading_max",
                 "overload_minutes", "sag_count", "thermal_stress"]

REASON_LABELS = {
    "age_years": "AGED-ASSET", "capacity_kva": "SMALL-DT", "loading_mean": "SUSTAINED-LOAD",
    "loading_max": "PEAK-OVERLOAD", "overload_minutes": "OVERLOAD-TIME",
    "sag_count": "VOLTAGE-SAGS", "thermal_stress": "THERMAL-STRESS",
}


class NumpyLogit:
    """Zero-dependency fallback risk model (standardized logistic regression).
    Production uses XGBoost; this keeps the product functional on any machine."""

    def fit(self, X, y, iters=400, lr=0.4):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1], dtype=np.float64)
        self.b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = Z.T @ (p - y) / len(y)
            self.w -= lr * g
            self.b -= lr * float((p - y).mean())
        return self

    def predict_proba(self, X):
        Z = (X - self.mu) / self.sd
        p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
        return np.column_stack([1 - p, p])

    def contribs(self, X):
        return ((X - self.mu) / self.sd) * self.w  # per-feature signed contribution


def _auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_risk_model(tx_history: pd.DataFrame, device: str = "cpu", fast: bool = False):
    X = tx_history[RISK_FEATURES].to_numpy(dtype=np.float32)
    y = tx_history["failed_next72h"].to_numpy(dtype=np.int8)
    n = len(y)
    idx = np.random.default_rng(7).permutation(n)
    cut = int(n * 0.85)
    tr, va = idx[:cut], idx[cut:]
    if HAS_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=120 if fast else 400, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist", device=device,
            eval_metric="auc", n_jobs=-1)
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        auc = float(clf.evals_result()["validation_0"]["auc"][-1])
    else:
        clf = NumpyLogit().fit(X[tr].astype(np.float64), y[tr].astype(np.float64))
        auc = _auc(y[va], clf.predict_proba(X[va].astype(np.float64))[:, 1])
    return clf, auc


def score_transformers(clf, txf: pd.DataFrame, transformers: pd.DataFrame) -> pd.DataFrame:
    """Score live features; attach top-3 SHAP reason codes per transformer."""
    f = txf.merge(transformers[["transformer_id", "age_years", "capacity_kva",
                                "lat", "lon"]], on="transformer_id", how="left")
    X = f[RISK_FEATURES].to_numpy(dtype=np.float32)
    f["p_fail_72h"] = clf.predict_proba(X)[:, 1]

    if HAS_XGB and isinstance(clf, xgb.XGBClassifier):
        booster = clf.get_booster()
        booster.set_param({"device": "cpu"})  # contribs on host; cheap vs training
        contribs = booster.predict(xgb.DMatrix(X, feature_names=RISK_FEATURES),
                                   pred_contribs=True)[:, :-1]  # drop bias term
    else:
        contribs = clf.contribs(X.astype(np.float64))
    order = np.argsort(-np.abs(contribs), axis=1)[:, :3]
    names = np.array(RISK_FEATURES)
    reasons, details = [], []
    for i in range(len(f)):
        codes, det = [], []
        for j in order[i]:
            if contribs[i, j] > 0.01:  # only factors pushing risk UP
                codes.append(REASON_LABELS[names[j]])
                det.append({"feature": str(names[j]), "shap": round(float(contribs[i, j]), 3),
                            "value": round(float(X[i, j]), 2)})
        reasons.append(codes)
        details.append(det)
    f["reason_codes"] = reasons
    f["shap_detail"] = details
    return f.sort_values("p_fail_72h", ascending=False)


# ---------------------------------------------------------------- demand forecast

def _pivot_load(fd_int: pd.DataFrame):
    """feeder x ts load matrix (small: ~100 feeders x few hundred intervals)."""
    pv = fd_int.pivot_table(index="feeder_id", columns="ts", values="kw", aggfunc="sum")
    pv = pv.sort_index(axis=1)
    return pv


def train_and_forecast(fd_int: pd.DataFrame, weather: pd.DataFrame,
                       horizon: int = 16, device: str = "cpu", fast: bool = False):
    """Train on history, then vectorized iterative multi-step forecast for all feeders."""
    pv = _pivot_load(fd_int)
    # to plain numpy on host (matrix is tiny; modeling logic identical on both engines)
    M = np.asarray(pv.to_numpy(dtype=np.float32))
    ts_cols = pd.DatetimeIndex(pv.columns)
    wx = weather.set_index("ts")["temp_c"].reindex(ts_cols).ffill().bfill()
    temp = np.asarray(wx.to_numpy(dtype=np.float32))
    F, T = M.shape
    lags = [1, 2, 3, 4, 96]
    max_lag = max(lags)
    if T <= max_lag + horizon:
        lags = [1, 2, 3, 4]
        max_lag = 4

    hod = (ts_cols.hour.values + ts_cols.minute.values / 60.0).astype(np.float32)

    def feats_at(t_idx: np.ndarray, hist: np.ndarray, temp_vec: np.ndarray,
                 hod_vec: np.ndarray):
        cols = [hist[:, t_idx - lag] for lag in lags]
        recent = np.mean(np.stack(cols[:4]), axis=0)
        s = np.sin(2 * np.pi * hod_vec[t_idx] / 24.0)
        c = np.cos(2 * np.pi * hod_vec[t_idx] / 24.0)
        rows = []
        for fi in range(hist.shape[0]):
            rows.append(np.column_stack(
                [c_[fi] for c_ in cols] + [recent[fi],
                 np.full(len(t_idx), fi, dtype=np.float32),
                 temp_vec[t_idx], s, c]))
        return np.concatenate(rows, axis=0)

    step = ts_cols[1] - ts_cols[0]
    fut_ts_all = [ts_cols[-1] + step * (h + 1) for h in range(horizon)]
    if not HAS_XGB:  # seasonal-naive fallback: same-time-yesterday x recent trend
        if T > 96 + 8:
            ratio = (M[:, -8:].mean(1, keepdims=True)
                     / (M[:, -104:-96].mean(1, keepdims=True) + 1e-6)).clip(0.6, 1.8)
            base = M[:, T - 96:T - 96 + horizon]
            fcM = (base * ratio).clip(min=0)
        else:
            fcM = np.repeat(M[:, -1:], horizon, axis=1)
        fc = pd.DataFrame(fcM, index=pv.index,
                          columns=[t.isoformat() for t in fut_ts_all])
        fut_temp = np.full(horizon, float(temp[-1]), dtype=np.float32)
        return fc, [t.isoformat() for t in fut_ts_all], fut_temp.tolist()

    # training set: every t in [max_lag, T)
    t_train = np.arange(max_lag, T)
    Xtr = feats_at(t_train, M, temp, hod)
    ytr = M[:, t_train].ravel()
    reg = xgb.XGBRegressor(
        n_estimators=150 if fast else 500, max_depth=7, learning_rate=0.08,
        subsample=0.9, tree_method="hist", device=device, n_jobs=-1)
    reg.fit(Xtr, ytr, verbose=False)

    # iterative forecast: extend matrix one step at a time (vectorized across feeders)
    hist = M.copy()
    step = ts_cols[1] - ts_cols[0]
    fut_ts = [ts_cols[-1] + step * (h + 1) for h in range(horizon)]
    fut_hod = np.array([(t.hour + t.minute / 60.0) for t in fut_ts], dtype=np.float32)
    # future temperature: same-time-yesterday + today's warming delta (heatwave persistence)
    if T > 96:
        delta = float(temp[-96:].mean() - temp[-192:-96].mean()) if T >= 192 else 0.0
        fut_temp = temp[-96:][:horizon] + delta
    else:
        fut_temp = np.full(horizon, float(temp[-1]), dtype=np.float32)

    hod_ext = np.concatenate([hod, fut_hod])
    temp_ext = np.concatenate([temp, fut_temp])
    for h in range(horizon):
        t_idx = np.array([T + h])  # hist has cols 0..T+h-1; lags reach back safely
        X = feats_at(t_idx, hist, temp_ext, hod_ext)
        yhat = reg.predict(X).astype(np.float32).reshape(F, 1).clip(min=0)
        hist = np.concatenate([hist, yhat], axis=1)

    fc = pd.DataFrame(hist[:, T:], index=pv.index,
                      columns=[t.isoformat() for t in fut_ts])
    return fc, [t.isoformat() for t in fut_ts], fut_temp.tolist()
