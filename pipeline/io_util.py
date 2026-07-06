"""
Table IO with graceful degradation: Parquet (production: GCS/BigQuery-native) when
pyarrow is available, transparent CSV fallback otherwise (zero-dependency quick start).
Same call sites either way; the pipeline never breaks on a missing wheel.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:  # pragma: no cover
    HAS_PARQUET = False

TS_COLS = ("ts",)


def save(df: pd.DataFrame, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.parquet" if HAS_PARQUET else f"{name}.csv")
    if HAS_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def _read(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    for c in TS_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load(data_dir: str, name: str) -> pd.DataFrame:
    for ext in ("parquet", "csv"):
        p = os.path.join(data_dir, f"{name}.{ext}")
        if os.path.exists(p):
            return _read(p)
    raise FileNotFoundError(f"{name}.(parquet|csv) not in {data_dir}")


def load_glob(data_dir: str, pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, pattern + ".parquet"))) or \
        sorted(glob.glob(os.path.join(data_dir, pattern + ".csv")))
    if not paths:
        raise FileNotFoundError(f"{pattern} not in {data_dir}")
    return pd.concat([_read(p) for p in paths], ignore_index=True)
