"""
Engine activation — the heart of AEGIS's one-codebase/two-engines design.

`activate("gpu")` installs NVIDIA RAPIDS cudf.pandas BEFORE pandas is imported anywhere,
so every pandas operation in the pipeline transparently executes on the GPU.
`activate("cpu")` leaves stock pandas in place. Identical downstream code either way.

MUST be called before any module that imports pandas.
"""
from __future__ import annotations

import importlib
import platform


def activate(engine: str) -> dict:
    info = {"engine": engine, "python": platform.python_version(), "cudf": None,
            "gpu_name": None, "cupy": False}
    if engine == "gpu":
        import cudf.pandas  # noqa: PLC0415
        cudf.pandas.install()
        import cudf  # noqa: PLC0415
        info["cudf"] = cudf.__version__
        try:
            import cupy  # noqa: PLC0415
            info["cupy"] = True
            info["gpu_name"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:  # pragma: no cover
            pass
    import pandas as pd  # noqa: PLC0415
    info["pandas"] = pd.__version__
    return info


def array_module(engine: str):
    """CuPy on GPU, NumPy on CPU — for the vectorized what-if plan evaluator."""
    if engine == "gpu":
        try:
            return importlib.import_module("cupy")
        except ImportError:
            pass
    return importlib.import_module("numpy")


def xgb_device(engine: str) -> str:
    return "cuda" if engine == "gpu" else "cpu"
