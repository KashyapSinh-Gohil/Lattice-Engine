"""
Lattice Decision API — one FastAPI service, two domains (GRID + AGRO), on Cloud Run.

Serves the platform landing page + both control-room dashboards + JSON endpoints over the
latest pipeline artifacts for each domain. Every response carries X-Process-Time. The live
what-if optimizer / irrigation allocator runs in-process (CuPy on the GPU VM, NumPy on Cloud
Run) so the deployed demo always works even when the GPU box is off — GPU benchmark numbers
come from bench/results.json.

Run local:  uvicorn api.main:app --port 8080   (from repo root)
Grid artifacts:  $GRID_OUT | out/gpu_run | out/cpu_run | api/data
Agro artifacts:  $AGRO_OUT | out/agro_gpu | out/agro_cpu | api/data_agro
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent import gemini_agent  # noqa: E402
from agro import allocate as allocate_mod  # noqa: E402
from pipeline import whatif as whatif_mod  # noqa: E402

BENCH_PATH = os.path.join(ROOT, "bench", "results.json")


def _gpu_xp():
    try:
        import cupy  # noqa: PLC0415
        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:
        return np


class GridStore:
    """Latest GRID pipeline artifacts, hot-reloadable."""
    domain = "grid"

    def __init__(self):
        self.dir = None
        self.feeders: list = []
        self.transformers: list = []
        self.system: dict = {}
        self.feeder_state: list = []
        self.whatif_bench: dict = {}
        self.loaded_at = 0.0
        self.pipeline_running = False

    def candidates(self):
        return [p for p in [os.environ.get("GRID_OUT"), os.environ.get("AEGIS_OUT"),
                            os.path.join(ROOT, "out", "gpu_run"),
                            os.path.join(ROOT, "out", "cpu_run"),
                            os.path.join(ROOT, "api", "data")] if p]

    def load(self) -> bool:
        for d in self.candidates():
            if os.path.exists(os.path.join(d, "feeders.json")):
                try:
                    self.feeders = json.load(open(os.path.join(d, "feeders.json")))
                    self.transformers = json.load(open(os.path.join(d, "transformers.json")))
                    self.system = json.load(open(os.path.join(d, "system.json")))
                    self.feeder_state = json.load(open(os.path.join(d, "feeder_state.json")))
                    self.whatif_bench = json.load(open(os.path.join(d, "whatif_bench.json")))
                    self.dir = d; self.loaded_at = time.time()
                    return True
                except Exception:
                    continue
        return False

    def timings(self) -> dict:
        p = os.path.join(self.dir or "", "timings.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def whatif(self, target_mw: float, n_candidates: int = 50000) -> dict:
        return whatif_mod.evaluate(self.feeder_state, target_mw, n_candidates=n_candidates,
                                   xp=_gpu_xp(), seed=int(time.time()))

    def benchmarks(self):
        return json.load(open(BENCH_PATH)) if os.path.exists(BENCH_PATH) else {"runs": []}


class AgroStore:
    """Latest AGRO pipeline artifacts, hot-reloadable."""
    domain = "agro"

    def __init__(self):
        self.dir = None
        self.villages: list = []
        self.triggers: list = []
        self.system: dict = {}
        self.village_state: list = []
        self.allocate_bench: dict = {}
        self.loaded_at = 0.0
        self.pipeline_running = False

    def candidates(self):
        return [p for p in [os.environ.get("AGRO_OUT"),
                            os.path.join(ROOT, "out", "agro_gpu"),
                            os.path.join(ROOT, "out", "agro_cpu"),
                            os.path.join(ROOT, "api", "data_agro")] if p]

    def load(self) -> bool:
        for d in self.candidates():
            if os.path.exists(os.path.join(d, "villages.json")):
                try:
                    self.villages = json.load(open(os.path.join(d, "villages.json")))
                    self.system = json.load(open(os.path.join(d, "system.json")))
                    self.village_state = json.load(open(os.path.join(d, "village_state.json")))
                    self.allocate_bench = json.load(open(os.path.join(d, "allocate_bench.json")))
                    tp = os.path.join(d, "triggers.json")
                    self.triggers = json.load(open(tp)) if os.path.exists(tp) else []
                    self.dir = d; self.loaded_at = time.time()
                    return True
                except Exception:
                    continue
        return False

    def timings(self) -> dict:
        p = os.path.join(self.dir or "", "timings.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def allocate(self, budget_ml: float, n_candidates: int = 50000) -> dict:
        return allocate_mod.allocate(self.village_state, budget_ml, n_candidates=n_candidates,
                                     xp=_gpu_xp(), seed=int(time.time()))

    def benchmarks(self):
        return json.load(open(BENCH_PATH)) if os.path.exists(BENCH_PATH) else {"runs": []}


grid = GridStore()
agro = AgroStore()
app = FastAPI(title="Lattice Decision API", version="2.0",
              description="GPU-accelerated resource allocation engine for grid + agriculture")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.middleware("http")
async def add_process_time(request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Time"] = f"{(time.perf_counter() - t0)*1000:.1f}ms"
    return resp


@app.on_event("startup")
def _startup():
    g, a = grid.load(), agro.load()
    print(f"[Lattice] grid={g} ({grid.dir}) | agro={a} ({agro.dir})")


class WhatIfReq(BaseModel):
    target_mw: float
    n_candidates: int = 50000


class AllocReq(BaseModel):
    budget_ml: float
    n_candidates: int = 50000


class AgentReq(BaseModel):
    message: str
    history: list = []
    domain: str = "grid"


class PipelineReq(BaseModel):
    engine: str = "cpu"
    domain: str = "grid"
    data: str | None = None


# ---------------------------------------------------------------- platform
@app.get("/api/domains")
def domains():
    return {"domains": [
        {"id": "grid", "name": "Lattice Grid", "loaded": bool(grid.system),
         "engine": grid.timings().get("engine"), "units": len(grid.feeders),
         "unit_label": "feeders"},
        {"id": "agro", "name": "Lattice Agro", "loaded": bool(agro.system),
         "engine": agro.timings().get("engine"), "units": len(agro.villages),
         "unit_label": "villages"}]}


@app.get("/api/health")
def health():
    return {"status": "ok",
            "grid": {"loaded": bool(grid.system), "dir": grid.dir, "units": len(grid.feeders)},
            "agro": {"loaded": bool(agro.system), "dir": agro.dir, "units": len(agro.villages)}}


@app.get("/api/benchmarks")
def benchmarks():
    return grid.benchmarks()


# ---------------------------------------------------------------- GRID
@app.get("/api/summary")
def grid_summary():
    if not grid.system:
        raise HTTPException(503, "no grid artifacts — run pipeline.run")
    return {"system": grid.system, "timings": grid.timings(),
            "freshness_seconds": round(time.time() - grid.loaded_at, 1),
            "whatif_bench": grid.whatif_bench}


@app.get("/api/feeders")
def feeders(limit: int = 200):
    return grid.feeders[:limit]


@app.get("/api/feeders/{feeder_id}")
def feeder_detail(feeder_id: str):
    for f in grid.feeders:
        if f["feeder_id"] == feeder_id:
            return f
    raise HTTPException(404, "feeder not found")


@app.get("/api/transformers")
def transformers(limit: int = 100):
    return grid.transformers[:limit]


@app.post("/api/whatif")
def whatif(req: WhatIfReq):
    if not grid.feeder_state:
        raise HTTPException(503, "no grid artifacts")
    return grid.whatif(req.target_mw, min(req.n_candidates, 500000))


# ---------------------------------------------------------------- AGRO
@app.get("/api/agro/summary")
def agro_summary():
    if not agro.system:
        raise HTTPException(503, "no agro artifacts — run agro.run")
    return {"system": agro.system, "timings": agro.timings(),
            "freshness_seconds": round(time.time() - agro.loaded_at, 1),
            "allocate_bench": agro.allocate_bench}


@app.get("/api/agro/villages")
def villages(limit: int = 300):
    return agro.villages[:limit]


@app.get("/api/agro/villages/{village_id}")
def village_detail(village_id: str):
    for v in agro.villages:
        if v["village_id"] == village_id:
            return v
    raise HTTPException(404, "village not found")


@app.get("/api/agro/triggers")
def triggers(limit: int = 300):
    return agro.triggers[:limit]


@app.post("/api/agro/allocate")
def agro_allocate(req: AllocReq):
    if not agro.village_state:
        raise HTTPException(503, "no agro artifacts")
    return agro.allocate(req.budget_ml, min(req.n_candidates, 500000))


# ---------------------------------------------------------------- agent (both)
@app.post("/api/agent")
def agent(req: AgentReq):
    store = agro if req.domain == "agro" else grid
    if not store.system:
        raise HTTPException(503, f"no {req.domain} artifacts")
    return gemini_agent.chat(store, req.message, req.history, domain=req.domain)


# ---------------------------------------------------------------- live re-run
@app.post("/api/pipeline/run")
def pipeline_run(req: PipelineReq):
    store = agro if req.domain == "agro" else grid
    if store.pipeline_running:
        return JSONResponse({"status": "already_running"}, status_code=409)
    module = "agro.run" if req.domain == "agro" else "pipeline.run"
    data = req.data or ("data/agro_1m" if req.domain == "agro" else "data/city_1m")
    if not os.path.exists(os.path.join(ROOT, data, "meta.json")):
        raise HTTPException(400, f"dataset {data} not found on this instance")

    def _run():
        store.pipeline_running = True
        try:
            out = os.path.join(ROOT, "out", f"{req.domain}_{req.engine}_live")
            subprocess.run([sys.executable, "-m", module, "--data", data,
                            "--engine", req.engine, "--out", out, "--fast"],
                           cwd=ROOT, timeout=3600, check=False)
            os.environ["AGRO_OUT" if req.domain == "agro" else "GRID_OUT"] = out
            store.load()
        finally:
            store.pipeline_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "engine": req.engine, "domain": req.domain}


@app.get("/api/pipeline/status")
def pipeline_status(domain: str = "grid"):
    store = agro if domain == "agro" else grid
    return {"running": store.pipeline_running, "loaded_at": store.loaded_at,
            "engine_of_artifacts": store.timings().get("engine")}


# ---- static single-page dashboard ----
web_dir = os.path.join(ROOT, "web")
if os.path.isdir(web_dir):
    app.mount("/_next", StaticFiles(directory=os.path.join(web_dir, "_next")), name="next_static")
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def home():
        return FileResponse(os.path.join(web_dir, "index.html"))

    @app.get("/{filename}")
    def get_static_file(filename: str):
        path = os.path.join(web_dir, filename)
        if os.path.isfile(path):
            return FileResponse(path)
        raise HTTPException(status_code=404, detail="Not Found")
