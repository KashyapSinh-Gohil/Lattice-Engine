"""
AEGIS benchmark harness — produces the four acceleration proofs with REAL numbers.

Runs the identical pipeline at multiple data scales on cpu and/or gpu engines
(each run is a fresh subprocess so cudf.pandas activation is clean), then merges
per-stage timings into bench/results.json. Run it on your laptop for CPU numbers and
once on the GCP GPU VM for GPU numbers — the JSON merges, the dashboard plots it.

  python -m bench.run_bench --scales 1e6,5e6 --engines cpu
  python -m bench.run_bench --scales 1e6,10e6,50e6,100e6 --engines cpu,gpu   # on GPU VM

Proofs derived:
  RUNTIME  — per-stage + total CPU vs GPU seconds
  SCALE    — total seconds vs rows (CPU DNF rows recorded as null)
  FRESHNESS— does the pipeline fit inside a 900s dispatch block?
  DECISION — what-if plans/second CPU vs GPU
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

INTERVALS_PER_DAY = 96
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")


def scale_to_params(rows: float) -> tuple[int, int]:
    """rows ~= meters * days * 96 (+3% mess). Prefer 2 days, grow meters."""
    days = 2
    meters = max(1000, int(rows / (days * INTERVALS_PER_DAY * 1.03)))
    return meters, days


DOMAINS = {
    "grid": {"gen": ("data_gen.generate", lambda s: ["--meters",
             str(max(1000, int(s / (2 * INTERVALS_PER_DAY * 1.03)))), "--days", "2"]),
             "run": "pipeline.run", "cand_flag": "--whatif-candidates",
             "rows_key": "readings_rows"},
    "agro": {"gen": ("agro.generate", lambda s: ["--plots",
             str(max(800, int(s / (24 * 1.026))))]),
             "run": "agro.run", "cand_flag": "--alloc-candidates",
             "rows_key": "ndvi_rows"},
}


def run(cmd: list[str], timeout: int | None = None) -> tuple[int, str]:
    print("  $", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:], sep="\n")
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1e6", help="comma list of target row counts")
    ap.add_argument("--engines", default="cpu")
    ap.add_argument("--data-root", default="data/bench")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--timeout", type=int, default=7200, help="per-run DNF cutoff (s)")
    ap.add_argument("--whatif-candidates", type=int, default=200000)
    ap.add_argument("--domain", choices=["grid", "agro"], default="grid")
    args = ap.parse_args()
    cfg = DOMAINS[args.domain]
    gen_module, gen_args = cfg["gen"]

    results = {"machine": {}, "runs": []}
    if os.path.exists(RESULTS):
        results = json.load(open(RESULTS))

    import platform
    results.setdefault("machine", {})[platform.node()] = {
        "platform": platform.platform(), "python": platform.python_version(),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for scale in [float(s) for s in args.scales.split(",")]:
        tag = (f"{int(scale/1e6)}m" if scale >= 1e6 else f"{int(scale/1e3)}k")
        tag = f"{args.domain}_{tag}"
        data_dir = os.path.join(args.data_root, tag)
        if not os.path.exists(os.path.join(data_dir, "meta.json")):
            print(f"[gen] {tag}")
            rc, _ = run([sys.executable, "-m", gen_module, *gen_args(scale), "--out", data_dir])
            if rc != 0:
                continue
        meta = json.load(open(os.path.join(data_dir, "meta.json")))

        for engine in args.engines.split(","):
            out_dir = f"out/bench_{tag}_{engine}"
            print(f"[run] domain={args.domain} scale={tag} engine={engine}")
            entry = {"scale_rows": meta[cfg["rows_key"]], "tag": tag, "engine": engine,
                     "domain": args.domain, "node": platform.node()}
            try:
                cmd = [sys.executable, "-m", cfg["run"], "--data", data_dir,
                       "--engine", engine, "--out", out_dir,
                       cfg["cand_flag"], str(args.whatif_candidates)]
                if args.fast:
                    cmd.append("--fast")
                rc, _ = run(cmd, timeout=args.timeout)
                if rc == 0:
                    t = json.load(open(os.path.join(out_dir, "timings.json")))
                    entry.update({"stages": t["stages"], "total_seconds": t["total_seconds"],
                                  "fits_in_block": t["fits_in_block"],
                                  "plans_per_second": t["whatif_plans_per_second"],
                                  "gpu_name": t["env"].get("gpu_name")})
                else:
                    entry.update({"total_seconds": None, "dnf": True, "reason": "error/OOM"})
            except subprocess.TimeoutExpired:
                entry.update({"total_seconds": None, "dnf": True,
                              "reason": f">{args.timeout}s timeout (DNF)"})
            # replace any previous run with same tag+engine+node
            results["runs"] = [r for r in results["runs"]
                               if not (r["tag"] == tag and r["engine"] == engine)]
            results["runs"].append(entry)
            with open(RESULTS, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"  -> total={entry.get('total_seconds')}s "
                  f"plans/s={entry.get('plans_per_second')}")

    # derived summary for the dashboard
    summarize(results)
    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"results -> {RESULTS}")


def summarize(results: dict):
    by = {}
    for r in results["runs"]:
        by.setdefault(r["tag"], {})[r["engine"]] = r
    pairs = []
    for tag, d in by.items():
        if "cpu" in d and "gpu" in d and d["cpu"].get("total_seconds") \
                and d["gpu"].get("total_seconds"):
            pairs.append({
                "tag": tag, "rows": d["cpu"]["scale_rows"],
                "cpu_s": d["cpu"]["total_seconds"], "gpu_s": d["gpu"]["total_seconds"],
                "speedup": round(d["cpu"]["total_seconds"] / d["gpu"]["total_seconds"], 1),
            })
    results["summary"] = {
        "pairs": sorted(pairs, key=lambda p: p["rows"]),
        "max_speedup": max([p["speedup"] for p in pairs], default=None),
    }


if __name__ == "__main__":
    main()
