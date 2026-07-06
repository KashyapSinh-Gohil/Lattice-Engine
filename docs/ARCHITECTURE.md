# AEGIS Architecture — technical deep-dive (for the code-review judge)

## Design principle

**One pipeline, one codebase, two engines.** All transform logic is written against the
pandas API. `pipeline/engine.py::activate("gpu")` calls `cudf.pandas.install()` *before any
pandas import*, so the identical code executes on NVIDIA RAPIDS. XGBoost flips
`device="cuda"`. The what-if optimizer takes an array module (`numpy` | `cupy`).
Benchmarking is therefore apples-to-apples by construction — the engine flag is the only
variable.

## Data flow (numbers at 100M-row city scale)

```
[GCS raw parquet: readings_*.parquet 100M rows, dims ~200K rows, weather]
   │  (BigQuery mirrors raw for SQL/BI; bq load in infra/bigquery_setup.sh)
   ▼
1_ingest        read chunked parquet                            (I/O bound)
2_clean         ts snap → dedupe → orphan join → unit fix →     groupby-transform median
                negative nulling → per-meter imputation          impute (GPU-heavy)
3_join_enrich   readings ⨝ meters ⨝ transformers ⨝ feeders      3 chained hash joins @100M
4_aggregate     → tx×ts (∑kw, sags, loading)                    2-level groupby @100M
                → feeder×ts (∑kw, capacity)
5_features      24h windows: loading max/mean, overload-min,     groupby aggs + rolling
                thermal ∑clip(l−0.8)², sag counts / feeder LF, volatility, utilization
6_risk_model    XGBoost cls on 90d tx history (327K rows) →      GPU hist trees
                p_fail_72h on live features + SHAP pred_contribs → reason codes
7_forecast      XGBoost reg, lag/temp/harmonic features →        16-step iterative,
                4h ahead, all feeders vectorized                 vectorized across fleet
8_score_rank    FCI = .35·forecast_stress + .30·tx_risk +        explainable components
                .20·utilization + .15·volatility  · pain = class mix + fairness + complaints
9_whatif_bench  K candidate plans (greedy + Bernoulli) →         (K×F)·(F) matmuls on
                relief/pain/fairness scoring → local search      CuPy; K up to 500K
10_write        JSON/parquet artifacts → api/data, BigQuery      (push_scores_bq.py)
```

Artifacts consumed by: Cloud Run FastAPI (dashboard + Copilot tools) and BigQuery
`aegis_scores.*` (Looker Studio).

## Why each GPU claim is honest

- Stages 2–5 are exactly the operations RAPIDS accelerates (joins, groupbys, window aggs,
  string keys). No exotic ops → minimal cudf.pandas fallback.
- Stage 6/7 use XGBoost `hist` on `device=cuda` — NVIDIA-maintained GPU path.
- Stage 9 is dense linear algebra on CuPy; throughput reported per run, per engine.
- `bench/run_bench.py` runs each engine in a **fresh subprocess** (clean cudf activation,
  no cache leakage), records per-stage wall clock, merges results across machines, and
  records CPU DNFs (timeout/OOM) as first-class data points.

## Failure-mode engineering (why the demo link can't die)

- Cloud Run image bundles a pre-computed artifact pack (`api/data/`) → the public URL
  works even with the GPU VM stopped and BigQuery unreachable.
- `DataStore.load()` walks `AEGIS_OUT → out/gpu_run → out/cpu_run → api/data`.
- Gemini agent falls back to a deterministic planner using the SAME tool functions —
  the Copilot answers even with no API key / quota exhausted (badge shows which engine).
- What-if runs in-process on NumPy on Cloud Run (still sub-second at city scale),
  and on CuPy when the API runs on the GPU VM — backend reported in the response.
- Every endpoint: CORS enabled, X-Process-Time header, /api/health for monitors.

## Security / hygiene

No secrets in repo; GEMINI_API_KEY via Cloud Run env. `.gitignore` excludes data/out.
Synthetic data only — zero PII. Protected-feeder exclusion enforced in the optimizer
(hard constraint), not in the UI.

## Scale path (the "10x more data" answer)

| Load | Change |
|---|---|
| 200M rows/day (2M meters) | same L4 VM, chunked parquet already supported |
| 1B+ rows backfill | `infra/spark_rapids/` Dataproc job (same logic in Spark SQL) |
| multi-city SaaS | per-tenant BigQuery datasets; Cloud Run scales horizontally; GPU VM → GKE + time-sliced L4s |
| streaming AMI | 15-min micro-batches land in GCS → Pub/Sub trigger → same pipeline (batch+RT both demonstrated) |
