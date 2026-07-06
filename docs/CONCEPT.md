# VAJRA·GRID — Adaptive Energy Grid Intelligence (grid domain pack)

> Deep-dive for the **GRID** pack. Platform thesis (one GPU engine, two lifelines) is in the
> top-level `README.md`; the **AGRO** pack's research basis is in `docs/RESEARCH.md`. Both packs
> share `pipeline/engine.py`, `pipeline/io_util.py`, and the vectorized allocation pattern.

> **GPU-accelerated grid intelligence that turns 100M+ smart-meter signals into an explainable,
> fairness-aware load-shed decision — inside a single 15-minute dispatch block.**

---

## The One-Sentence Formula (Kazuki's template)

> **AEGIS helps a Load Dispatch Engineer at a city electricity distribution company (DISCOM)
> decide which feeders to shed and which distribution transformers to pre-emptively de-load
> during peak-demand emergencies, by analyzing 100M+ smart-meter (AMI) and SCADA readings,
> producing an explainable, impact-ranked, fairness-aware shed plan and a transformer failure
> watchlist — refreshed every dispatch block instead of once a day.**

---

## 1. The Real User

**Priya Sharma, Load Dispatch Engineer, City DISCOM Control Room (e.g., Ahmedabad, ~2M meters).**

Every 15 minutes ("dispatch block"), she must keep feeder and transformer loading inside limits.
On a 46°C heatwave afternoon, demand spikes past supply allocation and she has minutes to answer:

1. **Which feeders do I shed to recover N megawatts?**
2. **Which shed plan hurts customers the least?** (never hospitals/water; respect rotational fairness — don't hit the same neighbourhood that was cut yesterday)
3. **Which distribution transformers are about to fail** so crews de-load them *before* a
   3-day unplanned outage?

## 2. The Decision Bottleneck (today, without AEGIS)

- AMI telemetry: 2M meters × 96 intervals/day ≈ **192M readings/day**. The CPU ETL that
  aggregates it runs **overnight** — the control room sees **yesterday's** grid.
- Shed decisions are made from static Excel rosters + operator gut feel → hospitals protected by
  memory, fairness violations, repeat complaints, and avoidable transformer burnouts
  (a failed 100 kVA DT = days of outage + lakhs in replacement).
- Evaluating *one* alternative shed plan by hand takes ~5 minutes. Nobody evaluates fifty.

**The bottleneck: the data pipeline is slower than the decision cadence.** Decisions happen every
15 minutes; the pipeline delivers insight every 24 hours.

## 3. The Data

| Source | Contents | Scale |
|---|---|---|
| AMI smart-meter readings | meter_id, ts (15-min), kWh, voltage, PF — **messy**: duplicates, nulls, Wh/kWh unit errors, clock skew, orphan meters | 100M+ rows |
| Grid topology master | meters → transformers → feeders → substations, capacities, age, critical-facility flags | ~200K entities |
| SCADA/weather | temperature, humidity (AC load driver), feeder breaker states | 96/day |
| Outage & shed history | past shed hours per feeder (fairness), transformer failure history (training labels) | 90 days |

## 4. The Output (what Priya gets)

- **Shed Priority Ranking** — feeders ranked by Feeder Criticality Index (FCI) with per-feeder
  **reason codes** (e.g., `OVERLOAD-4H`, `TX-RISK×3`, `LOW-PAIN`, `FAIR-OK`) — explainable AI.
- **What-If Shed Planner** — "I need 40 MW relief" → GPU evaluates **thousands of candidate
  plans** in seconds → top plans compared on MW relief vs. customer pain vs. fairness.
- **Transformer Failure Watchlist** — XGBoost 72-h failure probability with SHAP reason codes.
- **4-hour feeder demand forecast** and headroom per feeder.
- **Gemini NL console** — "What can I shed to save 40 MW without touching hospital feeders?"

## 5. Acceleration = a better decision (the 4 proofs)

| Proof | Without GPU | With NVIDIA GPU |
|---|---|---|
| **Runtime** | Full pipeline on 100M rows: tens of minutes (pandas) | Seconds (cudf.pandas + XGBoost-GPU) — same code |
| **Scale** | CPU chokes / OOMs beyond ~10–20M rows | 100M+ rows in GPU memory, single node |
| **Freshness** | Insight is ~24 h old; pipeline can't fit in a 15-min block | Re-scored **every block**; dashboard shows live data age |
| **Decision** | 1 hand-evaluated shed plan per 5 min | **Thousands of plans/s** evaluated → provably lower-pain plan chosen |

**The money line:** *Because the GPU pipeline finishes in seconds instead of tens of minutes,
Priya sheds load using the current block's data and picks the plan with the least customer pain —
instead of copying yesterday's roster.*

## 6. Architecture (one connected pipeline)

```
GOOGLE CLOUD DATA LAYER
  Cloud Storage (raw AMI parquet landing)
      └─► BigQuery (raw → curated → scores datasets)
                │  BigQuery Storage Read API
                ▼
NVIDIA ACCELERATION LAYER  — GCE VM with NVIDIA L4/A100 GPU
  RAPIDS cudf.pandas  ─ clean ─ join ─ aggregate ─ rolling features
  XGBoost (device=cuda) ─ transformer-failure risk + demand forecast (+SHAP)
  GPU-vectorized what-if plan evaluator (CuPy)
  [stretch: Dataproc Managed Spark + Spark RAPIDS for 500M-row backfill]
                │ scores written back to BigQuery + GCS
                ▼
APPLICATION & DECISION LAYER
  Cloud Run: FastAPI decision API + AEGIS control-room dashboard (public URL)
  Vertex AI Gemini: function-calling agent (NL console, operator briefings)
  Looker Studio: BI report on BigQuery scores (optional embed)
```

**GCP services (5+):** Cloud Storage, BigQuery, GCE NVIDIA GPU VM, Cloud Run, Vertex AI (Gemini), (+ Dataproc Managed Spark, Looker Studio).
**NVIDIA (2–3):** RAPIDS cudf.pandas (+CuPy), NVIDIA GPUs on Google Cloud, XGBoost-GPU, (+ Spark RAPIDS stretch).

## 7. Why this is hard to copy

1. **Fairness-aware what-if optimizer** — GPU-vectorized evaluation of thousands of shed plans
   under hard constraints (critical facilities) and soft constraints (rotational fairness,
   pain minimization). Not a chart — a decision.
2. **Explainability everywhere** — SHAP reason codes on every risk score, component breakdown on
   every ranking, Gemini briefings grounded in tool calls (no hallucinated numbers).
3. **One codebase, two engines** — identical pipeline runs pandas (CPU) or cudf.pandas (GPU);
   the benchmark harness is built in, per-stage, reproducible by judges with one command.
4. **Physics-informed synthetic city** — temperature-driven load, diurnal class profiles,
   transformer aging model, realistic data mess. Judges can regenerate at any scale (1M → 500M).
5. **Batch + real-time** — nightly 100M-row backfill (Spark RAPIDS path) + per-block micro-batch
   rescoring (cudf path) = both modes, more points.
