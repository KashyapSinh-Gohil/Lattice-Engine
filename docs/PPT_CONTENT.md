# PPT Content — VAJRA (pour into the official template, export PDF ≤5MB)

Fill `{X}` benchmark numbers from YOUR `bench/results.json` after the GPU run.

---

**Slide 1 — Title**
VAJRA — Vectorized Analytics for Joint Resilience & Allocation
*One GPU decision engine for two climate-stressed lifelines: power and food.*
Team {name} · Google Cloud × NVIDIA · Gen AI Academy APAC · PS2

**Slide 2 — The insight (why one platform)**
Grid load-shedding and smallholder drought response are the **same computational problem**:
under climate stress, allocate a **scarce resource** (megawatts / irrigation water) across
many **units** (feeders / villages), each with a **risk score** and a **fairness history**,
**faster than conditions change**. Build it once, ship two decision surfaces.
*(show the GRID↔AGRO mapping table from the README)*

**Slide 3 — Two real users, two real bottlenecks**
- GRID: Load Dispatch Engineer, 46 °C heatwave — AMI ETL runs **overnight on CPU** → decides on yesterday's grid.
- AGRO: District Agriculture Officer, monsoon dry spell — advice rides on **district bulletins lagging days–weeks** → reproductive window lost.
- Common bottleneck: **the data pipeline is slower than the decision cadence.**
- Scale: ~475 M smallholder farms, 74% in Asia (FAO); PMFBY = world's largest crop insurance (72.6 cr applications, ₹1.72 lakh cr claims). Citations in docs/RESEARCH.md.

**Slide 4 — What each user GETS (two dashboard screenshots)**
- GRID: FCI shed-priority ranking + reason codes, transformer 72-h failure watchlist (SHAP), 4-h forecast, shed-plan what-if, Gemini Copilot.
- AGRO: VAPI advisory ranking + reason codes (DRY-SPELL-REPRO, NDVI-DECLINE, TRIGGER-HIT), crop-loss/insurance-trigger watchlist (SHAP), yield forecast, irrigation allocator (tail-fairness), Gemini Copilot.

**Slide 5 — One connected pipeline (architecture diagram)**
GCS raw (AMI / Sentinel-2 NDVI) → BigQuery (raw/curated/scores) → **GCE + NVIDIA L4: RAPIDS
cudf.pandas clean/join/aggregate/features → XGBoost-GPU risk+forecast (+SHAP) → CuPy allocator**
→ scores back to BigQuery → Cloud Run (FastAPI + landing + both dashboards) + Vertex AI Gemini +
Looker. Stretch: Dataproc Spark RAPIDS 500M-row backfill.
*Shared core `pipeline/engine.py`; `--engine gpu` flips cudf.pandas + device=cuda for BOTH packs.*

**Slide 6 — Acceleration proof (4 types × 2 domains, REAL numbers)**
| Proof | GRID CPU→GPU | AGRO CPU→GPU |
|---|---|---|
| Runtime @ {N}M rows | {X}s → {Y}s (**{Z}×**) | {X}s → {Y}s (**{Z}×**) |
| Scale | DNF @ {X}M / GPU {Y}M | DNF @ {X}M / GPU {Y}M |
| Freshness | in 15-min block | in satellite-pass window |
| Decision | {N}k shed-plans/s | {N}k irrigation-plans/s |
*AGRO note: per-plot NDVI gap-fill is a groupby-interpolate — pandas is slow, cudf crushes it → biggest speedup lives here.*

**Slide 7 — Explainable & responsible**
FCI/VAPI component breakdowns; SHAP reason codes on every risk score; Copilot answers only from
tool results. GRID hard-excludes hospital/water/transit feeders. AGRO rewards under-served
tail-reach villages and flags PMFBY-style insurance triggers so payouts start early.
Data-quality ledger shown in-product (dedup / unit-fix / cloud-gap-fill counts).

**Slide 8 — Feasibility & market**
28 Indian state DISCOMs + 70+ city utilities (250M-meter AMI rollout under RDSS); ~475 M
smallholder farms across APAC + the PMFBY payout pipeline. Land-and-expand SaaS per feeder/village;
deploys on the customer's GCP project; scale by swapping L4→A100 with zero code change; the shared
core extends to a third lifeline (water / EV-charging).

**Slide 9 — Live demo + links**
Deployed: {Cloud Run URL} (landing → /grid, /agro) · GitHub: {repo} · Video: {link}
Demo flow = User → Pipeline → Acceleration → Decision, shown in BOTH domains.
