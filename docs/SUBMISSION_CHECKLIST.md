# Submission Checklist — deadline July 6, 11:59 PM IST

## Build (do in this order)
- [ ] Local smoke test BOTH domains: `pipeline.run` + `agro.run` → open /grid AND /agro, click every tab
- [ ] GPU VM up (`infra/gpu_vm_setup.sh`) → **bench --domain grid AND --domain agro** cpu,gpu at 1e6,10e6,50e6 (+grid 100e6)
- [ ] `nvidia-smi`, bench totals, BOTH Acceleration tabs: **screenshots**
- [ ] Big artifacts GPU: `pipeline.run` on city_100m → api/data/ ; `agro.run` on agro_100m → api/data_agro/
- [ ] `bench/results.json` committed (has cpu AND gpu rows, both domains)
- [ ] BigQuery: `infra/bigquery_setup.sh` + `push_scores_bq.py` → Looker Studio report screenshot
- [ ] (stretch) Dataproc Spark RAPIDS backfill: Spark UI screenshot showing `Gpu*` operators
- [ ] Cloud Run deploy with GEMINI_API_KEY → public URL (test landing → /grid → /agro)
- [ ] **STOP THE GPU VM**

## The 5 mandatory items
- [ ] **PPT → PDF ≤ 5 MB**, in the official template (content: docs/PPT_CONTENT.md, numbers from YOUR bench)
- [ ] **Deployed link** — test on laptop + phone + incognito + a friend's machine. All tabs. Twice. Thrice.
- [ ] **GitHub public** — this repo, README renders, no secrets committed (`git log -p | grep -i key` = nothing)
- [ ] **Demo video ≤ 3:00** — script in docs/DEMO_SCRIPT.md; upload YouTube unlisted; test the link logged-out
- [ ] **Brief description** — paste the one-liner + 3 bullets from README

## Kill-shot questions to rehearse (judges will ask)
1. "Who uses this?" → Load dispatch engineer at a city DISCOM. By name. By 15-min block.
2. "Why GPU?" → NOT "it's faster". → "Because the pipeline fits inside the dispatch block,
   she decides on current data; and the planner searches 1000s of plans, so the plan is
   provably lower-pain. Here are the measured numbers."
3. "Is it one pipeline?" → GCS→BigQuery→RAPIDS on GCE→scores→BigQuery→Cloud Run+Gemini. One flow, shown live.
4. "Real data?" → Physics-informed synthetic city, schema-compatible with RDSS AMI extracts;
   every cleaning fix counted in-product. Swap-in is a config change.
5. "Business?" → 28 DISCOMs, 250M-meter AMI rollout, DT failure + fairness penalties are budgeted pain today.

## Known judge traps
- No "coming soon" anywhere — delete or finish.
- Video over 3:00 = penalty. Cut ruthlessly.
- Empty Acceleration tab = you skipped the GPU run. Do not skip it.
- Claiming speedup without `bench/results.json` in the repo = losing move.
