# GPU Runbook — produce your REAL benchmark numbers (60–90 min total)

Everything below runs on your GCP project. The CPU side already works anywhere.
**Never present estimated numbers. Run this, screenshot it, present that.**

## 1. Provision the GPU VM (~10 min)

```bash
PROJECT=<your-project> ZONE=asia-south1-a bash infra/gpu_vm_setup.sh
```

If L4 quota is missing: request `NVIDIA_L4_GPUS` quota (usually minutes), or use
`ZONE=us-central1-a`. A100 alternative: `MACHINE=a2-highgpu-1g` + accelerator `nvidia-tesla-a100`.

## 2. Clone repo on the VM and verify RAPIDS

```bash
gcloud compute ssh aegis-gpu --zone asia-south1-a
git clone https://github.com/<you>/aegis && cd aegis
python -c "import cudf; print(cudf.__version__)"
nvidia-smi   # screenshot #1 — the GPU exists
```

## 3. Run the benchmark matrix (the core evidence)

```bash
# GRID — generates each scale once, runs cpu + gpu
python -m bench.run_bench --domain grid --scales 1e6,10e6,50e6 --engines cpu,gpu
python -m bench.run_bench --domain grid --scales 100e6 --engines cpu,gpu --timeout 3600
# AGRO — per-plot NDVI gap-fill is a groupby-interpolate; CPU is slow → biggest speedup lives here
python -m bench.run_bench --domain agro --scales 1e6,10e6,50e6 --engines cpu,gpu
```

Both domains write into the same `bench/results.json` (each run tagged with `domain`); each
dashboard's Acceleration tab plots only its own domain.
Screenshot #2: terminal totals. Screenshot #3: each dashboard's Acceleration tab.

## 4. Produce the big artifacts for the deployed demo (both domains)

```bash
# GRID
python -m data_gen.generate --meters 200000 --days 6 --out data/city_100m
python -m pipeline.run --data data/city_100m --engine gpu --out out/gpu_run
# AGRO
python -m agro.generate --plots 4000000 --out data/agro_100m       # ~96M NDVI rows
python -m agro.run --data data/agro_100m --engine gpu --out out/agro_gpu
```

## 5. Ship artifacts + benchmarks into the Cloud Run demo packs

```bash
mkdir -p api/data api/data_agro
cp out/gpu_run/*.json api/data/ && cp out/agro_gpu/*.json api/data_agro/
git add api/data api/data_agro bench/results.json
git commit -m "real GPU artifacts + benchmarks (grid+agro)" && git push
```

## 6. Deploy the public link (from any machine)

```bash
PROJECT=<proj> REGION=asia-south1 GEMINI_API_KEY=<key> bash infra/deploy_cloudrun.sh
```

## 7. BigQuery + Looker (architecture completeness, ~15 min)

```bash
PROJECT=<proj> bash infra/bigquery_setup.sh data/city_1m           # GCS + BigQuery raw (grid)
python infra/push_scores_bq.py --out out/gpu_run --project <proj>  # scores → BigQuery
# Looker Studio: new report → BigQuery → aegis_scores.feeder_scores (screenshot #4)
# (repeat bigquery_setup.sh with data/agro_1m for the agro warehouse tables)
```

## 8. STOP THE GPU VM

```bash
gcloud compute instances stop aegis-gpu --zone asia-south1-a
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cudf` import error | CUDA mismatch — use the DLVM image in the script (cu124) and `cudf-cu12` |
| GPU OOM at 100M rows | use A100 40GB, or `--scales 50e6`; cudf spills with `CUDF_SPILL=on` env |
| CPU run too slow at 50M | that IS the scale proof — let the timeout record a DNF |
| Cloud Run cold start | keep artifacts in image (Dockerfile already copies api/data) |
