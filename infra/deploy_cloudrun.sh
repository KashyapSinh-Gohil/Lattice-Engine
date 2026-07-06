#!/usr/bin/env bash
# Deploy Lattice platform (unified single-page dashboard + decision API) to Cloud Run.
# Usage: PROJECT=my-proj REGION=asia-south1 GEMINI_API_KEY=... bash infra/deploy_cloudrun.sh
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-asia-south1}"
SVC="${SVC:-lattice}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

# Build Next.js static frontend
echo "Building Next.js frontend..."
(cd frontend && npm install && npm run build)

# Demo datasets for both domains
[ -f data/city_1m/meta.json ] || python -m data_gen.generate --meters 5200 --days 2 \
  --out data/city_1m --seed 7
[ -f api/data/system.json ] || { python -m pipeline.run --data data/city_1m \
  --engine cpu --out out/cpu_run --whatif-candidates 200000 && \
  mkdir -p api/data && cp out/cpu_run/*.json api/data/; }
[ -f data/agro_1m/meta.json ] || python -m agro.generate --plots 40000 --out data/agro_1m --seed 7
[ -f api/data_agro/system.json ] || { python -m agro.run --data data/agro_1m \
  --engine cpu --out out/agro_cpu --alloc-candidates 200000 && \
  mkdir -p api/data_agro && cp out/agro_cpu/*.json api/data_agro/; }

gcloud run deploy "$SVC" \
  --project "$PROJECT" --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 900 --concurrency 40 \
  --set-env-vars "GRID_OUT=/app/api/data,AGRO_OUT=/app/api/data_agro${GEMINI_API_KEY:+,GEMINI_API_KEY=$GEMINI_API_KEY}"

URL=$(gcloud run services describe "$SVC" --project "$PROJECT" --region "$REGION" \
      --format='value(status.url)')
echo
echo "LIVE: $URL   (single-page unified dashboard with domain toggle)"
echo "Smoke test:"
curl -s "$URL/api/health" && echo
echo "→ Open $URL and toggle between Grid and Agro domains."
