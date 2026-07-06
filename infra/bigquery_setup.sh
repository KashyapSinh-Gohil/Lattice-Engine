#!/usr/bin/env bash
# AEGIS — Cloud Storage landing zone + BigQuery warehouse (raw -> curated -> scores).
# Usage: PROJECT=my-proj bash infra/bigquery_setup.sh [DATA_DIR]
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
LOC="${LOC:-asia-south1}"
BUCKET="${BUCKET:-${PROJECT}-aegis-landing}"
DATA_DIR="${1:-data/city_1m}"

# 1) Cloud Storage: raw landing zone
gsutil mb -p "$PROJECT" -l "$LOC" "gs://$BUCKET" 2>/dev/null || true
gsutil -m cp -r "$DATA_DIR"/*.parquet "gs://$BUCKET/raw/$(basename "$DATA_DIR")/"

# 2) BigQuery datasets
bq --project_id="$PROJECT" mk --location="$LOC" -d aegis_raw     2>/dev/null || true
bq --project_id="$PROJECT" mk --location="$LOC" -d aegis_curated 2>/dev/null || true
bq --project_id="$PROJECT" mk --location="$LOC" -d aegis_scores  2>/dev/null || true

# 3) Load raw parquet -> BigQuery (schema auto-detected from parquet)
for T in meters transformers feeders substations shed_history weather tx_history; do
  bq load --source_format=PARQUET --replace "aegis_raw.$T" \
    "gs://$BUCKET/raw/$(basename "$DATA_DIR")/$T.parquet"
done
bq load --source_format=PARQUET --replace aegis_raw.readings \
  "gs://$BUCKET/raw/$(basename "$DATA_DIR")/readings_*.parquet"

# 4) Curated view example (dedup happens on GPU; this is the SQL-side mirror for BI)
bq query --project_id="$PROJECT" --use_legacy_sql=false <<'SQL'
CREATE OR REPLACE VIEW aegis_curated.v_feeder_load AS
SELECT t.feeder_id, r.ts, SUM(r.kwh)*4 AS kw
FROM aegis_raw.readings r
JOIN aegis_raw.meters m USING(meter_id)
JOIN aegis_raw.transformers t USING(transformer_id)
WHERE r.kwh BETWEEN 0 AND 100
GROUP BY t.feeder_id, r.ts;
SQL

echo "BigQuery ready. After a pipeline run, push scores:"
echo "  python infra/push_scores_bq.py --out out/gpu_run --project $PROJECT"
echo "Point Looker Studio at aegis_scores.* for the BI layer."
