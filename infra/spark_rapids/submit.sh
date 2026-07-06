#!/usr/bin/env bash
# Dataproc (Managed Spark) cluster with NVIDIA GPUs + Spark RAPIDS accelerator,
# then submit the AEGIS backfill job. STRETCH LAYER — batch path for 500M+ rows.
# Usage: PROJECT=my-proj BUCKET=my-bucket bash infra/spark_rapids/submit.sh
set -euo pipefail
PROJECT="${PROJECT:?}"; BUCKET="${BUCKET:?}"
REGION="${REGION:-us-central1}"; CLUSTER="${CLUSTER:-aegis-spark}"

gcloud dataproc clusters create "$CLUSTER" \
  --project "$PROJECT" --region "$REGION" \
  --image-version=2.2-ubuntu22 \
  --master-machine-type=n1-standard-8 \
  --num-workers=2 --worker-machine-type=n1-standard-16 \
  --worker-accelerator=type=nvidia-tesla-t4,count=1 \
  --initialization-actions="gs://goog-dataproc-initialization-actions-${REGION}/spark-rapids/spark-rapids.sh" \
  --properties="spark:spark.plugins=com.nvidia.spark.SQLPlugin,spark:spark.executor.resource.gpu.amount=1,spark:spark.task.resource.gpu.amount=0.125,spark:spark.rapids.sql.enabled=true" \
  --optional-components=JUPYTER --enable-component-gateway

gcloud dataproc jobs submit pyspark infra/spark_rapids/backfill_job.py \
  --project "$PROJECT" --region "$REGION" --cluster "$CLUSTER" \
  -- "gs://$BUCKET/raw/city_100m" "gs://$BUCKET/curated"

echo "Check Spark UI physical plan for Gpu* operators (proof of GPU execution)."
echo "Teardown: gcloud dataproc clusters delete $CLUSTER --region $REGION"
