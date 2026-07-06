#!/usr/bin/env bash
# AEGIS — provision the NVIDIA GPU box on Google Cloud and install RAPIDS.
# Produces the machine that generates your REAL GPU benchmark numbers.
#
# Usage:  PROJECT=my-proj ZONE=asia-south1-a bash infra/gpu_vm_setup.sh
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=your-gcp-project}"
ZONE="${ZONE:-asia-south1-a}"          # Mumbai; L4 also in us-central1-a etc.
VM="${VM:-aegis-gpu}"
MACHINE="${MACHINE:-g2-standard-8}"    # 1x NVIDIA L4 24GB — best price/perf for cudf
# For 100M+ rows in one shot use: MACHINE=a2-highgpu-1g (A100 40GB) + --accelerator below

gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="type=nvidia-l4,count=1" \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

echo "Waiting for boot + driver install (~3 min)…"; sleep 180

gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --command='
set -e
nvidia-smi
# RAPIDS via NVIDIA pip index (cudf.pandas + cuml + cupy), matching CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.6.*" "cuml-cu12==25.6.*" cupy-cuda12x
pip install xgboost pyarrow fastapi uvicorn scikit-learn google-genai \
    google-cloud-bigquery google-cloud-storage pandas-gbq
python -c "import cudf, cupy; print(\"RAPIDS OK:\", cudf.__version__)"
'

cat <<EOF

VM ready. Next:
  gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE
  git clone <your-repo> aegis && cd aegis
  # REAL GPU benchmarks (fills bench/results.json):
  python -m bench.run_bench --scales 1e6,10e6,50e6,100e6 --engines cpu,gpu
  # big-city artifacts for the demo:
  python -m data_gen.generate --meters 200000 --days 6 --out data/city_100m
  python -m pipeline.run --data data/city_100m --engine gpu --out out/gpu_run
Remember: gcloud compute instances stop $VM   # when done (GPU \$\$)
EOF
