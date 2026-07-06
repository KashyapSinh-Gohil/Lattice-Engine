# Lattice

**Resource Allocation Engine**

> One engine for two climate-stressed lifelines: power and food.

Lattice is a unified decision platform that solves the same core problem across two domains:
under climate stress, a scarce resource must be allocated across many units, each carrying
a risk score and a fairness history. It turns large-scale sensor data into explainable,
fairness-aware allocation plans refreshed inside the operating window.

---

## Architecture

```
Synthetic Telemetry ──→ pandas pipeline ──→ XGBoost risk models ──→ Decision API
                                                                      │
                                     Single-page dashboard (Next.js) ←┘
```

| Component | Technology |
|-----------|-----------|
| Data generation | Python synthetic generators with real Gujarat geometry |
| Feature pipeline | pandas (CPU) |
| Risk models | XGBoost |
| Decisioning | Vectorized what-if evaluator (500k plans/second) |
| API | FastAPI with hot-reload artifacts |
| Frontend | Next.js 16 static export, Leaflet maps, Chart.js |
| Infrastructure | Google Cloud Run, Cloud Storage, Vertex AI |

## Domains

### Power Grid
- **Feeder Criticality Index (FCI)**: Composite risk score per feeder from load, temperature, sag events, and transformer age
- **Transformer Failure Prediction**: XGBoost model predicting 72h failure probability with SHAP explanations
- **Load Shedding What-If**: Evaluate 500,000+ candidate shed plans per second with fairness constraints
- **Protected Feeders**: Hospital, water treatment, and transit lines are automatically excluded

### Agriculture
- **Village Advisory Priority Index (VAPI)**: Composite score from NDVI deviation, rainfall deficit, soil moisture, and canal reach position
- **Insurance Trigger Detection**: PMFBY-style triggers based on NDVI drop below seasonal thresholds
- **Water Allocation**: Fairness-aware canal water allocation with tail-reach bonus
- **Yield Forecasting**: XGBoost prediction of per-village yield vs normal

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Generate synthetic datasets
python -m data_gen.generate --meters 5200 --days 2 --out data/city_1m --seed 7
python -m agro.generate --plots 40000 --out data/agro_1m --seed 7

# Run pipelines (CPU mode)
python -m pipeline.run --data data/city_1m --engine cpu --out out/cpu_run --whatif-candidates 200000
python -m agro.run --data data/agro_1m --engine cpu --out out/agro_cpu --alloc-candidates 200000

# Copy artifacts for the API
mkdir -p api/data api/data_agro
cp out/cpu_run/*.json api/data/
cp out/agro_cpu/*.json api/data_agro/

# Build the frontend
cd frontend && npm install && npm run build && cd ..

# Start the API
uvicorn api.main:app --reload --port 8080
```

Open [http://localhost:8080](http://localhost:8080) — single-page dashboard with domain toggle.

### Cloud Run Deployment

```bash
PROJECT=your-project REGION=asia-south1 bash infra/deploy_cloudrun.sh
```



## Project Structure

```
├── api/              # FastAPI backend
│   ├── main.py       # Routes, stores, middleware
│   ├── data/         # Grid pipeline artifacts (JSON)
│   └── data_agro/    # Agro pipeline artifacts (JSON)
├── pipeline/         # Grid feature pipeline + risk models
├── agro/             # Agriculture pipeline + risk models
├── agent/            # Gemini Copilot + security audit
├── data_gen/         # Synthetic data generators
├── bench/            # CPU benchmarking notebooks + results
├── frontend/         # Next.js 16 single-page dashboard
│   └── src/app/
│       └── page.tsx  # Unified dashboard (Grid + Agro)
├── web/              # Static export output (generated)
├── infra/            # Cloud Run deployment scripts
└── Dockerfile
```

## Data Policy

- All data is synthetically generated — no real consumer or farm PII
- Telemetry aggregated at feeder/transformer and canal-gate levels
- Compliant with India's DPDP Act 2023 and NDSAP guidelines

## Security

Six-check security audit (`python agent/security_audit.py .`):
1. Credential scan (API keys, tokens, passwords)
2. `.env` file detection
3. `.gitignore` validation
4. CORS configuration review
5. Debug mode detection
6. Dependency pinning verification

## License

MIT
