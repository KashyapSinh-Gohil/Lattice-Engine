# Lattice decision API + single-page dashboard — Cloud Run image.
# High-performance CPU image.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ pipeline/
COPY agro/ agro/
COPY agent/ agent/
COPY api/ api/
COPY web/ web/
COPY data_gen/ data_gen/
COPY bench/ bench/
# demo packs: pre-computed artifacts so the public URL works with zero warm-up
COPY api/data/ api/data/
COPY api/data_agro/ api/data_agro/
# small datasets so the LIVE RE-RUN button works on Cloud Run too (both domains)
COPY data/city_1m/ data/city_1m/
COPY data/agro_1m/ data/agro_1m/

ENV PORT=8080 GRID_OUT=/app/api/data AGRO_OUT=/app/api/data_agro
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
