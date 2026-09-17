# ragmill — one image, two entrypoints.
#   service (default):  uvicorn app.main:app — deployed via
#     gcloud run deploy ragmill --source .
#   ingestor (Cloud Run Job): managed jobs override the command:
#     --command python3 --args jobs/ingest_job.py
#     (CLOUD_RUN_TASK_INDEX / CLOUD_RUN_TASK_COUNT feed --shard/--num-shards)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY jobs ./jobs
COPY scripts ./scripts
COPY eval ./eval
COPY static ./static
# local-mode corpus is small (2MB) — bundled so the service demo works
# out of the box; cloud profiles point RAGMILL_SOURCE/QDRANT_URL elsewhere.
COPY data ./data

# warm the embedding model into the image so cold starts don't download
# ~80MB at request time (best effort — falls back at runtime if offline)
RUN python3 -c "from app.embeddings import get_embedder; \
get_embedder()" || echo "model warm-up skipped (ok)"

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
