#!/usr/bin/env bash
# ragmill — launch the 1M-doc sharded ingest as a Cloud Run Job.
#
# NOT run tonight (no Qdrant Cloud cluster / corpus in GCS yet — by design;
# the local 5K path is what shipped). This script is the runnable scale path:
# fill in the two env values below and execute. It is idempotent: every point
# ID is uuid5(arxiv_id#chunk), so re-executing never duplicates.
#
# Prereqs (one-time, ~15 min of setup, documented in README "The 1M plan"):
#   1. Qdrant Cloud free cluster  -> export QDRANT_URL / QDRANT_API_KEY
#      (free tier: 1 cluster, 0.5-1GB RAM — serves ~500K docs @ 384d int8)
#   2. Corpus in GCS as sharded jsonl:
#        gs://ragmill-corpus/arxiv-1m/shard-00000.jsonl ... shard-00099.jsonl
#      Produce it with scripts/fetch_arxiv.py + split, or the Kaggle dump:
#        kagglehub.dataset_download("Cornell-University/arxiv")
#        # stream the json -> head -1000000 | split -l 10000 -> gcloud storage cp
#      (The "prepare" pass never touches the laptop: run it on Cloud Shell.)
#   3. gcloud auth + application-default login for your project.
set -euo pipefail

PROJECT="${PROJECT:-personal-project-dg21}"
REGION="${REGION:-us-central1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-441384612427-compute@developer.gserviceaccount.com}"
TASKS="${TASKS:-10}"                # 10 parallel shards
MEMORY="${MEMORY:-512Mi}"           # task-specified size; see README note:
                                    # 512Mi works for bge-small + batch 100,
                                    # but 1Gi is the safer setting for the
                                    # full 1M run (ONNX + buffers).
CORPUS_GS="${CORPUS_GS:?set CORPUS_GS=gs://<bucket>/arxiv-1m}"
QDRANT_URL="${QDRANT_URL:?set QDRANT_URL=https://<cluster>.qdrant.io:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:?set QDRANT_API_KEY}"
STATUS_GS="${STATUS_GS:-gs://ragmill-status/ingest-1m}"

cd "$(dirname "$0")/.."

echo "== building + deploying job ragmill-ingest ($TASKS tasks, $MEMORY) =="
gcloud run jobs deploy ragmill-ingest \
  --source . \
  --region "$REGION" \
  --project "$PROJECT" \
  --service-account "$SERVICE_ACCOUNT" \
  --command python3 --args jobs/ingest_job.py,--shard-by,file \
  --set-env-vars "QDRANT_URL=$QDRANT_URL,QDRANT_API_KEY=$QDRANT_API_KEY,RAGMILL_SOURCE=$CORPUS_GS,RAGMILL_STATUS_DIR=$STATUS_GS,RAGMILL_UPSERT_BATCH=100,RAGMILL_EMBED_BATCH=64" \
  --tasks "$TASKS" \
  --parallelism "$TASKS" \
  --max-retries 1 \
  --task-timeout 60m \
  --cpu 1 \
  --memory "$MEMORY"

echo "== executing =="
gcloud run jobs execute ragmill-ingest \
  --region "$REGION" --project "$PROJECT" \
  --tasks "$TASKS" --parallelism "$TASKS" --wait

echo "== per-task throughput (from status objects) =="
for i in $(seq 0 $((TASKS - 1))); do
  gcloud storage cat "$STATUS_GS/task-$i.json" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"task {d[\"shard\"]}: {d[\"done\"]} docs, {d.get(\"docs_per_s\")} docs/s, errors={d[\"errors\"]}")' \
    || echo "task $i: no status object"
done
echo "done. point the service at the same QDRANT_URL and search."
