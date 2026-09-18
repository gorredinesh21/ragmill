#!/usr/bin/env bash
# Deploy the ragmill query service (Cloud Run, us-central1, scale-to-zero).
# Health endpoint is /api/healthz (the /api prefix matters — bare /healthz
# 404s behind the GFE on run.app; scar tissue from a previous project).
set -euo pipefail

PROJECT="${PROJECT:-personal-project-dg21}"
REGION="${REGION:-us-central1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-441384612427-compute@developer.gserviceaccount.com}"
SERVICE="${SERVICE:-ragmill}"

cd "$(dirname "$0")/.."

ENV_VARS="GCP_PROJECT=${PROJECT},RAGMILL_PROFILE=cloud,RAGMILL_LLM=${RAGMILL_LLM:-vertex}"
# point at a Qdrant Cloud cluster once provisioned:
if [[ -n "${QDRANT_URL:-}" ]]; then
  ENV_VARS+=",QDRANT_URL=${QDRANT_URL},QDRANT_API_KEY=${QDRANT_API_KEY:-}"
fi

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --project "$PROJECT" \
  --allow-unauthenticated \
  --service-account "$SERVICE_ACCOUNT" \
  --cpu 1 --memory 1Gi \
  --set-env-vars "$ENV_VARS"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" \
  --project "$PROJECT" --format 'value(status.url)')
echo "deployed: $URL"
echo "health:   $URL/api/healthz"
