#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${HF_REPO_ID:-onedean/UrbanFM-datasets}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS_DIR="${PROJECT_ROOT}/datasets"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face."
  echo "Run: hf auth login"
  echo "Or:  HF_TOKEN=hf_xxx bash scripts/upload_datasets_to_hf.sh"
  exit 1
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "Uploading ${DATASETS_DIR} to hf://${REPO_ID} (repo-type: dataset) ..."
echo "This may take several hours for ~10 GB of data."

hf upload "${REPO_ID}" "${DATASETS_DIR}" . \
  --repo-type dataset \
  --exclude ".git/*" \
  --commit-message "Upload UrbanFM datasets"

# Set dataset card README
hf upload "${REPO_ID}" "${DATASETS_DIR}/DATASET_CARD.md" "README.md" \
  --repo-type dataset \
  --commit-message "Add dataset card"

echo "Done: https://huggingface.co/datasets/${REPO_ID}"
