#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Onedean/UrbanFM-datasets"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/datasets"

echo "Downloading UrbanFM datasets from ${REPO_ID} ..."
echo "Target directory: ${LOCAL_DIR}"

huggingface-cli download "${REPO_ID}" \
  --repo-type dataset \
  --local-dir "${LOCAL_DIR}"

echo "Done. Datasets saved to ${LOCAL_DIR}"
