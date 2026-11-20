#!/usr/bin/env bash
# Example to run ONCE every UTC day using your own scheduler (e.g. 01:15 UTC).
# No job is installed by default. Requires a trained *real* local model and FIRMS_MAP_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FIRMS_MAP_KEY:?Set FIRMS_MAP_KEY outside the repository}"
python -m fireaq predict-live --bbox=-123,39,-122,40 --step 0.5 \
  --model output/model.joblib --out data/live --predictions output/predictions.json
