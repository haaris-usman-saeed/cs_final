#!/usr/bin/env bash
# Run ONCE per UTC day on your own scheduler (e.g. 00:30 UTC). No job is installed by default.
# Daily INFERENCE with the frozen model only; it never retrains.
# Keyless: grows data/live/nrt_store.parquet from the FIRMS 7-day NRT feed. Predictions are withheld
# until the store covers the model's 30-day fire window (or FIRMS_MAP_KEY is set for backfill).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m fireaq predict-live --model output/model.joblib --out data/live --predictions output/predictions.json
