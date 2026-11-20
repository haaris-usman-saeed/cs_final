#!/usr/bin/env bash
# Run ONCE per UTC day on your own scheduler (recommended 01:00-08:00 UTC, before NOAA-20's US day pass).
# Daily INFERENCE with the frozen v0.3 model only; it never retrains. No job is installed by default.
# The job refuses (non-zero exit, reason in data/live/live_status.json) on empty/stale feeds, history gaps or missing weather.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m fireaq predict-live --model output_v3/model_v3.joblib --live-dir data/live --predictions output_v3/predictions.json
