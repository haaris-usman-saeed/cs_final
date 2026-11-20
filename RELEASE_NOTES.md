# Release notes

See `CHANGELOG.md` (0.2.0) for changes and audit fixes, `release/RESULTS.md` for measured results,
and `docs/MODEL_CARD.md` / `docs/DATA_CARD.md`.

# FireRisk Open 0.1.0 — initial engineering release

**Built:** November 2026. **Status:** Offline software prototype with synthetic integration tests; no actual NASA or Open-Meteo observations have been retrieved inside the current runtime.

## Implemented

- Historical NASA FIRMS VIIRS standard-processing and live NRT fetchers with retries and immutable raw-file manifests (including SHA-256 hashes, key redaction).
- ERA5 and current/past Open-Meteo fetchers with hourly completeness checks.
- Per-grid-cell UTC-day dataset. Strict issue-time cutoff, hypothetical 3-hour NRT latency, future detection labels, prior seven-day eligibility, nearby detection and fire-radiative-power features.
- Matched weather-only vs weather+fire ML ablation: same model family and hyperparameters, temporal validation/test split, validation-only calibration and threshold selection, AP/ROC-AUC/Brier/F1 metrics.
- Regional scope restriction for model scoring; no unseen-geography extrapolation.
- Read-only dashboard plus `/health` and `/api/predictions` endpoints, correctly unavailable without predictions.
- Daily update command ready to schedule **after** a real model is trained.
- 9 passing local automated tests on explicitly synthetic fixtures.

## Unverified and not implemented

- Genuine data retrieval and actual dataset contents.
- Real held-out predictive accuracy, sensitivity, calibration or external validity.
- Satellite observation/clear-sky mask and globally consistent land mask.
- Independent spatial held-out tests, temporal block uncertainty and model improvement over climatology.
- Live-source domain-shift validation, retraining cadence, remote deployment and published downloadable dataset.
- Direct modeling of actual wildfire ignition, spread, smoke or PM2.5; this v0.1 label is **satellite detection** only.

## What to do next

1. Obtain a private NASA FIRMS MAP_KEY and exercise a small real-data download from a location with frequent vegetation fires, checking source schema and measured weather overlap.
2. Confirm that download windows are complete; stop if data source quotas or availability block reconstruction.
3. Scale to multi-year datasets and run train/validation/test baselines. Investigate whether target positives correspond to new fires, ongoing fires, or observation artifact.
4. Add observation opportunity masks and spatial evaluation before calling this a real wildfire-risk product.

The code is MIT-licensed; source datasets have their own upstream terms and attribution requirements.
