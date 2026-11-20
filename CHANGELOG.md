# Changelog

## 0.2.0 — 2026-11-19/20: audit fixes, real-data pilot, model comparison

### Audit of v0.1 (bundle as received, commit `89ae3a5`): 9/9 synthetic tests passed. Confirmed defects

| # | Where | Defect | Fix (test) |
|---|---|---|---|
| A1 | `sources._get` | `raise_for_status()` errors include the full request URL, so a FIRMS 4xx/5xx would print the **MAP_KEY** in tracebacks/logs. Network errors (ConnectionError, Timeout) were not retried. | `SourceError` with redacted URL/body; retries cover network errors, 429 (honours Retry-After and Open-Meteo minute/hour limits) and 5xx (`test_secret_redaction`) |
| A2 | `sources` | No caching or resume: an interrupted multi-year job restarted from zero. Weather was fetched for the whole range in one request per cell. | Raw files + `*.meta.json` sidecars verified by SHA-256 and reused. `.part` files are renamed only after success. Weather fetched in calendar-year chunks |
| A3 | `core.make_dataset` | `drop_duplicates(keep=False)` **silently deleted both copies** of duplicated weather cell-days, so data went missing without notice. | Now raises (`test_duplicate_weather_rows_refused`) |
| A4 | `core.make_dataset` | Neighbour fire features only saw detections inside the bbox, so edge cells were biased low. | FIRMS fetched for a bbox padded by 2 cells, and features computed on the padded grid (`test_neighbour_padding_sees_fires_outside_bbox`) |
| A5 | `core` | A day missing from the FIRMS record became an all-negative label day. | Per-day national acquisition status (`ok`/`suspect`/`missing`). Labels on non-ok target days are withheld, and rows whose eligibility window touches a non-ok day are excluded (`test_missing_firms_day_is_not_a_negative`). The real archive has 6 missing and 4 suspect NOAA-20 days |
| A6 | `model.train_and_evaluate` | The training split was defined by target date but validation/test by issue date, and there was no purge/embargo. Validation and test were one year each. The model family was chosen on the weather-only arm only, and selection and test scoring ran in one call, so nothing was frozen first. Metrics lacked accuracy, balanced accuracy, precision, recall and the confusion matrix, and AP was computed even when a split had no positives. | Replaced by `fireaq/experiment.py`: target-date splits, 32-day embargo, explicit key manifests with a disjointness proof, an 11-family candidate pool × 2 feature sets, validation-only selection/calibration/threshold, `frozen_champion.json`, and a **separate one-time** `evaluate-test` that refuses to run twice (`test_development_then_single_final_test`, `test_splits_embargo_and_disjointness`) |
| A7 | `cli.ingest` | Used local `date.today()` instead of the UTC date. The default bbox (-124,40,-122,42) contains Pacific Ocean cells, and there was no land mask. | UTC dates. Natural Earth land/lake fraction per cell. Default bbox is the pre-registered inland pilot |
| A8 | `cli.predict_live` | The S-NPP/NOAA-20 NRT feed was not checked for emptiness. **The keyless S-NPP NRT feed is currently header-only**, which would have produced "no recent fire" features everywhere. Training used `type==0` filtering, but NRT has no `type`, so live features included industrial/static sources. A 12-day window cannot supply 30-day history features. | Keyless NOAA-20 NRT feed with an empty-feed refusal, a rolling local store, a hard history-coverage check (`BLOCKED_INSUFFICIENT_NRT_HISTORY`), and a static-source mask built from 2019–2022 archive type-2/3 detections (`test_nrt_without_type_uses_static_mask`). The mismatch is quantified in `vintage-check` |
| A9 | `core.make_dataset` | Per-cell Python loops over `panel.xs`, which was slow for multi-year grids. | Vectorized `(day, y, x)` arrays. The full 7-year, 64-cell panel builds in about 2 s |

### Added
* Keyless historical source: FIRMS yearly country archive (`viirs-jpss1`, NOAA-20), checked available for 2018–2025.
* `fireaq/geo.py`: land fraction, exact spherical cell area, and bbox-independent `cell_global_id`.
* `fireaq/ingest.py`: raw → interim layers, ingest report and manifest. `fireaq/dataset.py` builds the released Parquet, CSV preview, data dictionary, coverage report and SHA256SUMS.
* Features: tmax, RH min, wind max, VPD, 30-day rain, dry-day run, ring-2 neighbours, own-cell 8–30-day history, and region-wide 3-day activity. QC columns: target-day overpass cloud proxy, issue-day detection, and explicit timestamps.
* `fireaq/vintage.py`: ERA5 vs archived operational forecast weather, and typed vs type-less fire labels.
* API: `/api/model`, `/api/status`, `?cell_id=` filter, probability validation and a staleness flag.
* `docs/PREREGISTRATION.md`, `docs/MODEL_CARD.md`, `requirements-lock.txt`.

### Not done / blocked
* FIRMS MAP_KEY API path: implemented but **not exercised** (no key in this environment).
* PM2.5 module: **BLOCKED**. OpenAQ v3 returns HTTP 401 without an API key.
* XGBoost/LightGBM: skipped, because both need a system `libomp` (Homebrew) on this Mac.
