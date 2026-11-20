# FireRisk Open v0.1

**Status: working research code, not yet a trained or deployed fire-forecasting system.**

An MIT-licensed Python pipeline for an **original geographic-grid × UTC-day dataset** connecting NASA FIRMS VIIRS active-fire detections and Open-Meteo/ERA5 meteorology. Train a reproducible binary classifier and compare weather-only vs weather+fire features, with a small read-only prediction dashboard.

> **Important:** This predicts whether a *qualifying satellite active-fire detection* appears in a grid cell on the next UTC day, conditional on no fire detection known in that cell during the preceding seven days. A satellite detection is not proof of wildfire ignition, and a non-detection is not proof of no wildfire. **Do not use experimental predictions for emergency management, fire suppression, evacuation, or public safety alerts.**

## Status of the release

| Component | Current status |
|---|---|
| NASA API downloader | Implemented; not exercised against live NASA in this environment |
| Open-Meteo weather downloader | Implemented; not exercised against live API here |
| Grid/time normalization and historical labels | Implemented, unit tested with synthetic cases |
| Baseline vs fire-feature model evaluation | Implemented, validated on synthetic test cases **only** |
| API and read-only dashboard | Implemented; returns 503 until real predictions exist |
| Real-world training dataset | **NOT YET ACQUIRED** |
| Trained real-world model and validated skill | **NOT YET AVAILABLE** |
| Periodic live refresh | CLI command exists; scheduling/deployment remains to be configured |

The folder `legacy/` contains Claude's original PM2.5 feasibility probe for reference. It is **not** the model pipeline for this wildfire-detection target. No test fixture should be presented as a real NASA or weather observation.

## Set up

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS / Linux
pip install -e .
pip install pytest
python -m pytest -q
```

Get your free NASA FIRMS MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/ (NASA requires a key for the API). Do not share or commit the key.

```bash
export FIRMS_MAP_KEY="your_private_key"
```

### Step 1. Download a **small genuine pilot**

The example is a four-cell regional experiment (0.5° cells), not a globally trained model. Choose a date range spanning multiple fire seasons **and a validation and test year** for a real model. The following three-year command is a plausible starting point but may require increased API quotas and longer extraction.

```bash
python -m fireaq ingest \
  --bbox=-123,39,-122,40 --step 0.5 \
  --start 2021-01-01 --end 2024-12-30 \
  --sources VIIRS_NOAA20_SP \
  --out data/norcal
```

The client requests FIRMS in ≤5-day chunks and each grid-cell centre's ERA5 hourly weather once (with feature padding). It stores original downloaded bytes under `data/norcal/raw/`, SHA-256 manifests, `firms.csv`, `weather.csv`, and `ingest_metadata.json`. HTTP errors stop the job rather than being relabelled as no fire.

**Important feasibility caveat:** this v0.1 downloader intentionally does not implement caching/restart/resume for partial multi-year downloads, and the ~1,500-day call can be expensive in API fair-use units. First probe a few months or one fire season; increase coverage only after requests succeed and quotas are checked. The 2019–2024 study design remains a goal, not a completed dataset.

For a smaller first ingestion test:

```bash
python -m fireaq ingest \
  --bbox=-123,39,-122,40 --step 0.5 \
  --start 2022-07-01 --end 2022-09-30 \
  --out data/small_probe
```

That 3-month sample is **too short** for a defensible multi-year model. The training command will reject inadequate train/validation/test splits.

### Step 2. Create a labelled dataset

```bash
python -m fireaq build --input data/norcal --out data/norcal_training.csv
```

Every row corresponds to a grid cell and an issue time of **00:00 UTC on `issue_date`**. The target is at **00:00–24:00 UTC on `target_date = issue_date + 1`**, 24–48 hours after issue. All features precede issue time, with an assumed 3-hour FIRMS NRT latency and 2-day lag on weather history. Standard-processed FIRMS and final ERA5 can be retrospectively revised or published months later, so this is a **hindcast design**, not proof of available-as-issued inputs.

### Step 3. Fit and test the baseline and fire models

```bash
python -m fireaq train \
  --data data/norcal_training.csv \
  --valid-year 2023 --test-year 2024 \
  --model output/model.joblib
```

The pipeline compares matched feature arms with logistic regression and histogram gradient boosting, selects only from validation average precision, and evaluates on a **temporally held-out** test year. It saves an experimental model plus `output/model_metrics.json`. The model refuses to train if any fold lacks enough positive observations. The code does *not* yet implement fully independent spatial block tests, season-balanced bootstraps, or operational weather-vintage comparisons.

**Do not publish model predictions unless held-out skill exceeds relevant baselines and a separate prospective evaluation verifies live-source performance.** Even then, use case-specific review before safety-critical applications.

### Step 4. Generate daily experimental predictions

With a real model trained in the *same geographic region*, run:

```bash
python -m fireaq predict-live \
  --bbox=-123,39,-122,40 --step 0.5 \
  --out data/live \
  --model output/model.joblib \
  --predictions output/predictions.json
```

This obtains NASA **VIIRS NOAA-20 NRT** data for the recent period and Open-Meteo live/previous weather, computes **the fixed 00:00 UTC issue-date snapshot**, and scores tomorrow's UTC day. Running it in the middle of today does **not** update the issue time or use afternoon readings in that morning's features; that's intentional. Historical `ERA5` and live forecast/past weather differ, as do science-quality and NRT FIRMS; this is **unvalidated domain shift**.

The published output is an **experimental satellite-detection probability**. The model refuses regions outside its training coordinate bounds. More extensive geographic transfer validation is required before expansion.

### Step 5. Open the website/API

```bash
uvicorn fireaq.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ and http://127.0.0.1:8000/api/predictions . The API responds with HTTP 503 until `output/predictions.json` exists. It does not improvise a model or display demo values as if real.

To periodically update predictions **on your own infrastructure**, see `scripts/update_daily.sh`. It is *not* a deployed hosted service or a scheduled background job yet.

## Model target, strengths, and limitations

* **Target:** detection of new qualifying VIIRS thermal activity in a grid cell on UTC day d+1 (not a confirmed ignition). Main analysis excludes cells with prior known in-cell detections over seven issue days.
* **Inputs:** fixed grid coordinates, calendar phase, ERA5 mean temperature / humidity / rain / wind from prior complete days, and neighboring VIIRS detection/FRP counts only as available at the issue timestamp.
* **Study unit:** square degree grid cells (0.5° by default), not equal-area pixels; use appropriate geodesic handling and land masking before global extension.
* **Sampling caveat:** no satellite overpass/clear-sky observation mask yet. A day with no detection cannot be distinguished from an unobserved day. This is a crucial bias and must be addressed before claiming calibrated real-world fire ignition risk.
* **Fire types:** includes vegetation thermal anomalies from wildfire, agricultural burning, and prescribed burning. NRT records may not include the `type` field and historical filtering is therefore only applied when available.
* **Holdout:** temporal, not yet spatially independent. Nearby cells and years can share underlying fire events.
* **Data vintages:** standard-processed FIRMS labels and final ERA5 are retrospective. They may not match the corresponding live NRT and forecast feeds.
* **User experience:** dashboard uses a schematic geographic grid, **not a navigable basemap**. No account system or production service is included.

See [docs/DATA_CARD.md](docs/DATA_CARD.md) and [docs/DECISIONS.md](docs/DECISIONS.md) for the open-data and scientific checklist.

## Open-source and data attribution

The **code** in this repository is MIT licensed (see `LICENSE`). NASA FIRMS observations follow NASA/LAADS open-data and citation guidance; use the precise VIIRS product citations and do not imply NASA endorsement. Weather acquired through Open-Meteo is offered under **CC BY 4.0 with attribution** (free hosted API for non-commercial usage). Any data release must include a source/DOI manifest, Open-Meteo and ECMWF ERA5 attribution, retrieval date, feature methods, and appropriate notices. Our software's MIT license does **not** supersede the upstream data conditions. Do not redistribute others' branding or claim official fire-service status.

Official references:
* NASA FIRMS API: https://firms.modaps.eosdis.nasa.gov/api/area/
* NASA FIRMS active-fire attribute guide: https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs
* Open-Meteo historical ERA5: https://open-meteo.com/en/docs/historical-weather-api
* Open-Meteo terms: https://open-meteo.com/en/terms
* ECMWF ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

## Claude Code full-build handoff

To continue from real-data ingestion through dataset construction, chronological/spatial splits, broad model comparisons, validation-only selection, and final held-out testing, use [`docs/CLAUDE_CODE_AUTONOMOUS_BUILD.md`](docs/CLAUDE_CODE_AUTONOMOUS_BUILD.md). It also specifies optional PM2.5 prediction baselines and open-source release checks. **This is a handoff prompt, not evidence that those additions have been implemented.**

## Research collaborator next steps

See [docs/DECISIONS.md](docs/DECISIONS.md) for Phase 0 checks before scaling, and [docs/CLAUDE_HANDOFF.md](docs/CLAUDE_HANDOFF.md) for specific follow-up tasks. **This repository was tested offline only; no genuine observations or validated trained model are shipped.**
