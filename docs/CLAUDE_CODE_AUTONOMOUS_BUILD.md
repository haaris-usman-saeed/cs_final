# Claude Code: execute FireRisk Open from authentic data to evaluated models

You are the lead engineer and ML researcher for **FireRisk Open**, a Dartmouth COSC 3 project intended to become a reusable open-source dataset, trained model, and public prediction API.

**Work in this existing repository. DO NOT start from scratch.** Inspect `README.md`, `docs/DECISIONS.md`, `docs/DATA_CARD.md`, `docs/CLAUDE_HANDOFF.md`, `fireaq/`, `tests/`, and the original PM2.5 feasibility probe in `legacy/`. Preserve what works; refactor where necessary. The source code is a v0.1 offline prototype: nine synthetic tests have reportedly passed, but it contains **no verified real dataset and no trained/validated real-world model**. Treat the code as unverified until you audit it.

## Overall deliverable

**Main model:** For a land grid cell at 00:00 UTC on issue day d, predict the probability of a **new qualifying satellite active-fire detection during day d+1** (24–48 hours after issue), conditional on the explicitly defined prior-fire exclusion window. This is NOT a validated probability of wildfire ignition, spread, or danger. Do not label every FIRMS non-detection a confirmed fire-free observation.

**Secondary module:** Include optional future support for station-level PM2.5 prediction/episodes using OpenAQ and weather/fire features, drawing on `legacy/feasibility_probe_pm25.py`. Implement only when authentic station coverage, data licenses, correct averaging-hour semantics, and sufficient labels are verified. The secondary task must not delay a genuinely working main fire-detection model. Mean/median/linear regression are essential regression baselines if this module has legitimate real-data labels; also use them for continuous fire-count or FRP targets only if methodologically warranted.

**Public utility:** A person other than us should be able to (a) download a permitted and documented sample dataset, (b) run train/evaluate from a clean checkout, (c) inspect honest metrics and model limitations, and (d) query the current experimental prediction interface. Keep the API/dashboard clearly labelled as research, not an official emergency or fire-service alert.

## Execute in this order (do not just write another feasibility report)

### 0. Audit and testing
- Run the existing tests; inspect the source APIs, UTC issue time, future label logic, data padding, NASA source names, model feature consistency, duplicate records, API error handling, and leakage guards.
- Fix confirmed defects with tests, maintaining a change log. Check current NASA FIRMS API availability for standard-processing (`_SP`) versus near-real-time (`_NRT`) VIIRS sources, and Open-Meteo's ERA5 historical/forecast interfaces. Use only source names and date ranges actually available.
- Never print, transmit in logs, or commit the `FIRMS_MAP_KEY`. Use local environment variables or a git-ignored `.env` loaded securely. If no key or internet access is available, continue implementing/test the pipeline but explicitly mark real-data acquisition as **BLOCKED** and give the shortest exact unblock instruction. Never fabricate observations or results.

### 1. Authentic, manageable data pilot
- Choose a defensible fire-relevant **land** region with adequate historical observations. Start with a small pilot, not a global download. Expand only after the pilot passes. Record the region and dates before examining model performance.
- Fetch authentic NASA FIRMS historical VIIRS standard-product detections and past-only ERA5 meteorology via Open-Meteo (or another legally usable, cited archive). Add caching, restart/resume, rate-limit handling, retries, and bounded chunk sizes. Retrieve raw response bytes and save source URL (secret redacted), UTC retrieval time, product/vintage, SHA-256 hash, schema, units, coverage, and expected date extent.
- Distinguish an API failure, a missing day, a truly empty returned table, and a day with adequate observation opportunity but no qualifying detected fire. Investigate/implement an observable-data/overpass/clear-sky mask or explicit quality limitation; do not quietly turn unavailable data into negatives.
- Verify valid on-land cells and correct geography; for expansion beyond a small regional grid, avoid treating unequal-area latitude/longitude cells as equal-area independent observations.
- Never present synthetic fixtures as genuine records. Report the exact raw-file counts, rows, time ranges, API calls, coverage, missingness and failure cases based on actual outputs.

### 2. Reproducible original dataset
- Build one deterministic, documented row per `cell_id × issue_time_utc`, with `target_start_utc`, `target_end_utc`, latitude/longitude, source-vintage fields, label, QC flags, available-at-time feature timestamps, and provenance identifiers. Prefer Parquet; a small CSV preview is useful. Store separate immutable `raw/`, QC `interim/`, and released `dataset/` layers.
- Features may include region/season/calendar, historical temperature, humidity, wind, precipitation, antecedent dryness where legitimately available, and fire counts/FRP **only from observations known before issue time**. Establish explicit data latency assumptions, respect availability time versus acquisition time, and test all windows at year boundaries.
- Clearly distinguish the **binary new-detection target** from a continuous outcome (e.g., next-day detection count) and from real ignition. Include strong coverage metadata and a data dictionary. If the presence/absence labels are too uncertain, stop and report why rather than publish falsely calibrated fire-risk probabilities.
- Freeze reasonable preprocessing, inclusion/exclusion, missingness and label rules before final test evaluation. Make dataset creation idempotent and verifiable.

### 3. Split before tuning; never contaminate the test set
- No random row-level train/test split as the primary evaluation; adjacent days and cells are autocorrelated. Use chronological splits by **target time** with a purge/embargo around boundaries at least as long as the longest overlapping feature/target window. Suggested if coverage permits: train 2019–2022, validate 2023, untouched final test 2024–2025. Adapt only after checking actual product availability and class counts, and log the reason before tuning.
- Also report a separate geographically held-out block/region evaluation, where feasible. Document counts of rows, positive detections, negative detections, distinct events, geography, years and class ratios in every split; fail if a split lacks positive labels. Save explicit train/validation/test index or file manifests and prove disjointness.
- Fit imputation, scaling, feature selection, resampling, calibration and hyperparameter choices **on training/validation only**, never on test. Do not repeatedly inspect test metrics and tweak the pipeline.

### 4. Comprehensive but sensible model comparison
- Build a common, reproducible experiment runner, identical dataset/split and metric definitions, train-only preprocessing pipelines, a bounded hyperparameter budget, fixed seeds, and model artifacts.
- **For the primary binary classification target**, train at least: DummyClassifier (most frequent, prior probability, stratified if useful), seasonal/geographic climatology, logistic regression (L2), decision tree, random forest, ExtraTrees, histogram gradient boosting, and a small MLP. Try XGBoost/LightGBM only if installable without disrupting the environment. SVM/Naive Bayes are optional if data size and assumptions justify them. Always compare a **weather/history-only** feature set with the same model family plus lagged fire features (matched ablation).
- **For a genuinely continuous target** such as PM2.5 concentration or fire count (separate task), include: global training mean and median constant predictions, linear regression, Ridge regression, random forest regressor, boosted-tree regressor, and an MLP regressor if useful. For every regression model report MAE, MSE, RMSE and R², and compare with the constant baselines. Do not call MSE a model; it is a metric. Do not present linear regression or mean/median regression as naturally calibrated binary probability models.
- For classification, record confusion matrix, test-set prevalence, ordinary accuracy, **balanced accuracy**, precision, recall, F1, ROC-AUC, **average precision (area under PR curve)**, Brier score and calibration/reliability. Also report majority-class and seasonal baseline accuracy to expose class imbalance. Select a decision threshold **using validation only**. If a split has only one class, mark metrics appropriately rather than manufacture a number.
- Select **one champion model by PRE-REGISTERED VALIDATION criterion**, preferably validation average precision (primary) with balanced accuracy, F1, recall and Brier as checks. If the user's intent is highest accuracy, also report the highest *validation balanced accuracy* and accuracy, but do not choose by final test accuracy. Freeze model, preprocessing, calibration and threshold before running final test ONCE. Publish an honest leaderboard including all validation metrics and the final one-time held-out test results. Do not claim statistical significance without uncertainty estimates.
- Save model predictions for the test set, model cards, feature importances where valid, and a brief error analysis for high-risk misses and false positives. Where possible compute temporally/spatially clustered bootstrap confidence intervals.

### 5. Live inference and public usability
- Adapt the existing `/api/predictions` endpoint, web UI, and `predict-live` command so new NASA FIRMS NRT detections + relevant weather inputs yield timestamped experimental predictions for **the same explicitly supported geography, horizon, units, feature schema, and issue time**.
- Investigate and quantify offline-trained standard-product/final-ERA5 versus operational NRT/weather-vintage mismatch. If it cannot be established that the model works under live inputs, leave live mode disabled or clearly **experimental/unvalidated**; never display made-up numbers. Distinguish **daily inference updates** from **retraining**, which requires new verified labels.
- Provide a documented API example, data download/readme, one-command reproducibility instructions, dependency lock/version details, attribution and upstream license checks, data card, model card, and a conspicuous disclaimer: not for evacuation, suppression, dispatch, or emergency decisions.

## Completion criteria and evidence to deliver
1. Updated, working repository preserving existing project structure, with source code, tests, reproducible commands and setup notes.
2. Verifiable **real-data** ingest manifest including SHA-256, source products, dates, rows and QC. If blocked, be precise and do NOT claim the dataset exists.
3. Original labelled dataset and train/validation/test manifest, with positive/negative counts and coverage reports, **only if real extraction worked**.
4. Baselines and all feasible classifiers/regressors trained on the genuine split, per-model validation leaderboard, and a final untouched-test evaluation after a fixed validation-only champion decision; saved model and metrics. If blocked by inadequate positives, say so.
5. Feature ablation, leakage tests, an honest assessment of class imbalance and spatial generalization, and if possible an operational-vintage check.
6. Working API/UI with verified example using real predictions only. Otherwise a clean no-model/no-prediction state.
7. End-of-run summary with exactly **what worked, what failed, actual measured scores (or 'not measured'), remaining blockers, files changed, and precise commands to reproduce**.

Make pragmatic engineering decisions yourself. Avoid rewriting the whole codebase, creating enormous unnecessary downloads, building an elaborate website before having trustworthy data, or optimizing a dataset until the test set leaks into development. Implement and run as much as your actual connected environment allows. Do not invent successes or perform work later in the background.

**Official source references to check against current docs:** NASA FIRMS API `https://firms.modaps.eosdis.nasa.gov/api/area` and data availability; Open-Meteo historical ERA5 `https://open-meteo.com/en/docs/historical-weather-api`; Open-Meteo Historical Forecast documentation `https://open-meteo.com/en/docs/historical-forecast-api`. This project requires source attribution and compliance with each upstream data license.
