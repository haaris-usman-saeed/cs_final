# Pre-registration — FireRisk Open pilot v0.2

Written **2026-11-19, before any real dataset was built or any model metric was seen.**
Changes after this point must be logged at the bottom with a reason and a note on
whether any test-period result had been seen.

## Source availability checked (2026-11-19)

| Source | Finding |
|---|---|
| FIRMS area API (`/api/area/csv/{MAP_KEY}/...`) | Needs a MAP_KEY. **None available in this environment**, so this path stays implemented but unexercised. |
| FIRMS yearly country archive `data/country/viirs-jpss1/{Y}/viirs-jpss1_{Y}_United_States.csv` | Keyless, HTTP 200 for 2018–2025, 2026 not published. Columns include `type` and `version`. Used for the historical dataset. |
| FIRMS yearly S-NPP archive `viirs-snpp` | Available, but **not used**: the keyless S-NPP NRT feed was empty (header only, last modified 2026-11-12), so a model trained on S-NPP could not be scored live. |
| FIRMS keyless NRT `data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_USA_contiguous_and_Hawaii_7d.csv` | Live, about 1 MB, version `2.0NRT`, **no `type` column**, word-valued `confidence`. Used for live mode. |
| Open-Meteo archive `models=era5` | Live. Hourly temperature, RH, precipitation, wind, cloud cover returned with units. |
| Open-Meteo historical-forecast API | Live. Used only for the operational-vintage check. |
| OpenAQ v3 | HTTP 401 without an API key. The PM2.5 module is **BLOCKED**. |

## Pilot design (frozen)

* **Region:** Northern California / southern Cascades / northern Sierra Nevada,
  bbox west=-123.5, south=38.5, east=-119.5, north=42.5, grid step 0.5° → 8×8 = 64 cells.
  Chosen for frequent vegetation fire (including Camp 2018, August Complex 2020, Dixie 2021,
  Park 2024) while staying small enough for keyless downloads.
* **Land rule:** a cell is kept if its Natural Earth 10 m land fraction (land minus lakes) is ≥ 0.5.
* **Satellite/product:** NOAA-20 (JPSS-1) VIIRS 375 m active fire, FIRMS yearly US archive.
  Detections qualify if `confidence ∈ {n, h}` and `type == 0` (presumed vegetation fire).
* **Issue dates:** 2019-01-01 … 2025-12-30 (00:00 UTC). Target = UTC day d+1.
* **Eligibility (new-activity condition):** no qualifying detection in the cell that would have been
  available at issue time during the previous 7 issue days (3 h assumed latency).
* **Splits by target date:** train targets 2019–2022; validation targets 2023; final test targets
  2024-01-01 … 2025-12-31. A **10-day embargo** at the start of validation and test removes rows whose
  feature window reaches back into the previous split.
* **Geographic hold-out:** 4 quadrant blocks (2°×2°). The frozen champion configuration is retrained
  on training years without block *k* and evaluated on block *k* during the test years.
* **Primary metric:** validation average precision (AP). Checks: balanced accuracy, F1, recall, Brier.
  The champion is the highest validation AP across all candidate models and feature sets.
  Threshold = maximize validation F1 on calibrated probabilities. Calibration = isotonic, fit on validation.
* **Final test:** run once, after champion, calibration and threshold are frozen and written to
  `output/frozen_champion.json`.
* **Label QC:** target days whose national NOAA-20 archive count is 0, or < 10 % of the trailing 31-day
  median, are marked `firms_day_suspect` and **excluded** (not treated as negatives). Target-day
  ERA5 cloud cover at the two NOAA-20 overpass hours ≥ 80 % is flagged `target_cloud_obscured_proxy`
  and used only for a sensitivity analysis, never as a feature or a primary filter.

## Change log after pre-registration

1. **2026-11-19, before the dataset was built and before any metric was seen: embargo changed from 10 to 32 days.**
   The final feature set includes a 30-day rainfall sum and 30-day own-cell fire history, both lagged
   by up to 2 days. The embargo must be at least as long as the longest feature window, so it is now 32 days
   at the start of validation (targets from 2023-02-02) and test (targets from 2024-02-02).
2. **Same time: an eligibility-uncertainty rule was added.** If any day in the 7-day eligibility window, or
   the issue day itself, has FIRMS archive status `missing` or `suspect`, eligibility cannot be established.
   Those rows are excluded and counted in the coverage report.
3. **Same time: the feature list was frozen** as `fireaq/core.py::WEATHER_FEATURES` (15) and `FIRE_FEATURES` (6).
   The candidate models and hyperparameter grids are frozen in `fireaq/experiment.py::candidate_models`.
