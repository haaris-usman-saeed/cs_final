# Pre-registration — FireRisk Open v0.3 (multi-region)

Written **2026-11-20 ~05:00 UTC, before any data from the new regions was downloaded or inspected and before any v0.3
model was trained.** The v0.2 NorCal 2024–25 test results are known; see "Contamination" below. Changes after this
point go in the change log at the bottom, with the reason and what had been seen.

## Target (unchanged from v0.2)
For a 0.5° land cell at 00:00 UTC on issue day d, among cells with no qualifying NOAA-20 VIIRS detection *known at
issue time* during the previous 7 issue days: does at least one qualifying detection (confidence nominal/high,
`type` = 0) occur in the cell during UTC day d+1? This is a satellite-detection target, **not ignition**.

## Regions (4°×4°, 0.5° grid, cells kept if Natural Earth land fraction ≥ 0.5)

| Role | Region | bbox (W,S,E,N) | Regime |
|---|---|---|---|
| development | `norcal` Northern California | −123.5, 38.5, −119.5, 42.5 | Mediterranean forest/shrub + Sacramento Valley agriculture |
| development | `nrockies` central Idaho / W Montana | −117, 44, −113, 48 | Interior conifer forest, summer lightning |
| development | `southwest` SE Arizona / SW New Mexico | −112, 32, −108, 36 | Pine/juniper/grassland, spring–early-summer fire |
| development | `southeast` S Georgia / N Florida | −85, 30, −81, 34 | Pine flatwoods, year-round prescribed fire |
| **untouched test** | `pnw` Oregon/Washington Cascades | −123, 43, −119, 47 | Conifer forest + Columbia Basin agriculture (near transfer) |
| **untouched test** | `ozarks` E Oklahoma / W Arkansas / SW Missouri | −96, 34, −92, 38 | Humid oak–hickory, pasture & prescribed burning (far transfer) |

## Data sources (all keyless)
* Fires: FIRMS yearly US archive `viirs-jpss1` (NOAA-20, 2018–2025, version 2). Fire climatology: FIRMS yearly US
  archive `viirs-snpp` 2012–2017, **before the study period**.
* Weather / soil: NASA POWER daily point API, UTC (MERRA-2; GEOS-IT for near-real-time; clouds from CERES SYN1DEG).
* Land cover: ESA WorldCover 2020 v100 (10 m, CC BY 4.0), class fractions per cell.
* Terrain: Copernicus DEM GLO-90 through the Open-Meteo elevation API (7×7 samples per cell).
* Land/lake mask: Natural Earth 10 m.

## Features (frozen)
* **Set A, "no fire information" (no coordinates):** `doy_sin`, `doy_cos`; POWER on day d−2: `t2m`, `t2m_max`,
  `rh2m`, `vpd` (from T2M_MAX/RH2M), `ws10m`, `ws10m_max`; precipitation sums over 3/7/30/90 days ending d−2; dry-day
  run (≤ 30); soil wetness `gwettop`, `gwetroot` on d−2; `gwetroot` change over 30 days; static: WorldCover tree, shrub,
  grass, crop, built, bare, wetland fractions; elevation mean and standard deviation.
* **Set B, "+ fire history" = A plus live-computable fire information only:** neighbour (8-cell) detections known in
  the last 3 and 7 issue days, neighbour FRP in the last 3 days, ring-2 detections in the last 7 days, regional per-cell
  mean detections in the last 3 days, and **static S-NPP 2012–2017 climatology**: the cell's mean annual fire-days and
  its fire-days in the issue month.
* No feature may use data after the issue time. Weather uses d−2. Fire uses `acq_time + 3 h ≤ issue`.

## Splits
* Development rows: development regions, target dates 2019-01-02 … 2023-12-31.
* **Selection CV (rolling origin):** three folds validating target years 2021, 2022, 2023 (Feb 2 – Dec 31 of that year,
  after a 32-day embargo). Each trains on targets from 2019-01-02 to Dec 31 of the previous year, pooled over the 4
  development regions.
* **Final model:** trained on development-region targets ≤ 2022-12-31. Isotonic calibration and the threshold
  (max F1) are fit on the 2023 fold only.
* **Untouched test, evaluated once after freezing:**
  * **T-time:** `nrockies`, `southwest`, `southeast`, targets 2024-02-02 … 2025-12-31.
  * **T-space:** `pnw`, `ozarks`, targets 2024-02-02 … 2025-12-31 (space + time; primary) and 2019-01-02 … 2023-12-31
    (space only; secondary).
  * `norcal` 2024–25 is **not** part of the fresh test. Because it was already evaluated in v0.2, it is reported in a
    separate, clearly labelled table and never used for tuning.
* During development no label from T-time or T-space rows is read. The experiment code excludes them, and the coverage
  report shows only label-free counts for them until the test is run.

## Candidates (bounded grids, seed 17)
Dummy (prior, most-frequent, stratified); historical S-NPP climatology (cell×month); training-label climatology
(cell×month, month-rate fallback); logistic regression L2 (C ∈ {0.01, 0.1, 1}); Gaussian NB; decision tree
(depth ∈ {4, 8}, leaf 100); random forest (200 trees, depth ∈ {12, None}, leaf 20); Extra Trees (300 trees, leaf ∈
{20, 50}); HistGradientBoosting (lr 0.05, leaves ∈ {15, 31}, 300 iterations); XGBoost (400 trees, lr 0.05,
depth ∈ {4, 6}, subsample 0.8, colsample 0.8, min_child_weight 5); MLP ((64, 32), α ∈ {1e-3, 1e-2}). Every learned
model is fit on both A and B.
Continuous secondary target `log1p(next-day detection count)`: mean, median, linear regression, ridge, random forest,
HistGB, XGBoost and MLP regressors, trained ≤ 2022 and validated on 2023 (MAE, MSE, RMSE, R²).

## Champion rule
1. For each (model, config, feature set), compute the **pooled out-of-fold AP** over the three CV folds.
2. Keep the best config per (model, feature set).
3. Compute a block-bootstrap SE of pooled AP (ISO-week clusters, 500 replicates) for the best model.
4. Among candidates whose pooled AP is within 1 SE of the best, choose the one with the highest
   **leave-one-region-out AP**: train on 3 development regions (targets ≤ 2022) and validate on the 4th in 2023,
   pooled over the 4 held-out regions. Ties are broken by lower cross-fitted calibrated Brier.
5. Balanced accuracy, F1 and recall at the cross-fitted threshold are reported as checks, not used to choose.

## Fire-history question (matched ablation)
For the champion family, compare A vs B with identical rows, folds, grid and selection. Report out-of-fold and test
Δ AP with week- and cell-cluster bootstrap 95% CIs, and event-level onset recall at matched alert rates.

## Event-level metrics
An event is a set of positive cell-days linked by 8-neighbour adjacency within ≤ 2 days. Reported: onset recall
(the event's first positive cell-day scored above threshold), event recall (any cell-day of the event above threshold),
and alerts per 1,000 cell-days.

## Contamination statement
v0.2 inspected NorCal 2019–2025 including test labels. NorCal therefore stays a development region only, and its
2024–25 numbers are never called "untouched". The new-region choice was made from fire-regime knowledge, not from
inspecting their FIRMS data.

## Change log after pre-registration
(none yet)
