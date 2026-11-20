# FireRisk Open dataset card — `firerisk_open_norcal` v0.2.0

**Status:** Built from real, hash-manifested NASA FIRMS and ERA5 downloads. Measured counts are in
`release/coverage_report.json` and `release/RESULTS.md` (both generated from the files themselves).
No synthetic records are included; the synthetic fixtures live only in `tests/`.

## What one row is

One `cell_id × issue_time_utc`: a 0.5° cell of the pilot grid at 00:00 UTC on `issue_date`.

* **Region:** bbox −123.5, 38.5, −119.5, 42.5 (northern California, southern Cascades, northern Sierra Nevada,
  Sacramento Valley, Modoc Plateau, small parts of NW Nevada). 8 × 8 = 64 cells, land fraction 0.88–1.00.
  Cells are **not equal area** (`cell_area_km2` ≈ 2,288–2,411 km²).
* **Issue dates:** 2019-01-01 … 2025-12-30. The file includes *all* cell-days, with flags.
  `in_primary_analysis` selects the modelling population.
* **Primary label `target_nextday_detection`:** 1 if at least one qualifying NOAA-20 VIIRS 375 m detection
  (`confidence` nominal/high, `type` = 0 "presumed vegetation fire") falls in the cell during
  [d+1 00:00, d+2 00:00) UTC, else 0. Labels are **withheld (NaN)**, never set to 0, when the target day's national
  NOAA-20 record is missing or suspect.
* **Population:** cells with no qualifying detection *known at issue time* in the preceding 7 issue days
  ("new activity"), assuming a 3 h FIRMS latency.
* **Secondary continuous outcomes:** `target_nextday_count` and `target_nextday_frp`.

## What the label is NOT

* Not an ignition. Agricultural burning, prescribed fire and continuing fires all count. The share of positives
  that already had a detection on the issue day (after the issue time, so unknown to the model) is reported.
* Not an observed absence. A 0 can mean cloud or smoke obscuration, an off-nadir miss, or a fire below the detection
  limit. `target_cloud_obscured_proxy` (ERA5 cloud ≥ 80 % at both overpass hours) marks the weakest negatives.
  It is a **proxy**: it is not the VIIRS cloud mask, which needs the Level-2 product.

## Sources and vintages

| Layer | Source | Vintage / notes |
|---|---|---|
| Fires | NASA FIRMS yearly country archive, `viirs-jpss1` (NOAA-20 VIIRS 375 m), United States files 2018–2025 | all rows `version = 2` (standard processing). The 2025 file was last modified upstream 2026-10-10 |
| Weather | ECMWF ERA5 via Open-Meteo archive API (`models=era5`), hourly, UTC | reanalysis, published ~5 days late. **Not as-issued** |
| Land mask | Natural Earth 5.1.2 `ne_10m_land` minus `ne_10m_lakes` | public domain |
| Live fires | FIRMS keyless NRT `J1_VIIRS_C2_USA_contiguous_and_Hawaii_7d.csv` | `2.0NRT`, no `type` column |
| Live weather | Open-Meteo forecast API `past_days` | operational models (best match), not ERA5 |

Every raw file is stored byte-for-byte with SHA-256, size, UTC retrieval time and request URL. The FIRMS key, if
any, is redacted. See `release/ingest_manifest.json`.

## Timing (all UTC)

* Fire features use detections with `acq_time + 3 h ≤ issue time`. `fire_obs_cutoff_utc` records this.
* Weather features use complete UTC days up to d−2. `weather_valid_through_utc` records this.
* All windows are continuous across year boundaries (tested). The panel is padded 45 days before the first issue date.

## Quality control

* National per-day NOAA-20 counts flag `missing` days (0 rows) and `suspect` days (no day or no night overpass,
  or < 10 % of the trailing 31-day median). Found: 2023-07-19…22, 2023-09-30 and 2024-02-04 missing;
  2023-10-01, 2024-02-03, 2024-02-05 and 2024-02-20 suspect.
* Weather days with fewer than 24 hourly values are NaN, never zero. Rows with any missing feature are dropped,
  and duplicate cell-days raise an error.
* Detections are de-duplicated on (time, lat, lon).

## Known biases and limitations

Cloud and smoke obscuration; VIIRS overpass timing (~01:30/13:30 local); unequal-area cells; strong
seasonality and class imbalance; agricultural burning in the Sacramento Valley; spatial autocorrelation between
neighbouring cells and days; ERA5 is retrospective. The static/industrial-source handling differs between the
archive (`type`) and NRT (location mask). The pilot covers one region and one satellite.

## Licence and attribution

Dataset: **CC BY 4.0**, inheriting Open-Meteo's CC BY 4.0 requirement.
Cite: NASA FIRMS (LANCE / ESDIS, NOAA-20 VIIRS 375 m Active Fire product, doi:10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002
for NRT), ECMWF ERA5 (Hersbach et al. 2020, Copernicus Climate Change Service), Open-Meteo (Zippenfenig, doi:10.5281/zenodo.7970649),
and Natural Earth. No endorsement by NASA, NOAA, ECMWF or Open-Meteo is implied. Code is MIT.

## Intended use

Benchmarking short-range satellite-detection forecasting, teaching leakage-safe spatio-temporal ML, and
studying label uncertainty. **Not** for fire danger rating, emergency alerts, evacuation, suppression or dispatch.
