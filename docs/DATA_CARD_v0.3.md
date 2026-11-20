# Dataset card — `firerisk_open_multiregion` v0.3.0

`release_v3/firerisk_open_multiregion_v0.3.0.parquet` (zstd Parquet, ~31.5 MB). Exact counts, the SHA-256 and the
per-region/role statistics are in `release_v3/coverage_report_v3.json`, generated from the file. Rebuilding from
the raw layer is byte-identical (verified).

## Rows
One row per `cell_key × issue_time_utc` (00:00 UTC). It covers 6 regions × 64 cells of 0.5°, keeping 381 cells with Natural Earth
land fraction ≥ 0.5 (3 coastal Southeast cells are dropped). Issue dates run 2019-01-01 … 2025-12-30. There are **973,836 cell-days**, 65 columns,
and **632,239** in the primary analysis (`in_primary_analysis`).

| Region | Role | bbox |
|---|---|---|
| norcal | development (2024–25 already evaluated in v0.2) | −123.5, 38.5, −119.5, 42.5 |
| nrockies | development; 2024–25 = untouched test | −117, 44, −113, 48 |
| southwest | development; 2024–25 = untouched test | −112, 32, −108, 36 |
| southeast | development; 2024–25 = untouched test | −85, 30, −81, 34 |
| pnw | untouched test region (all years) | −123, 43, −119, 47 |
| ozarks | untouched test region (all years) | −96, 34, −92, 38 |

The `role` column gives each row's role in the v0.3 evaluation.

## Label (identical rule to v0.2)
`target_nextday_detection` is 1 if at least one NOAA-20 VIIRS 375 m detection with confidence nominal/high and `type` 0 falls in the
cell during UTC day d+1. It is computed only for cells with no detection known at issue time (3 h latency) in the prior 7 issue days.
Labels are **withheld (NaN), never set to 0,** on target days when the national NOAA-20 record is missing or suspect. There
are 6 missing and 4 suspect days in 2018–2025. `target_nextday_count` and `target_nextday_frp` are secondary continuous outcomes.
**A satellite detection is not an ignition, and a non-detection is not proof of no fire.**

## Features (all known at issue time)
The full column list is in `release_v3/data_dictionary_v3.csv`.
* **Weather and soil:** NASA POWER daily UTC values on day d−2 for T2M, T2M_MAX, RH2M, VPD (computed), WS10M, WS10M_MAX, surface and
  root-zone soil wetness, plus the 30-day soil-wetness change. Precipitation sums over 3, 7, 30 and 90 days, and the dry-day run.
* **Static:** ESA WorldCover 2020 class fractions, sampled through COG overviews at about 90 m effective resolution. Copernicus DEM GLO-90
  elevation mean and standard deviation, computed by average-resampling the 90 m pixels to about 180 m.
* **Fire history:** neighbour, ring-2 and region-mean NOAA-20 detections known in the last 3 or 7 issue days. S-NPP 2012–2017 fire-day
  climatology per cell, annual and for the issue month. This predates the study period, so it cannot leak labels.
* **QC only, never features:** target-day POWER/CERES cloud amount, a cloud-obscured flag at ≥ 80%, and `detection_on_issue_day`.

## Sources, versions, licences

| Layer | Source | Licence / terms |
|---|---|---|
| Fire labels and features | NASA FIRMS yearly US archive `viirs-jpss1`, 2018–2025, version 2 | NASA open data; cite LANCE FIRMS / ESDIS; no endorsement |
| Fire climatology | NASA FIRMS yearly US archive `viirs-snpp`, 2012–2017 | as above |
| Weather and soil | NASA POWER daily API v2 (MERRA-2; CERES SYN1DEG clouds; GEOS-IT for near-real-time) | NASA open data; cite POWER Project (NASA LaRC) |
| Land cover | ESA WorldCover 10 m 2020 v100 | **CC BY 4.0**, © ESA WorldCover project / Contains modified Copernicus Sentinel data (2020) |
| Terrain | Copernicus DEM GLO-90 (AWS Open Data) | Copernicus DEM licence: free use with attribution "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA" |
| Land mask | Natural Earth 10 m land and lakes v5.1.2 | public domain |

The derived dataset is released under **CC BY 4.0**, and the attributions above must be kept. Every raw response is stored with
its SHA-256, byte count, UTC retrieval time, URL and product (`release_v3/ingest_manifest_v3.json`). No API key is used,
so none is stored.

## Known limitations
* WorldCover 2020 is a single snapshot. For 2019–2020 rows it partly reflects post-fire land cover, which is a mild leakage
  risk in the development years only. It predates the 2024–25 test.
* POWER switches from MERRA-2 to GEOS-IT for recent days, so live weather differs from training weather.
* VPD is approximated from daily maximum temperature and daily-mean RH.
* Cells are 0.5° and not equal-area (`cell_area_km2`). The data covers one satellite (NOAA-20) and 6 US regions.
* Cloud and smoke obscuration is only a reanalysis proxy. VIIRS Level-2 cloud masks need an Earthdata login.
* Agricultural and prescribed fires count as detections.
