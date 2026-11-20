# FireRisk Open dataset card (draft; no dataset has yet been released)

**Status:** Proposed original dataset. All current integration-test examples are synthetic; no real source records included in the v0.1 package.

## Intended object and population

One row represents the 00:00 UTC issue time for a half-degree regional grid cell, only when the cell has no qualifying known VIIRS detections in the previous seven issue days. Label = whether NASA VIIRS detects a qualifying thermal anomaly in the same cell during UTC day d+1. It is **not** a wildfire ignition label.

## Variables

Identifiers: `cell_id`, `issue_date`, `target_date`, `latitude`, `longitude`.
Features: day-of-year sine/cosine; 2-day lagged weather temperature/humidity/wind; historical precipitation sums at lag 2; 3- and 7-day counts of neighboring satellite detections and radiative power. See `fireaq/core.py` for the authoritative feature code.
Target: `target_nextday_detection` in {0,1}, where zero means **no qualifying detection was in the retrieved FIRMS dataset**. It does not prove clear-sky observing opportunity or absence of fire.

## Data provenance

FIRMS VIIRS NOAA-20 standard processing for historical development, `VIIRS_NOAA20_NRT` for live scoring. Open-Meteo historical endpoint `models=era5` with four hourly variables; live Open-Meteo forecast/past endpoint. Coordinates are degree-binned, weather sampled at cell centres. Immutable raw downloads include hashes and request metadata with NASA access token redacted.

## Known biases

Missing VIIRS detections due to clouds, smoke or overpass gaps; small fires undetected; vegetation burning not necessarily wildfire; island/coast/ocean cells not yet masked; unequal-area latitude/longitude grid; regional fire season/class imbalance; different satellite product vintages; historical ERA5 values not published as issued; 3-hour data latency is an assumption; no land-use or fuel features; no calibrated uncertainty guarantee. Spatially autocorrelated training/test cell histories can overestimate real-world generalization.

## Evaluation protocol

Same model candidate pool and tuning budget for weather-only and weather + prior satellite detections. Train on years before validation, validate on year N, test year N+1. Primary metric average precision and delta against matched weather-only baseline; report prevalence, precision/recall, Brier score and test year. The currently provided tests use only synthetic fixtures and do not report scientific skill.

## License and access

Source code MIT. Planned derived dataset: CC BY 4.0 with NASA product provenance and full Open-Meteo/ECMWF data citation, **subject to verifying precise upstream product terms**. No synthetic observations will be mixed into a release claiming genuine NASA observations.

## Intended users

Researchers, students, and developers studying satellite-detection forecasting. Not for emergency alerts or operational fire danger, spread or ignition forecasts.
