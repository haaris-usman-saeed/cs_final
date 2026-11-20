# Claude collaboration task, v0.1 to v0.2

We now have a working Python source package `FireRisk_Open_v0_1`, not a trained real model. Preserve its core schema, immutable-source manifest, leakage-safe temporal handling, no-fabricated-data rule, and read-only web/API architecture. The `legacy/feasibility_probe_pm25.py` file relates to the *prior* PM2.5 question; do not mistakenly train the new fire model on PM2.5 labels.

**First priority: do a real-data Phase 0 integration run, not a global build.**

1. Audit `fireaq/sources.py` against current NASA FIRMS and Open-Meteo API documentation. Confirm that `VIIRS_NOAA20_SP` returns the historical product in the selected pilot period and that the Open-Meteo ERA5 hourly variables and units match schema. Handle firm API errors, pagination/quota, restarts and caching without disguising failures.
2. Use the smallest meaningful high-fire pilot in an on-land North American region over one documented fire season. Save true raw bytes, SHA hashes and manifests; avoid leaking FIRMS MAP_KEY. Run `ingest`, `build` and inspect positives, event coverage and missing-data rates.
3. Separate *satellite non-detection* from *observed non-fire*. Investigate daily clear-sky/overpass opportunities from VIIRS and add an explicit coverage mask; without it, the negative label is weak.
4. Add a land-only mask and equal-area grid for multi-region/global generalization. Build a region-block test with temporal and block-bootstrap confidence intervals.
5. Validate real **as-issued** NRT and weather vintages. ERA5 historical values + standard-processing fires cannot certify operational 24–48 h forecasting. Quantify domain shift or train on an archived operational forecast series.
6. Add unit tests for available-time cutoff at midnight, UTC date boundaries, missing cloud/overpass coverage, schema errors, duplicate data, source swaps and command-line end to end.
7. Report factual download results (rows, dates, headers, SHA hashes, actual API call counts), no synthetic numbers portrayed as source facts. If the source API cannot be accessed, explicitly say so.
8. Do not build or market a global wildfire ignition model from this one regional NOAA-20 baseline. An original detection benchmark is a viable first deliverable.

Deliver the modified source code plus **proof of true observations**; do not claim to have a deployed platform unless public hosting and an automated update job are actually configured.
