# Decision log / release gates — November 2026

## Chosen MVP

Build a publicly reproducible **satellite-detection forecasting dataset**, rather than claim genuine wildfire ignition probability. NASA FIRMS is a detection source, not a complete ignition inventory. Start regionally, then consider a global expansion after coverage and geographic transfer validation.

## Milestones

- [x] Reuse Claude's PM2.5 feasibility probe in `legacy/` as source context.
- [x] Implement NASA FIRMS standard/NRT data retrieval and hashed raw manifests.
- [x] Implement ERA5 and near-real-time Open-Meteo hourly weather retrieval.
- [x] Construct timestamp-safe grid-cell, next-day detection labels and neighbor features.
- [x] Implement same-architecture weather-only ablation and temporally held-out test.
- [x] Implement small read-only web dashboard and API with no-prediction state.
- [x] Pass local synthetic unit/integration tests.
- [x] Download and inspect real observations: FIRMS NOAA-20 yearly archive (keyless) + Open-Meteo ERA5, SHA-256 manifested (2026-11-20). MAP_KEY API path still unexercised.
- [~] Acquisition completeness: national per-day status (6 missing, 4 suspect days) + ERA5 overpass-cloud proxy. A true VIIRS L2 cloud/observation mask is still missing.
- [x] 7 fire seasons (2019–2025), 4,042 positives; 982/195/427 distinct events in train/validation/test.
- [x] Leave-one-block-out (4 blocks), week- and cell-clustered bootstrap CIs, reliability tables (release/RESULTS.md).
- [~] ERA5 vs archived operational forecasts and typed vs type-less fire labels quantified (release/vintage_check.json). SP vs NRT detections not measured.
- [~] Natural Earth land/lake fraction, exact cell areas recorded, type==0 filter + static-source mask for NRT. Area weighting not applied (cell areas vary ±3% in the pilot).
- [x] Data card, model card, CC BY 4.0 dataset with upstream attribution (docs/).
- [ ] Schedule a production update job and deploy a production read-only endpoint.

## Minimum requirements before public operational claims

1. The complete source-derived dataset is authentic, with observation-time evidence, API/vintage provenance, and reproducible QC.
2. Source coverage checks distinguish 'no detection' from 'satellite could not observe'.
3. Temporal *and geographic* holdouts outperform a conservative non-fire and climatology baseline with reasonable uncertainty.
4. Training features are measurable using the actual operational data vintage and update timing.
5. Limit the public claim to what the label measures and clearly document drift/failures.

**Never confuse a research probability for an official safety alert.**

## Outcome of the v0.2 pilot (2026-11-20)

The detection benchmark is real and reproducible. The forecast skill is weak. On the one-time test the frozen
champion has AP 0.085, climatology 0.076 and the base rate is 0.036. The gain over climatology is not robust to
time-clustered uncertainty, and fire-history features add no measurable skill over weather-only. **Gate 3
("outperform climatology with reasonable uncertainty") is NOT met.** No operational claims. Live publishing stays
experimental, and it is blocked until 30 days of NRT history exist.
