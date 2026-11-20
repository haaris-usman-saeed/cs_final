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
- [ ] Successfully download and inspect genuinely sourced observations from both APIs.
- [ ] Confirm time, source, products and true FIRMS acquisition completeness (including overpass mask).
- [ ] Acquire 3+ fire seasons and enough independent positive events.
- [ ] Spatial leave-region-out test, bootstrapped uncertainty, calibration assessment.
- [ ] Assess standard-processing vs NRT and retrospective ERA5 vs forecast-era data shift.
- [ ] Add an appropriate land mask, grid-area weighting and non-vegetation/industrial-source filtering.
- [ ] Confirm data-release terms and publish verified data card + model card.
- [ ] Schedule a production update job and deploy a production read-only endpoint.

## Minimum requirements before public operational claims

1. The complete source-derived dataset is authentic, with observation-time evidence, API/vintage provenance, and reproducible QC.
2. Source coverage checks distinguish 'no detection' from 'satellite could not observe'.
3. Temporal *and geographic* holdouts outperform a conservative non-fire and climatology baseline with reasonable uncertainty.
4. Training features are measurable using the actual operational data vintage and update timing.
5. Limit the public claim to what the label measures and clearly document drift/failures.

**Never confuse a research probability for an official safety alert.**
