"""Operational-vintage check, run only AFTER the one-time test with the frozen model (no tuning).

1. Weather vintage: ERA5 reanalysis (training) vs Open-Meteo *historical forecast* archive
   (archived operational model output, closest available proxy for what live mode receives),
   on a 16-cell subset during the test years: per-variable bias/MAE/correlation, then the frozen
   champion scored on identical rows with each weather source.
2. Fire vintage proxy: live NRT files have no ``type`` field. Re-derive test-period labels and
   features from the archive with ``type`` removed and the 2019-2022 static-source mask applied,
   and compare with the type-filtered training definition.
Not measurable keylessly: standard-processing vs NRT geolocation/detection differences.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .core import FEATURES, Grid, make_dataset
from .experiment import TARGET, assign_splits
from .ingest import NEIGHBOR_PAD_CELLS
from .sources import fetch_weather_historical, write_manifest

COMPARE = {'temp_c': 'temperature', 'rh_pct': 'humidity', 'wind_ms': 'wind', 'precip_mm': 'precipitation',
           'tmax_c': 'tmax', 'rhmin_pct': 'rhmin', 'vpd_kpa': 'vpd'}


def _score(bundle, frame):
    raw = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    return bundle['calibrator'].predict(raw)


def _m(y, p):
    y = np.asarray(y, int)
    return {'n': int(len(y)), 'positives': int(y.sum()),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'roc_auc': float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            'brier': float(brier_score_loss(y, p)), 'mean_probability': float(np.mean(p))}


def run_vintage_check(ingest_dir: Path, outdir: Path, primary: pd.DataFrame, log=print) -> dict:
    ingest_dir, outdir = Path(ingest_dir), Path(outdir)
    if not (outdir / 'test_results.json').exists():
        raise RuntimeError('Run evaluate-test first; the vintage check must not influence development.')
    bundle = joblib.load(outdir / 'model.joblib')
    meta = json.loads((ingest_dir / 'ingest_metadata.json').read_text())
    grid = Grid(*meta['bbox'], meta['step'])
    interim = ingest_dir / 'interim'
    cells = pd.read_csv(interim / 'cells.csv')
    firms = pd.read_parquet(interim / 'firms_regional.parquet')
    status = pd.read_csv(interim / 'firms_day_status.csv')
    era5 = pd.read_parquet(interim / 'weather_daily.parquet')

    prim = primary[primary[TARGET].notna()].copy()
    prim['split'] = assign_splits(prim)
    test = prim[prim.split == 'test']
    issue_start, issue_end = test['issue_date'].min(), test['issue_date'].max()

    # ---- 1. weather vintage on a fixed 16-cell checkerboard subset
    subset = cells[(cells.iy % 2 == 0) & (cells.ix % 2 == 0)]
    manifest, frames = [], []
    for r in subset.itertuples():
        frames.append(fetch_weather_historical(r.cell_id, r.latitude, r.longitude, date(2023, 12, 1),
                                               date.fromisoformat(test['target_date'].max()),
                                               ingest_dir / 'raw', manifest, api='historical_forecast'))
    write_manifest(outdir / 'vintage_manifest.json', manifest)
    hf = pd.concat(frames, ignore_index=True)
    both = era5.merge(hf, on=['cell_id', 'date'], suffixes=('_era5', '_hf'))
    both = both[(both['date'] >= issue_start) & (both['n_hours_era5'] == 24) & (both['n_hours_hf'] == 24)]
    weather_diff = {}
    for col, name in COMPARE.items():
        a, b = both[f'{col}_era5'], both[f'{col}_hf']
        ok = a.notna() & b.notna()
        weather_diff[name] = {'n_cell_days': int(ok.sum()), 'bias_hf_minus_era5': float((b[ok]-a[ok]).mean()),
                              'mae': float((b[ok]-a[ok]).abs().mean()), 'corr': float(np.corrcoef(a[ok], b[ok])[0, 1])}
    hf_ds = make_dataset(firms, hf, grid, issue_start, issue_end, pad_cells=NEIGHBOR_PAD_CELLS,
                         day_status=status)
    key = ['cell_id', 'issue_date']
    paired = test.merge(hf_ds[key + [c for c in FEATURES if c not in ('latitude', 'longitude')]],
                        on=key, suffixes=('', '_hf'))
    hf_frame = paired.copy()
    for c in FEATURES:
        if f'{c}_hf' in hf_frame:
            hf_frame[c] = hf_frame[f'{c}_hf']
    p_era5, p_hf = _score(bundle, paired), _score(bundle, hf_frame)
    weather_eval = {'rows_paired': int(len(paired)),
                    'era5_inputs': _m(paired[TARGET], p_era5), 'historical_forecast_inputs': _m(paired[TARGET], p_hf),
                    'mean_abs_probability_change': float(np.mean(np.abs(p_hf - p_era5))),
                    'threshold_decision_agreement': float(np.mean((p_hf >= bundle['threshold']) ==
                                                                  (p_era5 >= bundle['threshold'])))}

    # ---- 2. type-less fire definition (what NRT supplies) vs type-filtered (training)
    mask = set(json.loads((interim / 'static_source_mask_2019_2022.json').read_text()))
    typeless = firms.drop(columns=['type'])
    nt_ds = make_dataset(typeless, era5, grid, issue_start, issue_end, pad_cells=NEIGHBOR_PAD_CELLS,
                         day_status=status, static_mask=mask, keep_ineligible=True)
    sp_ds = make_dataset(firms, era5, grid, issue_start, issue_end, pad_cells=NEIGHBOR_PAD_CELLS,
                         day_status=status, keep_ineligible=True)
    lab = sp_ds[key + [TARGET, 'eligible_new_activity']].merge(
        nt_ds[key + [TARGET, 'eligible_new_activity']], on=key, suffixes=('_typed', '_typeless')).dropna()
    a, b = lab[f'{TARGET}_typed'].astype(int), lab[f'{TARGET}_typeless'].astype(int)
    label_agreement = {'cell_days': int(len(lab)), 'both_positive': int(((a == 1) & (b == 1)).sum()),
                       'typed_only_positive': int(((a == 1) & (b == 0)).sum()),
                       'typeless_only_positive': int(((a == 0) & (b == 1)).sum()),
                       'eligibility_disagreement': int((lab['eligible_new_activity_typed'] !=
                                                        lab['eligible_new_activity_typeless']).sum())}
    nt_test = nt_ds[nt_ds['eligible_new_activity'] & ~nt_ds['eligibility_uncertain']].merge(test[key], on=key)
    nt_test = nt_test[nt_test[TARGET].notna()]
    typeless_eval = {'note': 'features AND labels re-derived without the type field (static mask applied); '
                             'rows = test rows eligible under both definitions',
                     'metrics': _m(nt_test[TARGET], _score(bundle, nt_test)),
                     'typed_reference_same_keys': _m(test.merge(nt_test[key], on=key)[TARGET],
                                                     _score(bundle, test.merge(nt_test[key], on=key)))}
    res = {'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
           'weather_subset_cells': subset['cell_id'].tolist(), 'weather_variable_differences': weather_diff,
           'weather_vintage_model_eval': weather_eval, 'fire_label_agreement_typed_vs_typeless': label_agreement,
           'typeless_fire_model_eval': typeless_eval,
           'not_measured': ['FIRMS standard-processing vs NRT detection/geolocation differences (needs MAP_KEY '
                            'NRT archive or prospective collection)', 'NRT latency distribution (3 h assumed)']}
    res['summary'] = {
        'weather_AP_era5_vs_hf': [weather_eval['era5_inputs']['average_precision'],
                                  weather_eval['historical_forecast_inputs']['average_precision']],
        'weather_decision_agreement': weather_eval['threshold_decision_agreement'],
        'label_agreement': label_agreement,
        'typeless_AP_vs_typed_AP': [typeless_eval['metrics']['average_precision'],
                                    typeless_eval['typed_reference_same_keys']['average_precision']]}
    (outdir / 'vintage_check.json').write_text(json.dumps(res, indent=2, default=str) + '\n')
    return res
