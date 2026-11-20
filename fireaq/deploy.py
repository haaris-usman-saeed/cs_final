"""Deterministic rebuild of the frozen v0.3 champion from the committed dataset, verified by a prediction fingerprint.

The frozen model file (~107 MB) is not committed. Deployments rebuild it from
``release_v3/firerisk_open_multiregion_v0.3.0.parquet`` with the frozen configuration (same rows, seed, calibration
and threshold rule), then compare its calibrated probabilities on a fixed 2,000-row reference sample with the
fingerprint taken from the original frozen model. Any mismatch beyond 1e-9 refuses deployment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

from .experiment_v3 import EMBARGO_DAYS, assign_roles, build
from .features_v3 import TARGET

TOL = 1e-9
N_REF = 2000


def _reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed, label-free sample: the first N_REF untouched-test rows by (cell_key, issue_date)."""
    t = df[assign_roles(df).isin(['test_time', 'test_space_primary'])]
    return t.sort_values(['cell_key', 'issue_date']).head(N_REF).reset_index(drop=True)


def _probs(bundle, X):
    return np.clip(bundle['calibrator'].predict(bundle['model'].predict_proba(X[bundle['features']])[:, 1]), 0, 1)


def fingerprint(bundle, df: pd.DataFrame) -> dict:
    ref = _reference_rows(df)
    p = _probs(bundle, ref)
    keys = (ref['cell_key'] + '|' + ref['issue_date']).str.cat(sep='\n').encode()
    return {'n': int(len(ref)), 'reference_keys_sha256': hashlib.sha256(keys).hexdigest(),
            'probabilities_rounded_sha256': hashlib.sha256(np.round(p, 9).tobytes()).hexdigest(),
            'probabilities': [float(x) for x in p], 'threshold': float(bundle['threshold']),
            'model': bundle['metadata']['champion']['key']}


def rebuild_champion(dataset: Path, frozen_json: Path, out: Path, log=print) -> Path:
    meta = json.loads(Path(frozen_json).read_text())
    df = pd.read_parquet(dataset)
    df = df[df['in_primary_analysis']]
    roles = assign_roles(df)
    dev = df[(roles == 'dev') & df[TARGET].notna()].reset_index(drop=True)
    from .experiment_v3 import _hash
    if _hash(dev) != meta['dev_dataset_sha256']:
        raise RuntimeError('Development rows differ from the frozen record; refusing to rebuild.')
    t = pd.to_datetime(dev['target_date'])
    tr = (t <= pd.Timestamp('2022-12-31')).to_numpy()
    va = ((t >= pd.Timestamp('2023-01-01') + pd.Timedelta(days=EMBARGO_DAYS)) & (t <= pd.Timestamp('2023-12-31'))).to_numpy()
    y = dev[TARGET].astype(int).to_numpy()
    c = meta['champion']
    pipe, cols = build(c['model'], c['feature_set'], c['params'])
    pipe.fit(dev.loc[tr, cols], y[tr])
    p23 = pipe.predict_proba(dev.loc[va, cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p23, y[va])
    c23 = iso.predict(p23)
    qs = np.unique(np.quantile(c23, np.linspace(0.5, 0.999, 300)))
    thr = float(max(qs, key=lambda q: f1_score(y[va], c23 >= q, zero_division=0)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {'model': pipe, 'calibrator': iso, 'features': cols, 'threshold': thr,
              'metadata': {**meta, 'rebuilt_from': str(dataset), 'rebuilt_at_utc': pd.Timestamp.now(tz='UTC').isoformat()}}
    path = out / 'model_v3.joblib'
    joblib.dump(bundle, path)
    log(f'rebuilt {c["key"]} -> {path} (threshold {thr:.6f})')
    return path


def verify(model_path: Path, dataset: Path, reference: Path, log=print) -> dict:
    bundle = joblib.load(model_path)
    df = pd.read_parquet(dataset)
    df = df[df['in_primary_analysis']]
    ref = json.loads(Path(reference).read_text())
    now = fingerprint(bundle, df)
    diff = float(np.max(np.abs(np.array(now['probabilities']) - np.array(ref['probabilities']))))
    res = {'reference_keys_match': now['reference_keys_sha256'] == ref['reference_keys_sha256'],
           'max_abs_probability_diff': diff, 'threshold_diff': abs(now['threshold'] - ref['threshold']),
           'exact_hash_match': now['probabilities_rounded_sha256'] == ref['probabilities_rounded_sha256']}
    res['ok'] = res['reference_keys_match'] and diff <= TOL and res['threshold_diff'] <= TOL
    log(f'fingerprint check: {res}')
    if not res['ok']:
        raise RuntimeError(f'Rebuilt model does not reproduce the frozen model: {res}. Refusing deployment.')
    return res
