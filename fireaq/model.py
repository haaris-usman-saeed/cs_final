"""Scoring with a frozen model bundle produced by ``fireaq.experiment.run_development``.

Training/selection lives in ``fireaq/experiment.py``; this module only applies a frozen bundle.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .core import FEATURES  # noqa: F401  (re-exported for backwards compatibility)


def predict_prob(bundle, frame: pd.DataFrame) -> np.ndarray:
    missing = set(bundle['features']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing prediction features: {missing}')
    if frame[bundle['features']].isna().any().any():
        raise ValueError('Prediction rows contain missing features; refusing to impute at inference time')
    b = bundle['metadata']['training_region_bounds']
    if (frame['latitude'].min() < b['min_latitude']-1e-6 or frame['latitude'].max() > b['max_latitude']+1e-6 or
            frame['longitude'].min() < b['min_longitude']-1e-6 or frame['longitude'].max() > b['max_longitude']+1e-6):
        raise ValueError('Cannot extrapolate beyond training geography without separate spatial validation')
    raw = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    return np.clip(bundle['calibrator'].predict(raw), 0, 1)


def score_predictions(bundle_path: Path, frame: pd.DataFrame):
    obj = joblib.load(bundle_path)  # load ONLY your own trusted model files (joblib = pickle)
    p = predict_prob(obj, frame)
    keep = [c for c in ['cell_id', 'cell_global_id', 'issue_date', 'target_date', 'issue_time_utc',
                        'target_start_utc', 'target_end_utc', 'latitude', 'longitude'] if c in frame]
    out = frame[keep].copy()
    out['probability_nextday_detection'] = np.round(p, 6)
    out['above_validation_threshold'] = p >= obj.get('threshold', 0.5)
    out['model_status'] = 'experimental; not an official fire forecast'
    return out, obj['metadata']
