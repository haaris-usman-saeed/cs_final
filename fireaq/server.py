"""Read-only website and JSON API. It never calls data sources or models on request and never invents
values: with no prediction file it answers HTTP 503, and stale files are flagged."""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .regions import REGIONS

APP_ROOT = Path(__file__).resolve().parent.parent
DISCLAIMER = ('Research prototype. Calibrated probability of a NEW NOAA-20 VIIRS satellite fire detection in a 0.5° cell '
              'on the next UTC day - not of wildfire ignition, spread or danger. Not for evacuation, suppression, '
              'dispatch, alerts or any emergency decision.')
app = FastAPI(title='FireRisk Open (Research Prototype)', version='0.3.0', description=DISCLAIMER)


def output_dir() -> Path:
    return Path(os.environ.get('FIRERISK_OUTPUT', str(APP_ROOT / 'output_v3')))


def prediction_file() -> Path:
    return Path(os.environ.get('FIRERISK_PREDICTIONS', str(output_dir() / 'predictions.json')))


def status_file() -> Path:
    return Path(os.environ.get('FIRERISK_LIVE_STATUS', str(APP_ROOT / 'data' / 'live' / 'live_status.json')))


def _read_json(path: Path, what: str):
    if not path.exists():
        raise HTTPException(503, detail=f'No {what} available. Nothing is shown until a real model and real inputs exist.')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, detail=f'{what} file is corrupt') from exc


@app.get('/health')
def health():
    return {'ok': True, 'predictions_exist': prediction_file().exists(),
            'model_metrics_exist': (output_dir() / 'frozen_champion_v3.json').exists(),
            'research_only': True, 'disclaimer': DISCLAIMER}


@app.get('/api/regions')
def regions():
    return {'step_degrees': 0.5, 'regions': {k: {'name': v['name'], 'bbox': v['bbox'], 'role_in_v03': v['role']}
                                             for k, v in REGIONS.items()},
            'geographic_limits': 'Predictions exist only inside these boxes; nothing is extrapolated elsewhere.'}


@app.get('/api/predictions')
def predictions(cell_id: str | None = None, region: str | None = None):
    d = _read_json(prediction_file(), 'verified model predictions')
    preds = d.get('predictions')
    if not isinstance(preds, list):
        raise HTTPException(500, detail='prediction file has no predictions list')
    for p in preds:
        v = p.get('probability_nextday_detection')
        if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise HTTPException(500, detail='prediction file contains an invalid probability; refusing to serve it')
    now = datetime.now(timezone.utc)
    try:
        age_h = (now - datetime.fromisoformat(d['generated_at_utc'])).total_seconds() / 3600
    except (KeyError, ValueError):
        age_h = None
    issue = str(d.get('issue_time_utc', ''))[:10]
    d['age_hours'] = None if age_h is None else round(age_h, 1)
    d['stale'] = age_h is None or age_h > 36 or issue != now.strftime('%Y-%m-%d')
    d['stale_reason'] = None if not d['stale'] else ('issue date is not today (UTC); the daily job did not publish'
                                                    if issue != now.strftime('%Y-%m-%d') else 'older than 36 h')
    d['disclaimer'] = DISCLAIMER
    if region is not None:
        if region not in REGIONS:
            raise HTTPException(404, detail=f'Unsupported region {region}. See /api/regions.')
        preds = [p for p in preds if p.get('region') == region]
    if cell_id is not None:
        preds = [p for p in preds if cell_id in (p.get('cell_id'), p.get('cell_key'), p.get('cell_global_id'))]
        if not preds:
            raise HTTPException(404, detail=f'No prediction for {cell_id} (outside the supported regions, or excluded '
                                            'because of a detection in the last 7 days or a missing input)')
    d['predictions'] = preds
    return d


@app.get('/api/model')
def model_card():
    frozen = _read_json(output_dir() / 'frozen_champion_v3.json', 'frozen model metadata')
    out = {k: frozen.get(k) for k in ['version', 'champion', 'selection_rule', 'best_oof_ap', 'one_se',
                                      'ablation_pair', 'validation_matched_ablation_bootstrap', 'frozen_at_utc',
                                      'test_evaluated']}
    out['disclaimer'] = DISCLAIMER
    t = output_dir() / 'test_results_v3.json'
    if t.exists():
        r = json.loads(t.read_text())
        out['test'] = {}
        for k in ['test_time', 'test_space_primary', 'test_space_secondary', 'previously_tested_norcal']:
            if isinstance(r.get(k), dict) and 'metrics' in r[k]:
                out['test'][k] = {'rows': r[k]['rows'], 'positives': r[k]['positives'], 'regions': r[k]['regions'],
                                  'metrics': {m: {x: r[k]['metrics'][m].get(x) for x in
                                                  ['average_precision', 'roc_auc', 'brier', 'balanced_accuracy',
                                                   'precision', 'recall', 'f1']} for m in r[k]['metrics']},
                                  'bootstrap_week': r[k]['bootstrap_week'],
                                  'events_at_f1_threshold': r[k]['events_at_f1_threshold'].get('champion'),
                                  'by_region': r[k]['by_region']}
        out['test_note'] = ('previously_tested_norcal was already evaluated in v0.2 and is NOT an untouched test.')
    return out


@app.get('/api/status')
def live_status():
    return _read_json(status_file(), 'live run status')


@app.get('/', response_class=HTMLResponse)
def homepage():
    return (APP_ROOT / 'web' / 'index.html').read_text()
