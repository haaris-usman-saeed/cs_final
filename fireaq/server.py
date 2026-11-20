"""Read-only website and JSON API. It never calls data sources or models on request and never
invents values: with no prediction file it answers HTTP 503."""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

APP_ROOT = Path(__file__).resolve().parent.parent
DISCLAIMER = ('Research prototype. Probability of a NOAA-20 VIIRS satellite fire detection, not of wildfire '
              'ignition, spread or danger. Not for evacuation, suppression, dispatch or any emergency decision.')
app = FastAPI(title='FireRisk Open (Research Prototype)', version='0.2.0', description=DISCLAIMER)


def prediction_file() -> Path:
    return Path(os.environ.get('FIRERISK_PREDICTIONS', str(APP_ROOT / 'output' / 'predictions.json')))


def output_dir() -> Path:
    return Path(os.environ.get('FIRERISK_OUTPUT', str(APP_ROOT / 'output')))


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
            'model_metrics_exist': (output_dir() / 'frozen_champion.json').exists(),
            'research_only': True, 'disclaimer': DISCLAIMER}


@app.get('/api/predictions')
def predictions(cell_id: str | None = None):
    d = _read_json(prediction_file(), 'verified model predictions')
    preds = d.get('predictions')
    if not isinstance(preds, list):
        raise HTTPException(500, detail='prediction file has no predictions list')
    for p in preds:
        v = p.get('probability_nextday_detection')
        if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise HTTPException(500, detail='prediction file contains an invalid probability; refusing to serve it')
    try:
        gen = datetime.fromisoformat(d['generated_at_utc'])
        age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600
    except (KeyError, ValueError):
        age_h = None
    d['age_hours'] = None if age_h is None else round(age_h, 1)
    d['stale'] = age_h is None or age_h > 36
    d['disclaimer'] = DISCLAIMER
    if cell_id is not None:
        d['predictions'] = [p for p in preds if p.get('cell_id') == cell_id or p.get('cell_global_id') == cell_id]
        if not d['predictions']:
            raise HTTPException(404, detail=f'No prediction for cell {cell_id} (outside region or excluded '
                                            'because of a detection in the last 7 days)')
    return d


@app.get('/api/model')
def model_card():
    frozen = _read_json(output_dir() / 'frozen_champion.json', 'frozen model metadata')
    test = output_dir() / 'test_results.json'
    out = {'champion': frozen['champion'], 'selection_rule': frozen['selection_rule'],
           'target': frozen['target'], 'issue_time': frozen['issue_time'],
           'prediction_horizon': frozen['prediction_horizon'], 'split': frozen['split'],
           'training_region_bounds': frozen['training_region_bounds'], 'disclaimer': DISCLAIMER}
    if test.exists():
        t = json.loads(test.read_text())
        out['test'] = {'champion': t['champion_test'], 'comparators': t['calibrated_comparators_test'],
                       'bootstrap_ci95_week_clusters': t['bootstrap_ci95_week_clusters']}
    return out


@app.get('/api/status')
def live_status():
    p = APP_ROOT / 'data' / 'live' / 'live_status.json'
    return _read_json(p, 'live run status')


@app.get('/', response_class=HTMLResponse)
def homepage():
    return (APP_ROOT / 'web' / 'index.html').read_text()
