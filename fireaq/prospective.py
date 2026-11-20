"""Append-only prospective evaluation log.

``<state>/prospective/log.jsonl``: one JSON record per line, hash-chained:
    record_hash = sha256(prev_hash + canonical_json(record without record_hash))
Record types:
* ``prediction`` - written when a forecast is published (model version + fingerprint, issue time, target window,
  input vintage, calibrated probability, alert flag). Re-logging the identical prediction is a no-op; a different
  value for an existing prediction_id is REFUSED (past predictions are never modified).
* ``outcome`` - appended later, once the target day is fully covered by NOAA-20 NRT data. The label rule is the
  training rule (conf nominal/high; static-source mask in place of the NRT-missing ``type`` field). Days with an
  NRT coverage gap get ``outcome_status = 'unresolvable_gap'`` - never a negative.
* future ``outcome_sp`` records may add standard-processing outcomes; earlier records are never edited.

No accuracy number is reported until MIN_RESOLVED_DAYS issue days and MIN_POSITIVES positive outcomes exist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Grid, clean_firms
from .live_v3 import coverage_by_day
from .regions import REGIONS, STEP

GENESIS = '0' * 64
RESOLVE_LAG_H = 12          # wait after target_end before resolving (NRT latency + next overpass)
MIN_RESOLVED_DAYS = 60
MIN_POSITIVES = 50
OUTCOME_SOURCE = 'NASA FIRMS NOAA-20 NRT (provisional; standard-processing re-check pending)'


def _canon(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(',', ':'), default=str)


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_chain(path: Path) -> dict:
    prev, n = GENESIS, 0
    for i, rec in enumerate(read_log(path)):
        h = rec.get('record_hash')
        body = {k: v for k, v in rec.items() if k != 'record_hash'}
        if body.get('prev_hash') != prev or hashlib.sha256((prev + _canon(body)).encode()).hexdigest() != h:
            return {'ok': False, 'broken_at_line': i + 1}
        prev, n = h, n + 1
    return {'ok': True, 'records': n, 'head': prev}


def _append(path: Path, records: list[dict]) -> int:
    if not records:
        return 0
    chk = verify_chain(path)
    if not chk['ok']:
        raise RuntimeError(f'Log hash chain broken at line {chk["broken_at_line"]}; refusing to append.')
    prev = chk.get('head', GENESIS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as fh:
        for r in records:
            body = {**r, 'prev_hash': prev}
            h = hashlib.sha256((prev + _canon(body)).encode()).hexdigest()
            fh.write(_canon({**body, 'record_hash': h}) + '\n')
            prev = h
    return len(records)


def prediction_id(model_fp: str, cell_key: str, issue: str) -> str:
    return hashlib.sha256(f'{model_fp}|{cell_key}|{issue}'.encode()).hexdigest()[:20]


def log_predictions(predictions_json: Path, log_path: Path, model_fingerprint: str, now=None) -> dict:
    now = now or pd.Timestamp.now(tz='UTC')
    d = json.loads(Path(predictions_json).read_text())
    existing = {r['prediction_id']: r for r in read_log(log_path) if r['record_type'] == 'prediction'}
    fresh, dup = [], 0
    fz = d.get('data_freshness', {})
    for p in d['predictions']:
        pid = prediction_id(model_fingerprint, p['cell_key'], p['issue_time_utc'])
        rec = {'record_type': 'prediction', 'prediction_id': pid, 'logged_at_utc': now.isoformat(),
               'model': d.get('model'), 'model_version': d.get('model_version'), 'model_fingerprint': model_fingerprint,
               'region': p['region'], 'cell_key': p['cell_key'], 'issue_time_utc': p['issue_time_utc'],
               'target_start_utc': p['target_start_utc'], 'target_end_utc': p['target_end_utc'],
               'probability': p['probability_nextday_detection'], 'above_threshold': bool(p['above_validation_threshold']),
               'input_vintage': {'generated_at_utc': d.get('generated_at_utc'),
                                 'nrt_newest_detection_utc': fz.get('nrt_newest_detection_utc'),
                                 'nrt_history_from_utc': fz.get('nrt_history_from_utc'),
                                 'weather_last_day_used': fz.get('weather_last_day_used'),
                                 'weather_sources_lag2': p.get('weather_source_lag2')}}
        if pid in existing:
            if existing[pid]['probability'] != rec['probability']:
                raise RuntimeError(f'Prediction {pid} already logged with probability {existing[pid]["probability"]}; '
                                   f'refusing to log a different value {rec["probability"]} (past predictions are immutable).')
            dup += 1
            continue
        fresh.append(rec)
    return {'appended': _append(log_path, fresh), 'already_logged': dup}


def resolve_outcomes(log_path: Path, nrt_store: pd.DataFrame, static_mask: set | None, now=None) -> dict:
    """Append outcomes for predictions whose target window ended >= RESOLVE_LAG_H ago and that have none yet."""
    now = now or pd.Timestamp.now(tz='UTC')
    recs = read_log(log_path)
    done = {r['prediction_id'] for r in recs if r['record_type'] == 'outcome'}
    todo = [r for r in recs if r['record_type'] == 'prediction' and r['prediction_id'] not in done
            and pd.Timestamp(r['target_end_utc']) + pd.Timedelta(hours=RESOLVE_LAG_H) <= now]
    if not todo:
        return {'resolved': 0, 'pending': 0}
    store = nrt_store.copy()
    if 'acq_utc' not in store:
        t = store['acq_time'].astype(str).str.zfill(4)
        store['acq_utc'] = pd.to_datetime(store['acq_date'].astype(str) + ' ' + t.str[:2] + ':' + t.str[2:], utc=True)
    newest = store['acq_utc'].max()
    days = sorted({r['target_start_utc'][:10] for r in todo})
    cov = coverage_by_day(store, days).set_index('day')
    labels = {}
    for reg in {r['region'] for r in todo}:
        g = Grid(*REGIONS[reg]['bbox'], STEP)
        c = clean_firms(store.drop(columns=[x for x in ['type'] if x in store]), g, static_mask=static_mask)
        c['day'] = c['observed_utc'].dt.strftime('%Y-%m-%d')
        labels[reg] = c.groupby(['cell_id', 'day']).size().to_dict()
    out = []
    for r in todo:
        day = r['target_start_utc'][:10]
        end = pd.Timestamp(r['target_end_utc'])
        if newest < end:
            continue                                    # NRT store has not reached the end of the window yet
        cell_id = r['cell_key'].split(':', 1)[1]
        n = int(labels[r['region']].get((cell_id, day), 0))
        gap = bool(cov.loc[day, 'gap']) if day in cov.index else True
        out.append({'record_type': 'outcome', 'prediction_id': r['prediction_id'], 'resolved_at_utc': now.isoformat(),
                    'outcome_source': OUTCOME_SOURCE, 'target_day_utc': day,
                    'outcome_status': 'unresolvable_gap' if gap else 'resolved',
                    'observed_detection': None if gap else int(n > 0), 'qualifying_detections': None if gap else n,
                    'target_day_nrt_counts': {k: int(cov.loc[day, k]) for k in ['n_all', 'n_day', 'n_night']}
                    if day in cov.index else None})
    return {'resolved': _append(log_path, out), 'pending': len(todo) - len(out)}


def summary(log_path: Path) -> dict:
    recs = read_log(log_path)
    preds = pd.DataFrame([r for r in recs if r['record_type'] == 'prediction'])
    outs = pd.DataFrame([r for r in recs if r['record_type'] == 'outcome'])
    chain = verify_chain(log_path)
    s = {'chain': chain, 'predictions': int(len(preds)), 'outcomes': int(len(outs)),
         'min_requirements': {'resolved_issue_days': MIN_RESOLVED_DAYS, 'positive_outcomes': MIN_POSITIVES},
         'accuracy': None, 'accuracy_status': 'NOT AVAILABLE: insufficient prospective outcomes'}
    if preds.empty:
        return s
    s['issue_days'] = int(preds['issue_time_utc'].nunique())
    s['first_issue'], s['last_issue'] = preds['issue_time_utc'].min(), preds['issue_time_utc'].max()
    if outs.empty:
        return s
    res = outs[outs['outcome_status'] == 'resolved'].merge(preds, on='prediction_id')
    s['resolved'] = int(len(res))
    s['unresolvable_gap'] = int((outs['outcome_status'] == 'unresolvable_gap').sum())
    s['resolved_issue_days'] = int(res['issue_time_utc'].nunique()) if len(res) else 0
    s['positive_outcomes'] = int(res['observed_detection'].sum()) if len(res) else 0
    if s['resolved_issue_days'] >= MIN_RESOLVED_DAYS and s['positive_outcomes'] >= MIN_POSITIVES:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        y, p = res['observed_detection'].astype(int).to_numpy(), res['probability'].astype(float).to_numpy()
        s['accuracy'] = {'average_precision': float(average_precision_score(y, p)),
                         'roc_auc': float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
                         'brier': float(brier_score_loss(y, p)), 'base_rate': float(y.mean()),
                         'mean_predicted': float(p.mean()),
                         'alert_precision': float(y[res['above_threshold'].to_numpy()].mean())
                         if res['above_threshold'].any() else None}
        s['accuracy_status'] = 'PROVISIONAL: NRT outcomes; standard-processing re-check pending'
    return s
