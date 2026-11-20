"""Near-real-time inference for v0.3 (daily INFERENCE with the frozen model; never retrains).

Safety rules (each failure writes data/live/live_status.json and publishes nothing new):
* NOAA-20 NRT feed empty, unreachable, or newest detection older than MAX_FEED_AGE_H  -> REFUSED_FEED
* NRT history does not cover the 7-day eligibility/feature window                      -> REFUSED_HISTORY
* a UTC day inside that window has no / anomalously few NOAA-20 detections (gap)       -> REFUSED_COVERAGE_GAP
* POWER weather for day d-2 missing for > MAX_MISSING_WEATHER of cells                 -> REFUSED_WEATHER
Cells with an individual input problem are listed as unscored, never filled with zeros.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .core import ELIGIBILITY_DAYS, FIRE_LATENCY_H, Grid
from .features_v3 import PAD_DAYS, make_dataset_v3
from .regions import REGIONS, STEP
from .sources import FIRMS_BASE, NRT_FILES, SourceError, _get, fetch_firms, fetch_firms_nrt_keyless, fetch_power_daily, write_manifest

MAX_FEED_AGE_H = 18
MAX_MISSING_WEATHER = 0.10
GAP_FRACTION = 0.10
FEEDS = {'noaa20': NRT_FILES['noaa20'], 'noaa21': 'data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_{region}_{window}.csv',
         'snpp': NRT_FILES['snpp'], 'modis': 'data/active_fire/modis-c6.1/csv/MODIS_C6_1_{region}_{window}.csv'}


def feed_health(now: pd.Timestamp) -> dict:
    """HEAD every keyless US 7-day feed. Only NOAA-20 is used for scoring (training sensor)."""
    out = {}
    for name, rel in FEEDS.items():
        url = f"{FIRMS_BASE}/{rel.format(region='USA_contiguous_and_Hawaii', window='7d')}"
        try:
            import requests
            r = requests.head(url, timeout=30)
            lm = r.headers.get('Last-Modified')
            age = (now - pd.Timestamp(lm).tz_convert('UTC')).total_seconds() / 3600 if lm else None
            size = int(r.headers.get('Content-Length', 0))
            out[name] = {'http': r.status_code, 'bytes': size, 'last_modified': lm,
                         'hours_since_update': None if age is None else round(age, 1),
                         'looks_empty': size < 500, 'used_for_scoring': name == 'noaa20'}
        except Exception as exc:   # report, don't crash: health is informational for non-N20 feeds
            out[name] = {'error': type(exc).__name__, 'used_for_scoring': name == 'noaa20'}
    return out


def update_store(live: Path, manifest: list, key: str, issue, now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    store_p, snaps_p = live / 'nrt_store.parquet', live / 'nrt_snapshots.csv'
    new = fetch_firms_nrt_keyless(live / 'raw', manifest, satellite='noaa20')
    if key:
        parts = []
        for r in REGIONS.values():
            parts.append(fetch_firms(key, r['bbox'], issue - timedelta(days=ELIGIBILITY_DAYS + 3), issue,
                                     'VIIRS_NOAA20_NRT', live / 'raw', manifest))
        new = pd.concat([new] + parts, ignore_index=True)
    for c in ['acq_date', 'acq_time', 'confidence', 'version']:
        new[c] = new[c].astype(str)
    new['acq_time'] = new['acq_time'].str.replace(r'\.0$', '', regex=True).str.zfill(4)
    new['acq_utc'] = pd.to_datetime(new['acq_date'] + ' ' + new['acq_time'].str[:2] + ':' + new['acq_time'].str[2:], utc=True)
    new['first_seen_utc'] = now
    k = ['latitude', 'longitude', 'acq_date', 'acq_time']
    if store_p.exists():
        old = pd.read_parquet(store_p)
        fresh = new.merge(old[k], on=k, how='left', indicator=True)
        fresh = fresh[fresh['_merge'] == 'left_only'].drop(columns='_merge')
        store = pd.concat([old, fresh], ignore_index=True)
    else:
        fresh, store = new, new
    store = store.drop_duplicates(k, keep='first')
    store.to_parquet(store_p, index=False)
    prev_t = None
    if snaps_p.exists():
        prev = pd.read_csv(snaps_p)
        prev_t = pd.Timestamp(prev['fetched_at_utc'].iloc[-1]) if len(prev) else None
    # latency upper bound: detections first seen now that were acquired after the previous snapshot
    lat = fresh if prev_t is None else fresh[fresh['acq_utc'] > prev_t]
    lat_h = ((now - lat['acq_utc']).dt.total_seconds() / 3600) if len(lat) else pd.Series(dtype=float)
    snap = {'fetched_at_utc': now.isoformat(), 'feed_rows': len(new), 'new_rows': len(fresh),
            'feed_oldest_acq_utc': new['acq_utc'].min().isoformat(), 'feed_newest_acq_utc': new['acq_utc'].max().isoformat(),
            'latency_upper_bound_h_median': float(lat_h.median()) if len(lat_h) else None,
            'latency_upper_bound_h_p90': float(lat_h.quantile(0.9)) if len(lat_h) else None,
            'latency_note': 'first_seen - acq for detections acquired after the previous snapshot; '
                            'an upper bound that tightens with more frequent runs' if prev_t is not None else
                            'first snapshot: latency not measurable yet'}
    pd.DataFrame([snap]).to_csv(snaps_p, mode='a', header=not snaps_p.exists(), index=False)
    return store, snap


def coverage_by_day(store: pd.DataFrame, days: list) -> pd.DataFrame:
    d = store.assign(day=store['acq_utc'].dt.strftime('%Y-%m-%d'),
                     dn=store.get('daynight', pd.Series('?', index=store.index)))
    g = d.groupby(['day', 'dn']).size().unstack(fill_value=0)
    out = pd.DataFrame({'day': days})
    out['n_all'] = out['day'].map(g.sum(axis=1)).fillna(0).astype(int)
    out['n_day'] = out['day'].map(g['D'] if 'D' in g else pd.Series(dtype=int)).fillna(0).astype(int)
    out['n_night'] = out['day'].map(g['N'] if 'N' in g else pd.Series(dtype=int)).fillna(0).astype(int)
    med = out['n_all'].median()
    out['gap'] = (out['n_all'] == 0) | (out['n_all'] < GAP_FRACTION * med) | (out['n_day'] == 0) | (out['n_night'] == 0)
    return out


def run_live_v3(model_path: Path = Path('output_v3/model_v3.joblib'), live: Path = Path('data/live'),
                predictions: Path = Path('output_v3/predictions.json'), static_mask_path: Path =
                Path('data/v03/interim/static_source_mask_2019_2022_all_regions.json'),
                test_results: Path = Path('output_v3/test_results_v3.json'), regions: list | None = None,
                key: str = '', now: pd.Timestamp | None = None, log=print) -> dict:
    now = now or pd.Timestamp.now(tz='UTC')
    issue = now.date()
    live.mkdir(parents=True, exist_ok=True)
    status = {'checked_at_utc': now.isoformat(), 'issue_time_utc': f'{issue}T00:00:00Z', 'state': 'STARTED',
              'reasons': []}

    def refuse(state, reason):
        status.update(state=state)
        status['reasons'].append(reason)
        (live / 'live_status.json').write_text(json.dumps(status, indent=2, default=str) + '\n')
        log(f'{state}: {reason}')
        return status

    bundle = joblib.load(model_path)
    meta = bundle['metadata']
    if not meta.get('test_evaluated'):
        return refuse('REFUSED_MODEL', 'model has not completed its one-time untouched test')
    status['feeds'] = feed_health(now)
    manifest = []
    try:
        store, snap = update_store(live, manifest, key, issue, now)
    except SourceError as exc:
        return refuse('REFUSED_FEED', str(exc))
    status['nrt_snapshot'] = snap
    newest_age = (now - pd.Timestamp(snap['feed_newest_acq_utc'])).total_seconds() / 3600
    status['nrt_newest_detection_age_h'] = round(newest_age, 1)
    if newest_age > MAX_FEED_AGE_H:
        return refuse('REFUSED_FEED', f'newest NOAA-20 NRT detection is {newest_age:.1f} h old (> {MAX_FEED_AGE_H} h)')
    need_from = pd.Timestamp(issue, tz='UTC') - pd.Timedelta(days=ELIGIBILITY_DAYS, hours=FIRE_LATENCY_H)
    status['history_required_from_utc'] = need_from.isoformat()
    status['history_available_from_utc'] = store['acq_utc'].min().isoformat()
    if store['acq_utc'].min() > need_from:
        return refuse('REFUSED_HISTORY', f'NRT store starts {store.acq_utc.min()}, need {need_from}')
    win_days = [(pd.Timestamp(issue) - pd.Timedelta(days=k)).strftime('%Y-%m-%d') for k in range(ELIGIBILITY_DAYS + 1, 0, -1)]
    cov = coverage_by_day(store, win_days)
    status['nrt_daily_counts_window'] = cov.to_dict(orient='records')
    # the first window day is only partly needed (from 21:00 UTC); judge completeness on full days
    if cov['gap'].iloc[1:].any():
        return refuse('REFUSED_COVERAGE_GAP', f"NOAA-20 NRT gap on {cov.loc[cov.gap, 'day'].tolist()}: "
                                              'cannot distinguish "no fire" from "no data"')
    mask = set(json.loads(static_mask_path.read_text())) if static_mask_path.exists() else None
    if mask is None:
        status['reasons'].append('static-source mask missing: NRT industrial sources may be counted')
    tests = json.loads(test_results.read_text()) if test_results.exists() else {}
    out_regions, preds, unscored = {}, [], {}
    wstart, wend = issue - timedelta(days=PAD_DAYS + 5), issue - timedelta(days=1)
    for reg in (regions or list(REGIONS)):
        spec = REGIONS[reg]
        cells = pd.read_csv(Path('data/v03/interim') / reg / 'cells.csv')
        land = cells[cells.included_land_cell]
        wx = []
        for c in land.itertuples():
            try:
                wx.append(fetch_power_daily(c.cell_id, c.latitude, c.longitude, wstart, wend, live / 'raw' / reg,
                                            manifest, live=True, pace_seconds=0.2))
            except SourceError as exc:
                unscored.setdefault(reg, []).append({'cell_id': c.cell_id, 'reason': f'weather: {exc}'[:200]})
        if not wx:
            return refuse('REFUSED_WEATHER', f'no POWER weather for {reg}')
        wx = pd.concat(wx, ignore_index=True)
        d2 = (pd.Timestamp(issue) - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        miss = wx[(wx.date == d2) & wx[['t2m', 'rh2m', 'precip_mm', 'gwetroot']].isna().any(axis=1)]['cell_id']
        if len(miss) > MAX_MISSING_WEATHER * len(land):
            return refuse('REFUSED_WEATHER', f'{reg}: POWER d-2 ({d2}) missing for {len(miss)}/{len(land)} cells '
                                             '(near-real-time latency)')
        ds = make_dataset_v3(store, wx, cells, reg, spec['bbox'], STEP, str(issue), str(issue), predict_only=True,
                             static_mask=mask)
        scored = set(ds['cell_id'])
        for cid in set(land.cell_id) - scored:
            unscored.setdefault(reg, []).append({'cell_id': cid, 'reason': 'detection known in the last 7 days '
                                                 '(not eligible by design) or a missing input'})
        if ds.empty:
            continue
        raw = bundle['model'].predict_proba(ds[bundle['features']])[:, 1]
        p = np.clip(bundle['calibrator'].predict(raw), 0, 1)
        o = ds[['region', 'cell_key', 'cell_id', 'cell_global_id', 'latitude', 'longitude', 'issue_time_utc',
                'target_start_utc', 'target_end_utc', 'weather_source_lag2']].copy()
        o['probability_nextday_detection'] = np.round(p, 5)
        o['above_validation_threshold'] = p >= bundle['threshold']
        preds.append(o)
        role = spec['role']
        perf = {}
        for tset in ['test_time', 'test_space_primary', 'test_space_secondary', 'previously_tested_norcal']:
            br = tests.get(tset, {}).get('by_region', {}).get(reg)
            if br:
                perf[tset] = br.get('champion')
        out_regions[reg] = {'name': spec['name'], 'bbox': spec['bbox'], 'role_in_v03': role, 'cells_scored': len(o),
                            'test_performance_champion': perf or 'no test result for this region',
                            'weather_sources_lag2': sorted(o['weather_source_lag2'].dropna().unique().tolist())}
    write_manifest(live / 'manifest.json', manifest)
    if not preds:
        return refuse('REFUSED_NOTHING_TO_SCORE', 'no eligible cells with complete inputs')
    allp = pd.concat(preds, ignore_index=True)
    rel = tests.get('test_time', {}).get('reliability_champion')
    obj = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'issue_time_utc': f'{issue}T00:00:00Z',
        'target_date': str(issue + timedelta(days=1)),
        'target_window_utc': [f'{issue + timedelta(days=1)}T00:00:00Z', f'{issue + timedelta(days=2)}T00:00:00Z'],
        'risk_definition': 'calibrated probability of a NEW qualifying NOAA-20 VIIRS satellite detection in the 0.5° cell '
                           'during the target UTC day. NOT a probability of wildfire ignition, spread or danger.',
        'forecast_status': 'EXPERIMENTAL RESEARCH OUTPUT - NOT FOR EMERGENCY USE - live inputs (NRT, GEOS-IT) differ '
                           'from training inputs (standard processing, MERRA-2)',
        'model': meta['champion']['key'], 'model_version': meta.get('version'),
        'uncertainty': {'test_summary': meta.get('test_summary'), 'reliability_test_time_champion': rel,
                        'note': 'probabilities are calibrated on 2023; reliability shows observed rates per probability '
                                'bin in the untouched 2024-25 test'},
        'data_freshness': {'nrt_newest_detection_utc': snap['feed_newest_acq_utc'],
                           'nrt_newest_detection_age_h': round(newest_age, 1),
                           'nrt_history_from_utc': store['acq_utc'].min().isoformat(),
                           'weather_last_day_used': (pd.Timestamp(issue) - pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                           'latency_upper_bound_h_median': snap['latency_upper_bound_h_median']},
        'regions': out_regions, 'unscored_cells': unscored,
        'geographic_limits': 'Only the 6 listed 4°×4° regions. No other location is supported or extrapolated.',
        'predictions': allp.to_dict(orient='records'),
    }
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    status.update(state='OK', cells_scored=len(allp), predictions_file=str(predictions))
    (live / 'live_status.json').write_text(json.dumps(status, indent=2, default=str) + '\n')
    log(f'OK: {len(allp)} cells scored across {len(out_regions)} regions')
    return status
