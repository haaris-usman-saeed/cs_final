"""v0.3 tests. ALL data here are synthetic; they test code correctness, not forecast skill."""
import json

import numpy as np
import pandas as pd
import pytest

import fireaq.experiment_v3 as ex
from fireaq.features_v3 import SET_A, SET_B, TARGET, make_dataset_v3
from fireaq.sources import SourceError, power_json_to_daily, worldcover_tile

BBOX = (-124.0, 40.0, -123.0, 40.5)   # 2 x 1 cells of 0.5 deg


def cells_frame(land=(True, True)):
    c = pd.DataFrame({'cell_id': ['r0_c0', 'r0_c1'], 'ix': [0, 1], 'iy': [0, 0], 'latitude': [40.25, 40.25],
                      'longitude': [-123.75, -123.25], 'land_fraction': [1.0, 1.0], 'cell_area_km2': [2300.0, 2300.0],
                      'cell_global_id': ['g_a', 'g_b'], 'region': 'testreg', 'included_land_cell': list(land),
                      'elev_mean_m': [500.0, 900.0], 'elev_std_m': [50.0, 120.0], 'clim_fire_days_year': [3.0, 0.5]})
    c['cell_key'] = 'testreg:' + c['cell_id']
    for k in ['lc_tree', 'lc_shrub', 'lc_grass', 'lc_crop', 'lc_built', 'lc_bare', 'lc_wetland']:
        c[k] = 0.1
    for m in range(1, 13):
        c[f'clim_fire_days_m{m:02d}'] = [float(m), 0.0]
    return c


def power_weather(start, end, cells=('r0_c0', 'r0_c1'), precip=0.0):
    rows = []
    for c in cells:
        for d in pd.date_range(start, end):
            rows.append({'cell_id': c, 'date': str(d.date()), 't2m': 20.0, 't2m_max': 28.0, 't2m_min': 12.0,
                         'rh2m': 30.0, 'precip_mm': precip, 'ws10m': 3.0, 'ws10m_max': 6.0, 'gwettop': 0.2,
                         'gwetroot': 0.4 - 0.001 * (d.dayofyear % 50), 'cloud_pct': 10.0, 'vpd_kpa': 2.6,
                         'power_sources': 'MERRA2'})
    return pd.DataFrame(rows)


def fire(date_, lat=40.25, lon=-123.75, hour=12):
    return {'latitude': lat, 'longitude': lon, 'acq_date': date_, 'acq_time': f'{hour:02d}00',
            'confidence': 'n', 'frp': 5.0, 'type': 0}


def test_power_fill_values_become_nan_and_utc_enforced():
    payload = {'header': {'time_standard': 'UTC', 'fill_value': -999.0, 'sources': ['MERRA2']},
               'properties': {'parameter': {k: {'20240101': 1.0, '20240102': -999.0} for k in
                                            ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'QV2M', 'PRECTOTCORR', 'WS10M',
                                             'WS10M_MAX', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT']}}}
    out = power_json_to_daily(payload, 'r0_c0')
    assert out.loc[1, ['t2m', 'precip_mm', 'gwetroot']].isna().all()
    payload['header']['time_standard'] = 'LST'
    with pytest.raises(SourceError):
        power_json_to_daily(payload, 'r0_c0')


def test_worldcover_tile_names():
    assert worldcover_tile(40.25, -121.25) == 'N39W123'
    assert worldcover_tile(30.25, -84.75) == 'N30W087'
    assert worldcover_tile(33.9, -81.1) == 'N33W084'


def test_v3_labels_timing_statics_and_no_leakage():
    fires = pd.DataFrame([fire('2022-03-01'), fire('2022-07-12'), fire('2022-07-20', lon=-123.25)])
    out = make_dataset_v3(fires, power_weather('2022-03-01', '2022-07-31'), cells_frame(), 'testreg', BBOX, 0.5,
                          '2022-07-05', '2022-07-25')
    r = out[(out.issue_date == '2022-07-11') & (out.cell_id == 'r0_c0')].iloc[0]
    assert r[TARGET] == 1 and r['neighbor_fire_3d'] == 0
    assert r['clim_fire_days_month'] == 7.0 and r['elev_mean_m'] == 500.0 and r['cell_key'] == 'testreg:r0_c0'
    assert not ((out.cell_id == 'r0_c0') & out.issue_date.between('2022-07-13', '2022-07-19')).any()
    nb = out[out.cell_id == 'r0_c1'].set_index('issue_date')
    assert nb.loc['2022-07-12', 'neighbor_fire_3d'] == 0 and nb.loc['2022-07-13', 'neighbor_fire_3d'] == 1
    assert nb.loc['2022-07-13', 'region_mean_fire_3d'] == pytest.approx(0.5)     # 1 detection / 2 land cells
    assert nb.loc['2022-07-19', TARGET] == 1
    assert set(SET_B) <= set(out.columns) and not {'latitude', 'longitude'} & set(SET_A)


def test_v3_dry_days_and_rain_windows_and_missing_weather():
    w = power_weather('2022-03-01', '2022-07-31')
    w.loc[w.date == '2022-07-01', 'precip_mm'] = 5.0
    out = make_dataset_v3(pd.DataFrame([fire('2022-03-01')]), w, cells_frame(), 'testreg', BBOX, 0.5,
                          '2022-07-05', '2022-07-10').set_index(['cell_id', 'issue_date'])
    assert out.loc[('r0_c1', '2022-07-05'), 'dry_days_lag2'] == 2          # rain on Jul 1, lag-2 day is Jul 3
    assert out.loc[('r0_c1', '2022-07-05'), 'rain_7d_lag2'] == 5.0
    w.loc[(w.date == '2022-07-06') & (w.cell_id == 'r0_c1'), 't2m'] = np.nan
    out2 = make_dataset_v3(pd.DataFrame([fire('2022-03-01')]), w, cells_frame(), 'testreg', BBOX, 0.5,
                           '2022-07-05', '2022-07-10')
    assert not ((out2.cell_id == 'r0_c1') & (out2.issue_date == '2022-07-08')).any()


def test_non_land_cells_are_not_rows():
    out = make_dataset_v3(pd.DataFrame([fire('2022-03-01')]), power_weather('2022-03-01', '2022-07-31'),
                          cells_frame(land=(True, False)), 'testreg', BBOX, 0.5, '2022-07-05', '2022-07-06')
    assert set(out.cell_id) == {'r0_c0'}


def test_roles():
    df = pd.DataFrame({'region': ['norcal', 'norcal', 'southeast', 'southeast', 'pnw', 'pnw', 'ozarks'],
                       'target_date': ['2023-06-01', '2024-06-01', '2024-01-15', '2025-03-01', '2020-05-01',
                                       '2024-08-01', '2019-01-01']})
    assert list(ex.assign_roles(df)) == ['dev', 'previously_tested_norcal', 'unused', 'test_time',
                                         'test_space_secondary', 'test_space_primary', 'unused']


def test_event_metrics():
    df = pd.DataFrame({'region': 'r', 'cell_id': ['r0_c0', 'r0_c1', 'r5_c5', 'r9_c9'],
                       'target_date': ['2022-07-01', '2022-07-02', '2022-07-01', '2022-07-01'],
                       TARGET: [1, 1, 1, 0]})
    m = ex.event_metrics(df, np.array([0.9, 0.1, 0.1, 0.8]), 0.5)
    assert m['events'] == 2 and m['onset_recall'] == 0.5 and m['alerts_per_1000_cell_days'] == 500.0


def synthetic_multiregion():
    rng = np.random.default_rng(3)
    rows = []
    for reg in ['norcal', 'nrockies', 'southwest', 'southeast', 'pnw', 'ozarks']:
        for d in pd.date_range('2019-01-01', '2025-12-30', freq='2D'):
            for cid in ['r0_c0', 'r1_c1', 'r2_c2']:
                z = rng.normal()
                row = {f: float(rng.normal()) for f in SET_B}
                row.update({'region': reg, 'cell_id': cid, 'cell_key': f'{reg}:{cid}', 'issue_date': str(d.date()),
                            'target_date': str((d + pd.Timedelta(days=1)).date()),
                            'doy_sin': np.sin(2*np.pi*d.dayofyear/365.25), 'doy_cos': np.cos(2*np.pi*d.dayofyear/365.25),
                            'clim_fire_days_month': max(0.0, z + 1), 'neighbor_fire_3d': max(0.0, z),
                            TARGET: float(rng.random() < (0.3 if z > 0.5 else 0.03)),
                            'detection_on_issue_day': False, 'target_cloud_obscured_proxy': False})
                for f in ['neighbor_fire_7d', 'neighbor_frp_3d', 'ring2_fire_7d', 'region_mean_fire_3d',
                          'clim_fire_days_year', 'rain_3d_lag2', 'rain_7d_lag2', 'rain_30d_lag2', 'rain_90d_lag2',
                          'elev_std_m']:
                    row[f] = abs(row[f])
                row['target_nextday_count'] = row[TARGET]
                rows.append(row)
    return pd.DataFrame(rows)


def test_v3_development_never_sees_test_rows_and_test_runs_once(tmp_path, monkeypatch):
    full = ex.candidates()
    small = {k: full[k] for k in ['dummy_prior', 'dummy_most_frequent', 'dummy_stratified', 'climatology_hist_snpp',
                                  'climatology_train_cell_month', 'logistic_l2']}
    monkeypatch.setattr(ex, 'candidates', lambda: small)
    df = synthetic_multiregion()
    poisoned = df.copy()
    roles = ex.assign_roles(poisoned)
    poisoned.loc[roles.isin(['test_time', 'test_space_primary', 'test_space_secondary']), SET_B] = np.nan
    meta = ex.run_development_v3(poisoned, tmp_path, log=lambda *_: None)   # NaN in test rows must not matter
    meta_clean = ex.run_development_v3(df, tmp_path / 'b', log=lambda *_: None)
    assert meta['dev_dataset_sha256'] == meta_clean['dev_dataset_sha256']
    board = pd.read_csv(tmp_path / 'validation_leaderboard_v3.csv')
    assert board.iloc[0]['oof_ap_pooled'] >= board['oof_ap_pooled'].max() - 1e-12
    assert meta['champion']['feature_set'] in ('A_no_fire_info', 'B_plus_fire_history', 'n/a')
    assert set(meta['ablation_pair']) == {'A_no_fire_info', 'B_plus_fire_history'}
    ex.regression_v3(df, tmp_path, log=lambda *_: None)
    res = ex.run_final_test_v3(df, tmp_path, log=lambda *_: None)
    assert {'test_time', 'test_space_primary', 'test_space_secondary', 'previously_tested_norcal'} <= set(res)
    assert 'events_at_f1_threshold' in res['test_time']
    with pytest.raises(RuntimeError, match='already been run'):
        ex.run_final_test_v3(df, tmp_path, log=lambda *_: None)
    assert json.loads((tmp_path / 'frozen_champion_v3.json').read_text())['test_evaluated'] is True


def test_live_gap_detection_flags_missing_and_one_sided_days():
    from fireaq.live_v3 import coverage_by_day
    rows = []
    for day, nd, nn in [('2026-11-10', 500, 400), ('2026-11-11', 0, 0), ('2026-11-12', 480, 0), ('2026-11-13', 510, 390)]:
        rows += [{'acq_utc': pd.Timestamp(f'{day} 10:00', tz='UTC'), 'daynight': 'D'}] * nd
        rows += [{'acq_utc': pd.Timestamp(f'{day} 21:00', tz='UTC'), 'daynight': 'N'}] * nn
    cov = coverage_by_day(pd.DataFrame(rows), ['2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13'])
    assert cov['gap'].tolist() == [False, True, True, False]


def test_api_marks_yesterdays_issue_as_stale(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from fastapi.testclient import TestClient
    from fireaq import server
    now = datetime.now(timezone.utc)
    p = tmp_path / 'p.json'
    p.write_text(json.dumps({'generated_at_utc': now.isoformat(),
                             'issue_time_utc': (now - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z'),
                             'predictions': [{'region': 'pnw', 'cell_id': 'r0_c0', 'probability_nextday_detection': 0.1}]}))
    monkeypatch.setenv('FIRERISK_PREDICTIONS', str(p))
    body = TestClient(server.app).get('/api/predictions').json()
    assert body['stale'] is True and 'not today' in body['stale_reason']


def test_power_vintage_double_difference_recovers_planted_shift():
    from fireaq.vintage_v3 import WINDOWS, compute_shifts
    rng = np.random.default_rng(0)
    rows = []
    for c in range(12):
        for w, (a, b) in WINDOWS.items():
            for d in pd.date_range(a, b):
                e = 10 + 5 * rng.normal()
                sm = 0.2 + 0.05 * rng.random()
                bias = 1.0 if w == 'geosit_2026' else 0.0          # planted +1.0 C GEOS-IT shift in t2m only
                rows.append({'cell_key': f'r:{c}', 'region': 'norcal', 'window': w, 'date': str(d.date()),
                             'power_sources': 'X', 't2m': e + 0.3 + bias + 0.1 * rng.normal(), 'era5_t2m': e,
                             't2m_max': e + 5, 'era5_t2m_max': e + 5, 'rh2m': 50.0, 'era5_rh': 50.0, 'ws10m': 2.0,
                             'era5_ws': 2.0, 'ws10m_max': 4.0, 'era5_ws_max': 4.0, 'precip_mm': 1.0, 'era5_precip': 1.0,
                             'gwettop': 0.1 + 2 * sm, 'era5_sm_top': sm, 'gwetroot': 0.1 + 2 * sm, 'era5_sm_root': sm})
    summary, ratio, pc = compute_shifts(pd.DataFrame(rows), log=lambda *_: None)
    assert summary['t2m']['shift_mean'] == pytest.approx(1.0, abs=0.05)
    assert abs(summary['t2m']['null_mean_2025_minus_2024']) < 0.05
    assert abs(summary['gwetroot']['shift_mean']) < 0.01 and abs(summary['rh2m']['shift_mean']) < 1e-9


def test_nrt_vs_sp_matching_recovers_planted_differences():
    from fireaq.nrt_vs_sp import compare
    rng = np.random.default_rng(1)
    n = 400
    sp = pd.DataFrame({'latitude': 40.0 + rng.random(n) * 2, 'longitude': -122.5 + rng.random(n) * 2,
                       'acq_date': '2025-08-01', 'acq_time': rng.choice(['0930', '2112'], n),
                       'confidence': rng.choice(['n', 'h', 'l'], n, p=[.6, .3, .1]), 'frp': 5 + rng.random(n) * 20,
                       'daynight': 'D', 'type': 0})
    keep = rng.random(n) > 0.1                                   # NRT misses 10 %
    nrt = sp[keep].copy()
    nrt['latitude'] += 200 / 6_371_008.8 * 180 / np.pi            # 200 m north shift
    plus1 = nrt['acq_time'].map({'0930': '0931', '2112': '2113'})                 # +1 minute for half of them
    nrt['acq_time'] = np.where(rng.random(len(nrt)) < 0.5, nrt['acq_time'], plus1)
    nrt['confidence'] = nrt['confidence'].map({'n': 'nominal', 'h': 'high', 'l': 'low'})
    nrt = nrt.drop(columns='type')
    extra = sp.sample(20, random_state=2).assign(latitude=lambda d: d.latitude + 0.2, confidence='nominal').drop(columns='type')
    nrt = pd.concat([nrt, extra], ignore_index=True)
    r = compare(sp, nrt, '2025-08-01', '2025-08-01', regions=['norcal'])['regions']['norcal']
    ns = int((sp.latitude.between(38.5, 42.5) & sp.longitude.between(-123.5, -119.5)).sum())
    assert r['sp_detections'] == ns
    assert r['sp_matched_share'] == pytest.approx(0.9, abs=0.05)
    assert r['geolocation_m']['mean_north'] == pytest.approx(200, abs=5)
    assert r['confidence_matrix_sp_rows_nrt_cols']['nominal'].get('high', 0) == 0
    assert r['unmatched_nrt_by_conf'].get('nominal', 0) >= 1


def _pred_file(tmp_path, issue='2026-11-20', p=0.2):
    d = {'model': 'extra_trees|B', 'model_version': '0.3.0', 'generated_at_utc': f'{issue}T03:00:00+00:00',
         'data_freshness': {'nrt_newest_detection_utc': f'{issue}T00:00:00Z'},
         'predictions': [{'region': 'norcal', 'cell_key': 'norcal:r0_c0', 'issue_time_utc': f'{issue}T00:00:00Z',
                          'target_start_utc': '2026-11-21T00:00:00Z', 'target_end_utc': '2026-11-22T00:00:00Z',
                          'probability_nextday_detection': p, 'above_validation_threshold': True}]}
    f = tmp_path / f'pred_{p}.json'
    f.write_text(json.dumps(d))
    return f


def test_prospective_log_is_append_only_and_tamper_evident(tmp_path):
    from fireaq import prospective as pr
    log = tmp_path / 'log.jsonl'
    assert pr.log_predictions(_pred_file(tmp_path), log, 'fp1')['appended'] == 1
    assert pr.log_predictions(_pred_file(tmp_path), log, 'fp1') == {'appended': 0, 'already_logged': 1}
    with pytest.raises(RuntimeError, match='immutable'):
        pr.log_predictions(_pred_file(tmp_path, p=0.9), log, 'fp1')
    assert pr.verify_chain(log)['ok']
    lines = log.read_text().splitlines()
    log.write_text(lines[0].replace('0.2', '0.3') + '\n')
    assert not pr.verify_chain(log)['ok']
    with pytest.raises(RuntimeError, match='chain broken'):
        pr.log_predictions(_pred_file(tmp_path, issue='2026-11-21'), log, 'fp1')


def test_prospective_outcomes_resolve_only_with_coverage_and_no_accuracy_claim(tmp_path):
    from fireaq import prospective as pr
    log = tmp_path / 'log.jsonl'
    pr.log_predictions(_pred_file(tmp_path), log, 'fp1')
    def store(days, fire_day=None):
        rows = []
        for d in days:
            rows += [{'latitude': 30.1, 'longitude': -90.1, 'acq_date': d, 'acq_time': '0900', 'confidence': 'nominal',
                      'frp': 1.0, 'daynight': 'N'}] * 300
            rows += [{'latitude': 30.1, 'longitude': -90.1, 'acq_date': d, 'acq_time': '2100', 'confidence': 'nominal',
                      'frp': 1.0, 'daynight': 'D'}] * 300
        if fire_day:
            rows.append({'latitude': 38.75, 'longitude': -123.25, 'acq_date': fire_day, 'acq_time': '2100',
                         'confidence': 'high', 'frp': 9.0, 'daynight': 'D'})
        rows.append({'latitude': 30.1, 'longitude': -90.1, 'acq_date': '2026-11-22', 'acq_time': '0900',
                     'confidence': 'nominal', 'frp': 1.0, 'daynight': 'N'})
        return pd.DataFrame(rows)
    early = pd.Timestamp('2026-11-22T06:00', tz='UTC')
    assert pr.resolve_outcomes(log, store(['2026-11-21'], '2026-11-21'), set(), now=early)['resolved'] == 0
    later = pd.Timestamp('2026-11-22T13:00', tz='UTC')
    assert pr.resolve_outcomes(log, store(['2026-11-21'], '2026-11-21'), set(), now=later)['resolved'] == 1
    out = [r for r in pr.read_log(log) if r['record_type'] == 'outcome'][0]
    assert out['observed_detection'] == 1 and out['outcome_status'] == 'resolved'
    s = pr.summary(log)
    assert s['accuracy'] is None and 'NOT AVAILABLE' in s['accuracy_status'] and s['chain']['ok']


def test_prospective_gap_day_is_unresolvable_not_negative(tmp_path):
    from fireaq import prospective as pr
    log = tmp_path / 'log.jsonl'
    pr.log_predictions(_pred_file(tmp_path), log, 'fp1')
    st = pd.DataFrame([{'latitude': 30.1, 'longitude': -90.1, 'acq_date': '2026-11-22', 'acq_time': '0900',
                        'confidence': 'nominal', 'frp': 1.0, 'daynight': 'N'}])
    pr.resolve_outcomes(log, st, set(), now=pd.Timestamp('2026-11-22T13:00', tz='UTC'))
    out = [r for r in pr.read_log(log) if r['record_type'] == 'outcome'][0]
    assert out['outcome_status'] == 'unresolvable_gap' and out['observed_detection'] is None
