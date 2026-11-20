"""ALL observations in this test module are explicitly synthetic.
Tests are for code correctness, not for wildfire forecast skill.
"""
import json

import numpy as np
import pandas as pd
import pytest

from fireaq.core import FEATURES, Grid, build_static_source_mask, clean_firms, make_dataset
from fireaq.experiment import (Climatology, SPLIT, assign_splits, count_events, run_development,
                               run_final_test, write_split_manifest)
from fireaq.model import score_predictions
from fireaq.sources import (SourceError, _redact, firms_day_status, overpass_hours_utc, save_raw,
                            weather_json_to_daily)


def synthetic_fire(date_, lat=40.25, lon=-123.75, hour=12, frp=12.4, minute=0):
    return {'latitude': lat, 'longitude': lon, 'acq_date': date_,
            'acq_time': f'{hour:02d}{minute:02d}', 'confidence': 'n', 'frp': frp, 'type': 0}


def synthetic_weather(grid, start, end, rain=1.5):
    rows = []
    for row in grid.cells().itertuples():
        for d in pd.date_range(start, end):
            rows.append({'cell_id': row.cell_id, 'date': str(d.date()), 'temp_c': 20.0, 'tmax_c': 25.0,
                         'rh_pct': 40.0, 'rhmin_pct': 20.0, 'precip_mm': rain, 'wind_ms': 2.0,
                         'windmax_ms': 4.0, 'vpd_kpa': 1.2, 'cloud_pct': 30.0, 'cloud_overpass_pct': 30.0,
                         'n_hours': 24})
    return pd.DataFrame(rows)


def test_grid_and_product_type_filter():
    g = Grid(-124, 40, -123, 41, 0.5)
    assert len(g.cells()) == 4
    raw = pd.DataFrame([synthetic_fire('2022-06-01'),
                        {**synthetic_fire('2022-06-01', 40.75, -123.25, 23), 'type': 2},
                        {**synthetic_fire('2022-06-01', 40.75, -123.25, 20), 'confidence': 'l'}])
    c = clean_firms(raw, g)
    assert len(c) == 1
    assert c.iloc[0]['cell_id'] == 'r0_c0'


def test_nrt_without_type_uses_static_mask():
    g = Grid(-124, 40, -123, 41, 0.5)
    archive = pd.DataFrame([{**synthetic_fire('2021-06-01', 40.751, -123.251), 'type': 2}])
    mask = build_static_source_mask(archive)
    nrt = pd.DataFrame([synthetic_fire('2026-11-01', 40.752, -123.252), synthetic_fire('2026-11-01')]).drop(columns='type')
    nrt['confidence'] = 'nominal'
    c = clean_firms(nrt, g, static_mask=mask)
    assert list(c['cell_id']) == ['r0_c0']


@pytest.mark.parametrize('hour,minute,expected', [(23, 0, '2022-05-06'), (20, 59, '2022-05-05'),
                                                  (21, 0, '2022-05-05'), (21, 1, '2022-05-06')])
def test_fire_latency_at_midnight(hour, minute, expected):
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    c = clean_firms(pd.DataFrame([synthetic_fire('2022-05-04', hour=hour, minute=minute)]), g)
    # observation + 3 h, rounded up to the next UTC midnight (exactly midnight counts as usable)
    assert str(c.iloc[0]['first_usable_issue'].date()) == expected


def test_dataset_has_no_future_leakage_and_new_detection_label():
    g = Grid(-124, 40, -123, 40.5, 0.5)
    fires = pd.DataFrame([synthetic_fire('2022-05-12', 40.25, -123.75, 12),
                          synthetic_fire('2022-05-20', 40.25, -123.25, 10)])
    wx = synthetic_weather(g, '2022-03-01', '2022-05-23')
    out = make_dataset(fires, wx, g, '2022-05-10', '2022-05-20')
    row = out[(out['issue_date'] == '2022-05-11') & (out['cell_id'] == 'r0_c0')].iloc[0]
    assert row['target_nextday_detection'] == 1        # May 12 detection is the May 11 issue's target
    assert row['neighbor_fire_3d'] == 0 and row['region_fire_3d'] == 0
    assert row['target_start_utc'] == '2022-05-12T00:00:00Z' and row['target_end_utc'] == '2022-05-13T00:00:00Z'
    # May 12 detection becomes known at May 13 00:00; issues May 13..19 are not eligible.
    assert len(out[(out['issue_date'] == '2022-05-14') & (out['cell_id'] == 'r0_c0')]) == 0
    assert len(out[(out['issue_date'] == '2022-05-19') & (out['cell_id'] == 'r0_c0')]) == 0
    assert len(out[(out['issue_date'] == '2022-05-20') & (out['cell_id'] == 'r0_c0')]) == 1
    # ...but the neighbour sees it from May 13 issue onwards, not before.
    nb = out[out['cell_id'] == 'r0_c1'].set_index('issue_date')
    assert nb.loc['2022-05-12', 'neighbor_fire_3d'] == 0
    assert nb.loc['2022-05-13', 'neighbor_fire_3d'] == 1
    assert out[(out['issue_date'] == '2022-05-19') & (out['cell_id'] == 'r0_c1')].iloc[0]['target_nextday_detection'] == 1


def test_windows_cross_year_boundary():
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    fires = pd.DataFrame([synthetic_fire('2021-12-29', hour=10), synthetic_fire('2022-01-02', hour=10)])
    wx = synthetic_weather(g, '2021-10-01', '2022-01-31')
    out = make_dataset(fires, wx, g, '2021-12-25', '2022-01-20').set_index('issue_date')
    assert out.loc['2021-12-28', 'target_nextday_detection'] == 1
    assert '2021-12-31' not in out.index and '2022-01-01' not in out.index   # Dec 29 known Dec 30
    assert '2022-01-06' not in out.index                     # Jan 2 detection known Jan 3
    assert out.loc['2022-01-15', 'own_fire_30d'] == 2        # both detections now 8..30 days old
    assert out.loc['2021-12-29', 'rain_30d_lag2'] == pytest.approx(45.0)
    assert out.loc['2021-12-29', 'target_nextday_detection'] == 0


def test_neighbour_padding_sees_fires_outside_bbox():
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    fires = pd.DataFrame([synthetic_fire('2022-05-12', 40.25, -123.25), synthetic_fire('2022-03-01')])
    wx = synthetic_weather(g, '2022-03-01', '2022-05-30')
    unpadded = make_dataset(fires, wx, g, '2022-05-14', '2022-05-14').iloc[0]
    padded = make_dataset(fires, wx, g, '2022-05-14', '2022-05-14', pad_cells=2).iloc[0]
    assert unpadded['neighbor_fire_3d'] == 0 and padded['neighbor_fire_3d'] == 1


def test_missing_firms_day_is_not_a_negative():
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    fires = pd.DataFrame([synthetic_fire('2022-03-01')])
    wx = synthetic_weather(g, '2022-02-01', '2022-05-30')
    cov = pd.DataFrame({'date': pd.date_range('2022-02-01', '2022-05-30').strftime('%Y-%m-%d'),
                        'n_day': 500, 'n_night': 300, 'n_all': 800})
    cov.loc[cov['date'] == '2022-05-12', ['n_day', 'n_night', 'n_all']] = 0
    st = firms_day_status(cov, '2022-02-01', '2022-05-30')
    assert st.set_index('date').loc['2022-05-12', 'status'] == 'missing'
    out = make_dataset(fires, wx, g, '2022-05-08', '2022-05-25', day_status=st, keep_ineligible=True).set_index('issue_date')
    assert out.loc['2022-05-11', 'label_status'] == 'firms_target_day_not_ok'
    assert np.isnan(out.loc['2022-05-11', 'target_nextday_detection'])
    assert out.loc['2022-05-15', 'eligibility_uncertain'] and not out.loc['2022-05-21', 'eligibility_uncertain']


def test_missing_weather_not_imputed_as_zero():
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    fires = pd.DataFrame([synthetic_fire('2022-03-05')])
    wx = synthetic_weather(g, '2022-03-01', '2022-05-20')
    wx.loc[wx['date'] == '2022-05-04', 'n_hours'] = 10
    out = make_dataset(fires, wx, g, '2022-05-05', '2022-05-09')
    assert '2022-05-06' not in set(out['issue_date'])          # lag-2 day incomplete -> row omitted


def test_duplicate_weather_rows_refused():
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    wx = synthetic_weather(g, '2022-03-01', '2022-05-20')
    with pytest.raises(ValueError, match='duplicated'):
        make_dataset(pd.DataFrame([synthetic_fire('2022-03-05')]), pd.concat([wx, wx.tail(1)]), g,
                     '2022-05-05', '2022-05-09')


def test_weather_24h_integrity_and_overpass_cloud():
    times = pd.date_range('2022-07-01', periods=48, freq='h')
    payload = {'hourly': {'time': times.strftime('%Y-%m-%dT%H:%M').tolist(),
                          'temperature_2m': [20]*48, 'relative_humidity_2m': [50]*48,
                          'precipitation': [1]*48, 'wind_speed_10m': [2]*48,
                          'cloud_cover': [0]*10 + [100] + [0]*37}}
    out = weather_json_to_daily(payload, 'r0_c0', lon=-121.0)
    assert len(out) == 2 and set(out['n_hours']) == {24}
    assert np.allclose(out['precip_mm'], 24)
    assert overpass_hours_utc(-121.0) == (10, 22)
    assert out.iloc[0]['cloud_overpass_pct'] == 50.0          # hours 10 (100%) and 22 (0%)
    for x in payload['hourly']:
        payload['hourly'][x] = payload['hourly'][x][:-1]
    short = weather_json_to_daily(payload, 'r0_c0')
    assert pd.isna(short.iloc[-1]['precip_mm'])


def test_duplicate_hourly_weather_timestamps_refused():
    payload = {'hourly': {'time': ['2022-07-01T01:00']*24, 'temperature_2m': [20]*24,
                          'relative_humidity_2m': [40]*24, 'precipitation': [1]*24, 'wind_speed_10m': [2]*24}}
    with pytest.raises(ValueError, match='duplicate'):
        weather_json_to_daily(payload, 'r0_c0')


def test_empty_firms_refused():
    g = Grid(-124, 40, -123.5, 40.5, 0.5)
    with pytest.raises(ValueError, match='empty'):
        make_dataset(pd.DataFrame(), synthetic_weather(g, '2022-01-01', '2022-01-21'), g, '2022-01-10', '2022-01-11')


def test_secret_redaction(tmp_path):
    secret = 'abcdef0123456789abcdef0123456789'
    manifest = []
    save_raw(tmp_path, 'x.csv', b'a,b\n', 'FIRMS', f'https://host/api/area/csv/{secret}/X/1', manifest, key=secret)
    text = (tmp_path / 'x.csv.meta.json').read_text() + json.dumps(manifest)
    assert secret not in text and '[REDACTED_FIRMS_KEY]' in text
    assert secret not in _redact(f'error at {secret}', secret)
    assert issubclass(SourceError, RuntimeError)


def test_splits_embargo_and_disjointness(tmp_path):
    dates = pd.date_range('2019-01-01', '2025-12-30')
    df = pd.DataFrame({'issue_date': dates.strftime('%Y-%m-%d'),
                       'target_date': (dates + pd.Timedelta(days=1)).strftime('%Y-%m-%d')})
    s = assign_splits(df)
    t = pd.to_datetime(df['target_date'])
    assert t[s == 'train'].max() == pd.Timestamp('2022-12-31')
    assert t[s == 'validation'].min() == pd.Timestamp('2023-01-01') + pd.Timedelta(days=SPLIT['embargo_days'])
    assert t[s == 'test'].min() == pd.Timestamp('2024-01-01') + pd.Timedelta(days=SPLIT['embargo_days'])
    assert (s == 'embargo').sum() == 2 * SPLIT['embargo_days']


def test_event_clustering():
    df = pd.DataFrame({'cell_id': ['r0_c0', 'r0_c1', 'r5_c5', 'r0_c0'],
                       'target_date': ['2022-07-01', '2022-07-02', '2022-07-01', '2022-08-01'],
                       'target_nextday_detection': [1, 1, 1, 1]})
    assert count_events(df) == 3


def test_climatology_falls_back_for_unseen_cells():
    X = pd.DataFrame({'latitude': [40.25]*4, 'longitude': [-121.25]*4,
                      'doy_sin': [np.sin(2*np.pi*200/365.25)]*4, 'doy_cos': [np.cos(2*np.pi*200/365.25)]*4})
    m = Climatology(prior_strength=0).fit(X, [1, 0, 0, 0])
    assert m.predict_proba(X)[0, 1] == pytest.approx(0.25)
    X2 = X.assign(latitude=41.75)
    assert m.predict_proba(X2)[0, 1] == pytest.approx(0.25)


def _synthetic_model_frame():
    rng = np.random.default_rng(18)
    dates = pd.date_range('2019-01-01', '2025-12-30')
    rows = []
    for d in dates:
        for iy, ix in [(0, 0), (0, 1), (3, 3), (3, 4)]:
            z = float(rng.normal())
            row = {x: float(rng.normal()) for x in FEATURES}
            row.update({'cell_id': f'r{iy}_c{ix}', 'issue_date': str(d.date()),
                        'target_date': str((d + pd.Timedelta(days=1)).date()),
                        'target_nextday_detection': float(rng.random() < (0.25 if z > 0 else 0.03)),
                        'target_nextday_count': 0.0, 'latitude': 39.75 + iy, 'longitude': -122.75 + ix,
                        'doy_sin': np.sin(2*np.pi*d.dayofyear/365.25), 'doy_cos': np.cos(2*np.pi*d.dayofyear/365.25),
                        'neighbor_fire_3d': max(0, z), 'neighbor_fire_7d': max(0, z+0.1),
                        'neighbor_frp_3d': max(0, z*3), 'ring2_fire_7d': 0.0, 'own_fire_30d': 0.0,
                        'region_fire_3d': 1.0, 'rain_3d_lag2': 1.0, 'rain_7d_lag2': 2.0, 'rain_30d_lag2': 3.0,
                        'target_cloud_obscured_proxy': False, 'detection_on_issue_day': False})
            row['target_nextday_count'] = row['target_nextday_detection']
            rows.append(row)
    return pd.DataFrame(rows)


def test_development_then_single_final_test(tmp_path, monkeypatch):
    import fireaq.experiment as ex
    small = {k: v for k, v in ex.candidate_models().items()
             if k in ('dummy_prior', 'dummy_most_frequent', 'climatology_cell_month', 'logistic_l2')}
    monkeypatch.setattr(ex, 'candidate_models', lambda: small)
    df = _synthetic_model_frame()
    meta = run_development(df, tmp_path, log=lambda *_: None)
    assert meta['test_evaluated'] is False
    man = json.loads((tmp_path / 'splits' / 'split_manifest.json').read_text())
    assert all(v == 0 for v in man['pairwise_key_overlap'].values())
    board = pd.read_csv(tmp_path / 'validation_leaderboard.csv')
    assert board.iloc[0]['model'] == meta['champion']['model']
    assert not any(c.startswith('test_') for c in board.columns)       # no test metrics during development
    ex.run_regression_development(df, tmp_path, log=lambda *_: None)
    res = run_final_test(df, tmp_path, log=lambda *_: None)
    assert res['champion_test']['n'] > 500
    with pytest.raises(RuntimeError, match='already been run'):
        run_final_test(df, tmp_path, log=lambda *_: None)
    scores, m = score_predictions(tmp_path / 'model.joblib', df.head(4))
    assert scores['probability_nextday_detection'].between(0, 1).all() and m['test_evaluated']
    assert set(res['regression_log1p_count_test']) >= {'mean_constant', 'median_constant', 'linear_regression'}


def test_split_manifest_fails_without_positives(tmp_path):
    df = _synthetic_model_frame()
    df['split'] = assign_splits(df)
    df.loc[df.split == 'validation', 'target_nextday_detection'] = 0.0
    with pytest.raises(ValueError, match='lacks labels'):
        write_split_manifest(df, tmp_path)


def test_api_no_predictions_does_not_fabricate(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fireaq import server
    monkeypatch.setenv('FIRERISK_PREDICTIONS', str(tmp_path / 'does_not_exist.json'))
    monkeypatch.setenv('FIRERISK_OUTPUT', str(tmp_path))
    c = TestClient(server.app)
    assert c.get('/health').json()['predictions_exist'] is False
    r = c.get('/api/predictions')
    assert r.status_code == 503
    assert c.get('/api/model').status_code == 503
    assert 'Not for emergency' in c.get('/').text


def test_api_serves_and_validates_predictions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fireaq import server
    p = tmp_path / 'predictions.json'
    p.write_text(json.dumps({'generated_at_utc': '2026-11-19T01:00:00+00:00', 'issue_time_utc': '2026-11-19T00:00:00Z',
                             'target_date': '2026-11-20', 'forecast_status': 'EXPERIMENTAL',
                             'predictions': [{'cell_id': 'r0_c0', 'latitude': 38.75, 'longitude': -123.25,
                                              'probability_nextday_detection': 0.01}]}))
    monkeypatch.setenv('FIRERISK_PREDICTIONS', str(p))
    c = TestClient(server.app)
    body = c.get('/api/predictions').json()
    assert body['predictions'][0]['cell_id'] == 'r0_c0' and 'stale' in body
    assert c.get('/api/predictions?cell_id=nope').status_code == 404
    p.write_text('{"predictions": [{"probability_nextday_detection": 7}]}')
    assert c.get('/api/predictions').status_code == 500
