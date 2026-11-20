"""ALL observations in this test module are explicitly synthetic.
Tests are for code correctness, not for wildfire forecast skill.
"""
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fireaq.core import Grid, clean_firms, make_dataset, FEATURES
from fireaq.model import train_and_evaluate, score_predictions
from fireaq.sources import weather_json_to_daily


def synthetic_fire(date_,lat=40.25,lon=-123.75,hour=12,frp=12.4):
    return {'latitude':lat,'longitude':lon,'acq_date':date_,
            'acq_time':hour*100,'confidence':'n','frp':frp,'type':0}


def synthetic_weather(grid,start,end):
    rows=[]
    for row in grid.cells().itertuples():
        for d in pd.date_range(start,end):
            rows.append({'cell_id':row.cell_id,'date':str(d.date()),'temp_c':20.0,
                         'rh_pct':40.0,'precip_mm':1.5,'wind_ms':2.0,'n_hours':24})
    return pd.DataFrame(rows)


def test_grid_and_product_type_filter():
    g=Grid(-124,40,-123,41,0.5)
    assert len(g.cells())==4
    raw=pd.DataFrame([synthetic_fire('2022-06-01'),
                      {**synthetic_fire('2022-06-01',40.75,-123.25,23),'type':2},
                      {**synthetic_fire('2022-06-01',40.75,-123.25,20),'confidence':'l'}])
    c=clean_firms(raw,g)
    assert len(c)==1
    assert c.iloc[0]['cell_id']=='r0_c0'


def test_fire_latency_at_midnight():
    g=Grid(-124,40,-123.5,40.5,0.5)
    a=pd.DataFrame([synthetic_fire('2022-05-04',hour=23)])
    c=clean_firms(a,g)
    # 23:00 observation + 3h = May 5 02:00, first usable at May 6 midnight.
    assert str(c.iloc[0]['first_usable_issue'].date())=='2022-05-06'


def test_dataset_has_no_future_leakage_and_new_detection_label():
    g=Grid(-124,40,-123,40.5,0.5)
    # Pre-training padding event outside region; 2 valid events, one for target day.
    fires=pd.DataFrame([synthetic_fire('2022-05-12',40.25,-123.75,12),
                        synthetic_fire('2022-05-20',40.25,-123.25,10)])
    wx=synthetic_weather(g,'2022-04-25','2022-05-23')
    out=make_dataset(fires,wx,g,'2022-05-10','2022-05-20')
    row=out[(out['issue_date']=='2022-05-11') & (out['cell_id']=='r0_c0')].iloc[0]
    # May 12 detection is the May 11 issue day's target (24-48h), not a feature.
    assert row['target_nextday_detection']==1
    assert row['neighbor_fire_3d']==0
    # May 12 detection becomes known May 13 midnight. May 14 issue is not eligible.
    assert len(out[(out['issue_date']=='2022-05-14') & (out['cell_id']=='r0_c0')])==0
    # May 20 detection belongs to another cell, not to r0c0.
    assert out[(out['issue_date']=='2022-05-19') & (out['cell_id']=='r0_c1')].iloc[0]['target_nextday_detection']==1


def test_missing_weather_not_imputed_as_zero():
    g=Grid(-124,40,-123.5,40.5,0.5)
    fires=pd.DataFrame([synthetic_fire('2022-05-05')])
    wx=synthetic_weather(g,'2022-04-20','2022-05-20')
    wx.loc[wx['date']=='2022-05-04','n_hours']=10
    out=make_dataset(fires,wx,g,'2022-05-05','2022-05-09')
    # May 6 uses lag-2 May 4, which is incomplete => omit the row.
    assert '2022-05-06' not in set(out['issue_date'])


def test_weather_24h_integrity():
    times=pd.date_range('2022-07-01',periods=48,freq='h')
    payload={'hourly':{'time':times.strftime('%Y-%m-%dT%H:%M').tolist(),
                       'temperature_2m':[20]*48,'relative_humidity_2m':[50]*48,
                       'precipitation':[1]*48,'wind_speed_10m':[2]*48}}
    out=weather_json_to_daily(payload,'r0_c0')
    assert len(out)==2 and set(out['n_hours'])=={24}
    assert np.allclose(out['precip_mm'],24)
    payload['hourly']['time']=payload['hourly']['time'][:-1]
    for x in ['temperature_2m','relative_humidity_2m','precipitation','wind_speed_10m']:
        payload['hourly'][x]=payload['hourly'][x][:-1]
    short=weather_json_to_daily(payload,'r0_c0')
    assert pd.isna(short.iloc[-1]['precip_mm'])


def test_temporal_training_and_no_fire_ablation(tmp_path):
    # 1,800 purely SYNTHETIC days, balanced-ish synthetic outcomes.
    rng=np.random.default_rng(18)
    dates=pd.date_range('2020-01-01','2024-12-30')
    rows=[]
    for d in dates:
        for cell_id in ('r0_c0','r0_c1'):
            z=float(rng.normal())
            label=int(rng.random() < (0.20 if z>0 else 0.035))
            row={x:float(rng.normal()) for x in FEATURES}
            row.update({'cell_id':cell_id,'issue_date':str(d.date()),
                        'target_date':str((d+pd.Timedelta(days=1)).date()),
                        'target_nextday_detection':label,
                        'latitude':40.25,'longitude':-123.25 if cell_id=='r0_c1' else -123.75,
                        'neighbor_fire_3d':max(0,z),'neighbor_fire_7d':max(0,z+0.1),
                        'neighbor_frp_3d':max(0,z*3)})
            rows.append(row)
    df=pd.DataFrame(rows)
    path=tmp_path/'model.joblib'
    stats=train_and_evaluate(df,path,valid_year=2023,test_year=2024)
    assert path.exists()
    assert stats['test_n']>600
    assert stats['baseline']['test']['positives']>5
    scores,meta=score_predictions(path,df.head(4))
    assert len(scores)==4
    assert scores['probability_nextday_detection'].between(0,1).all()
    assert meta['status']=='experimental_research_model'


def test_empty_firms_refused():
    g=Grid(-124,40,-123.5,40.5,0.5)
    with pytest.raises(ValueError,match='empty'):
        make_dataset(pd.DataFrame(),synthetic_weather(g,'2022-01-01','2022-01-21'),
                     g,'2022-01-10','2022-01-11')


def test_duplicate_hourly_weather_timestamps_refused():
    times=['2022-07-01T01:00']*24
    payload={'hourly':{'time':times,'temperature_2m':[20]*24,
         'relative_humidity_2m':[40]*24,'precipitation':[1]*24,
         'wind_speed_10m':[2]*24}}
    with pytest.raises(ValueError,match='duplicate'):
        weather_json_to_daily(payload,'r0_c0')


def test_api_no_predictions_does_not_fabricate(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from fireaq import server
    monkeypatch.setenv('FIRERISK_PREDICTIONS',str(tmp_path/'does_not_exist.json'))
    assert server.health()['predictions_exist'] is False
    with pytest.raises(HTTPException) as e:
        server.predictions()
    assert e.value.status_code == 503
