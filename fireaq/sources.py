"""Small, conservative NASA FIRMS and Open-Meteo data clients.

No fabricated observations, no hidden retries that turn failures into zeros.
Raw responses saved with SHA-256 checksums; API keys redacted in manifests.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

FIRMS_API = 'https://firms.modaps.eosdis.nasa.gov/api'
ARCHIVE_API = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_API = 'https://api.open-meteo.com/v1/forecast'
HOURLY = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m']


def _get(url: str, params: dict | None = None, timeout=90) -> requests.Response:
    """Fail loudly on 400/401/403/404 and after bounded 429/5xx retry."""
    for attempt in range(3):
        r = requests.get(url, params=params, timeout=timeout,
                         headers={'User-Agent': 'FireRiskOpen/0.1 academic research (contact: repository maintainer)'})
        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt < 2:
                time.sleep(2 ** attempt + 1)
                continue
        r.raise_for_status()
        return r
    raise RuntimeError('unreachable')


def save_raw(outdir: Path, filename: str, content: bytes, source: str, url: str,
             manifest: list, key: str = '') -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / filename
    if dst.exists() and dst.read_bytes() != content:
        # Keep NRT revision snapshots instead of overwriting original source bytes.
        stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        dst = outdir / f'{Path(filename).stem}_{stamp}{Path(filename).suffix}'
    if not dst.exists():
        dst.write_bytes(content)
    manifest.append({
        'source': source, 'file': str(dst), 'sha256': hashlib.sha256(content).hexdigest(),
        'bytes': len(content), 'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'request_url_redacted': url.replace(key, '[REDACTED_FIRMS_KEY]') if key else url,
    })


def fetch_firms(key: str, bbox: tuple[float,float,float,float], start: date, end: date,
                source: str, rawdir: Path, manifest: list, sleep_seconds=0.5) -> pd.DataFrame:
    """Inclusive dates, ≤5-day chunks (safe under both regional/global API variants)."""
    if not key or '/' in key or '\\' in key:
        raise ValueError('Set a valid FIRMS_MAP_KEY in your environment')
    if source not in {'VIIRS_NOAA20_SP','VIIRS_SNPP_SP','VIIRS_NOAA20_NRT','VIIRS_SNPP_NRT'}:
        raise ValueError('Unsupported FIRMS source; use SP for historical, NRT for live')
    b = ','.join(f'{v:g}' for v in bbox)
    parts = []
    day = start
    while day <= end:
        n = min(5, (end - day).days + 1)
        url = f'{FIRMS_API}/area/csv/{key}/{source}/{b}/{n}/{day:%Y-%m-%d}'
        r = _get(url, timeout=150)
        save_raw(rawdir / 'firms', f'{source}_{day:%Y%m%d}_{n}d.csv', r.content,
                 source, r.url, manifest, key=key)
        # API errors may be returned as HTTP 200 plain text; never parse as empty data.
        try:
            data = pd.read_csv(io.BytesIO(r.content))
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f'FIRMS returned empty response for {day} {source}') from exc
        must = {'latitude', 'longitude', 'acq_date', 'acq_time', 'confidence', 'frp'}
        if not must.issubset(data.columns):
            raise ValueError(f'Unexpected FIRMS payload: {list(data.columns)}; response begins {r.text[:100]}')
        data['source_product'] = source
        parts.append(data)
        day += timedelta(days=n)
        time.sleep(sleep_seconds)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def weather_json_to_daily(payload: dict, cell_id: str) -> pd.DataFrame:
    hourly = payload.get('hourly', {})
    needed = set(HOURLY + ['time'])
    if not needed.issubset(hourly):
        raise ValueError(f'Open-Meteo missing columns {needed - set(hourly)}')
    h = pd.DataFrame({k: hourly[k] for k in needed})
    h['timestamp'] = pd.to_datetime(h['time'], utc=True)
    h['date'] = h['timestamp'].dt.strftime('%Y-%m-%d')
    if h['timestamp'].duplicated().any():
        raise ValueError('Open-Meteo returned duplicate hourly timestamps; refusing to count duplicates as hours')
    for k in HOURLY:
        h[k] = pd.to_numeric(h[k], errors='coerce')
    g = h.groupby('date', sort=True)
    result = g.agg(
        temp_c=('temperature_2m', 'mean'),
        rh_pct=('relative_humidity_2m', 'mean'),
        precip_mm=('precipitation', 'sum'),
        wind_ms=('wind_speed_10m', 'mean'),
        n_hours=('temperature_2m', 'count'),
    ).reset_index()
    # Each variable must have 24 real hourly values; incomplete rainfall sums are invalid.
    valid_by_day = g[HOURLY].count()
    complete_days = valid_by_day.eq(24).all(axis=1)
    for k in ['temp_c', 'rh_pct', 'precip_mm', 'wind_ms']:
        result.loc[~result['date'].isin(complete_days[complete_days].index), k] = float('nan')
    result['cell_id'] = cell_id
    return result[['cell_id', 'date', 'temp_c', 'rh_pct', 'precip_mm', 'wind_ms', 'n_hours']]


def fetch_weather_historical(cell_id: str, lat: float, lon: float, start: date, end: date,
                             rawdir: Path, manifest: list) -> pd.DataFrame:
    params = {'latitude': lat, 'longitude': lon, 'start_date': str(start), 'end_date': str(end),
              'hourly': ','.join(HOURLY), 'models': 'era5', 'timezone': 'GMT',
              'wind_speed_unit': 'ms', 'cell_selection': 'nearest'}
    r = _get(ARCHIVE_API, params=params, timeout=150)
    save_raw(rawdir / 'weather', f'{cell_id}_ERA5.json', r.content, 'Open-Meteo ERA5', r.url, manifest)
    payload = r.json()
    if payload.get('error'):
        raise ValueError(payload.get('reason', 'Open-Meteo error'))
    return weather_json_to_daily(payload, cell_id)


def fetch_weather_live(cell_id: str, lat: float, lon: float, rawdir: Path, manifest: list) -> pd.DataFrame:
    params = {'latitude': lat, 'longitude': lon, 'hourly': ','.join(HOURLY),
              'past_days': 10, 'forecast_days': 2, 'timezone': 'GMT', 'wind_speed_unit': 'ms'}
    r = _get(FORECAST_API, params=params)
    save_raw(rawdir / 'weather', f'{cell_id}_live.json', r.content, 'Open-Meteo forecast', r.url, manifest)
    payload = r.json()
    if payload.get('error'):
        raise ValueError(payload.get('reason', 'Open-Meteo error'))
    return weather_json_to_daily(payload, cell_id)


def write_manifest(path: Path, manifest: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
