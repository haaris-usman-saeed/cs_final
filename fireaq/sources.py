"""Conservative NASA FIRMS, Open-Meteo and Natural Earth data clients.

Rules enforced here:
* Nothing is fabricated. An API failure raises ``SourceError``; it never becomes an empty table.
* Every raw response is written byte-for-byte under ``raw/`` with a sidecar ``*.meta.json``
  holding the SHA-256, byte count, UTC retrieval time, redacted request URL, product and schema.
* Downloads are cached: if the raw file and its sidecar exist and the hash matches, the file is
  reused (restart/resume). Partial downloads are written to ``*.part`` and renamed only on success.
* The FIRMS MAP_KEY is never written to disk, logs or exception messages.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FIRMS_BASE = 'https://firms.modaps.eosdis.nasa.gov'
FIRMS_API = f'{FIRMS_BASE}/api'
ARCHIVE_API = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_API = 'https://api.open-meteo.com/v1/forecast'
HIST_FORECAST_API = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
NE_GEOJSON = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/{name}.geojson'

REQUIRED_HOURLY = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m']
OPTIONAL_HOURLY = ['cloud_cover']
HOURLY = REQUIRED_HOURLY + OPTIONAL_HOURLY
USER_AGENT = 'FireRiskOpen/0.2 research data client'

# Yearly FIRMS country archive folder names (verified 2026-11-19).
ARCHIVE_SATELLITES = {'noaa20': 'viirs-jpss1', 'snpp': 'viirs-snpp', 'modis': 'modis'}
NRT_FILES = {
    'noaa20': 'data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_{region}_{window}.csv',
    'snpp': 'data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_{region}_{window}.csv',
}
FIRMS_REQUIRED = {'latitude', 'longitude', 'acq_date', 'acq_time', 'confidence', 'frp'}


class SourceError(RuntimeError):
    """An upstream source failed. Never convert this into an empty/negative dataset."""


def _redact(text: str, secret: str = '') -> str:
    return text.replace(secret, '[REDACTED_FIRMS_KEY]') if secret else text


def _get(url: str, params: dict | None = None, timeout: int = 120, secret: str = '',
         max_attempts: int = 8, stream: bool = False) -> requests.Response:
    """GET with bounded retries for 429/5xx/network errors. Errors never echo the secret."""
    last = ''
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout, stream=stream,
                             headers={'User-Agent': USER_AGENT})
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = f'{type(exc).__name__}'
            time.sleep(min(120, 5 * 2 ** attempt))
            continue
        if r.status_code == 429 or 500 <= r.status_code < 600:
            last = f'HTTP {r.status_code}: {_redact(r.text[:200], secret)}'
            wait = r.headers.get('Retry-After')
            if wait and wait.isdigit():
                delay = int(wait)
            elif r.status_code == 429 and 'Hourly' in r.text:
                delay = 600
            elif r.status_code == 429:
                delay = 65
            else:
                delay = min(120, 5 * 2 ** attempt)
            time.sleep(delay)
            continue
        if r.status_code >= 400:
            raise SourceError(f'HTTP {r.status_code} from {_redact(r.url, secret)}: '
                              f'{_redact(r.text[:300], secret)}')
        return r
    raise SourceError(f'Giving up after {max_attempts} attempts on {_redact(url, secret)} ({last})')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + '.meta.json')


def cached_entry(path: Path) -> dict | None:
    """Return the sidecar entry if the raw file exists and its hash still matches."""
    mp = _meta_path(path)
    if not (path.exists() and mp.exists()):
        return None
    meta = json.loads(mp.read_text())
    if sha256_file(path) != meta.get('sha256'):
        raise SourceError(f'Cached raw file {path} no longer matches its recorded SHA-256; '
                          'delete it and its .meta.json to re-download.')
    return {**meta, 'cache_hit': True}


def save_raw(outdir: Path, filename: str, content: bytes, source: str, url: str,
             manifest: list, key: str = '', **extra) -> Path:
    """Write immutable raw bytes plus sidecar metadata; append the entry to ``manifest``.

    If a different payload already exists under ``filename`` (e.g. a revised NRT file), the new
    bytes are stored under a timestamped name instead of overwriting the original.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / filename
    if dst.exists() and dst.read_bytes() != content:
        stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        dst = outdir / f'{Path(filename).stem}_{stamp}{Path(filename).suffix}'
    if not dst.exists():
        dst.write_bytes(content)
    entry = {
        'source': source, 'file': str(dst), 'sha256': hashlib.sha256(content).hexdigest(),
        'bytes': len(content), 'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'request_url_redacted': _redact(url, key), **extra,
    }
    _meta_path(dst).write_text(json.dumps(entry, indent=2) + '\n')
    manifest.append({**entry, 'cache_hit': False})
    return dst


# --------------------------------------------------------------------------- FIRMS
def fetch_firms_country_year(year: int, rawdir: Path, manifest: list, satellite: str = 'noaa20',
                             country: str = 'United_States') -> Path:
    """Download (or reuse) one keyless FIRMS yearly country archive CSV, streaming to disk."""
    folder = ARCHIVE_SATELLITES[satellite]
    rel = f'data/country/{folder}/{year}/{folder}_{year}_{country}.csv'
    url = f'{FIRMS_BASE}/{rel}'
    dst = rawdir / 'firms' / Path(rel).name
    hit = cached_entry(dst)
    if hit:
        manifest.append(hit)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = _get(url, timeout=300, stream=True)
    part = dst.with_name(dst.name + '.part')
    h, n = hashlib.sha256(), 0
    with open(part, 'wb') as fh:
        for block in r.iter_content(1 << 20):
            fh.write(block)
            h.update(block)
            n += len(block)
    expected = r.headers.get('Content-Length')
    if expected and int(expected) != n:
        raise SourceError(f'Truncated download for {url}: {n} of {expected} bytes')
    with open(part, 'rb') as fh:
        header = fh.readline().decode('utf-8', 'replace').strip().split(',')
    if not FIRMS_REQUIRED.issubset(header):
        raise SourceError(f'Unexpected FIRMS archive header in {url}: {header[:20]}')
    part.rename(dst)
    entry = {'source': 'NASA FIRMS yearly country archive', 'product': f'{folder} (VIIRS 375 m active fire)',
             'satellite': satellite, 'year': year, 'country': country, 'file': str(dst),
             'sha256': h.hexdigest(), 'bytes': n,
             'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
             'request_url_redacted': url, 'last_modified_upstream': r.headers.get('Last-Modified'),
             'schema': header}
    _meta_path(dst).write_text(json.dumps(entry, indent=2) + '\n')
    manifest.append({**entry, 'cache_hit': False})
    return dst


def read_firms_archive(path: Path, bbox: tuple[float, float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detections inside bbox, national daily coverage table) from one archive CSV.

    The coverage table counts *all* detections (any type/confidence) per UTC day and day/night
    flag across the whole file. It is used to spot missing or truncated acquisition days.
    """
    df = pd.read_csv(path, dtype={'acq_time': str, 'version': str, 'confidence': str})
    if not FIRMS_REQUIRED.issubset(df.columns):
        raise SourceError(f'{path} missing columns {FIRMS_REQUIRED - set(df.columns)}')
    cov = df.groupby(['acq_date', 'daynight']).size().unstack(fill_value=0)
    cov = cov.reindex(columns=['D', 'N'], fill_value=0).rename(columns={'D': 'n_day', 'N': 'n_night'})
    cov['n_all'] = cov['n_day'] + cov['n_night']
    cov = cov.reset_index().rename(columns={'acq_date': 'date'})
    w, s, e, n = bbox
    reg = df[df['latitude'].between(s, n) & df['longitude'].between(w, e)].copy()
    return reg.reset_index(drop=True), cov


def firms_day_status(coverage: pd.DataFrame, start: str, end: str,
                     min_fraction: float = 0.10) -> pd.DataFrame:
    """Classify each UTC day as ``ok``, ``missing`` (no rows at all) or ``suspect`` (very low count).

    A day missing from a yearly archive is *not* a fire-free day: it is unknown.
    """
    days = pd.DataFrame({'date': pd.date_range(start, end).strftime('%Y-%m-%d')})
    cov = coverage.groupby('date', as_index=False)[['n_day', 'n_night', 'n_all']].sum()
    out = days.merge(cov, on='date', how='left').fillna({'n_day': 0, 'n_night': 0, 'n_all': 0})
    med = out['n_all'].rolling(31, min_periods=7, center=False).median().shift(1)
    out['trailing_median'] = med
    out['status'] = 'ok'
    out.loc[out['n_all'] < min_fraction * med.fillna(0), 'status'] = 'suspect'
    out.loc[(out['n_day'] == 0) | (out['n_night'] == 0), 'status'] = 'suspect'
    out.loc[out['n_all'] == 0, 'status'] = 'missing'
    return out


def fetch_firms(key: str, bbox: tuple[float, float, float, float], start: date, end: date,
                source: str, rawdir: Path, manifest: list, sleep_seconds=0.5) -> pd.DataFrame:
    """MAP_KEY area API (inclusive dates, ≤5-day chunks, cached per chunk)."""
    if not key or '/' in key or '\\' in key:
        raise ValueError('Set a valid FIRMS_MAP_KEY in your environment')
    if source not in {'VIIRS_NOAA20_SP', 'VIIRS_SNPP_SP', 'VIIRS_NOAA21_NRT',
                      'VIIRS_NOAA20_NRT', 'VIIRS_SNPP_NRT'}:
        raise ValueError('Unsupported FIRMS source; use *_SP for historical, *_NRT for live')
    b = ','.join(f'{v:g}' for v in bbox)
    parts = []
    day = start
    while day <= end:
        n = min(5, (end - day).days + 1)
        fname = f'{source}_{b.replace(",", "_")}_{day:%Y%m%d}_{n}d.csv'
        dst = rawdir / 'firms' / fname
        hit = cached_entry(dst)
        if hit:
            content = dst.read_bytes()
            manifest.append(hit)
        else:
            url = f'{FIRMS_API}/area/csv/{key}/{source}/{b}/{n}/{day:%Y-%m-%d}'
            r = _get(url, timeout=150, secret=key)
            content = r.content
            save_raw(rawdir / 'firms', fname, content, source, r.url, manifest, key=key,
                     product=source, first_day=str(day), n_days=n)
        try:
            data = pd.read_csv(io.BytesIO(content), dtype={'acq_time': str})
        except pd.errors.EmptyDataError as exc:
            raise SourceError(f'FIRMS returned an empty body for {day} {source}') from exc
        if not FIRMS_REQUIRED.issubset(data.columns):
            raise SourceError(f'Unexpected FIRMS payload for {day} {source}: '
                              f'{_redact(content[:120].decode("utf-8", "replace"), key)}')
        data['source_product'] = source
        parts.append(data)
        day += timedelta(days=n)
        time.sleep(sleep_seconds)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_firms_nrt_keyless(rawdir: Path, manifest: list, satellite: str = 'noaa20',
                            region: str = 'USA_contiguous_and_Hawaii', window: str = '7d') -> pd.DataFrame:
    """Keyless rolling NRT file. Always re-downloaded and kept as a timestamped snapshot."""
    rel = NRT_FILES[satellite].format(region=region, window=window)
    url = f'{FIRMS_BASE}/{rel}'
    r = _get(url, timeout=150)
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
    save_raw(rawdir / 'firms', f'{Path(rel).stem}_{stamp}.csv', r.content,
             'NASA FIRMS keyless NRT', url, manifest, product=Path(rel).stem,
             last_modified_upstream=r.headers.get('Last-Modified'))
    try:
        data = pd.read_csv(io.BytesIO(r.content), dtype={'acq_time': str, 'version': str})
    except pd.errors.EmptyDataError as exc:
        raise SourceError(f'Empty NRT body from {url}') from exc
    if not FIRMS_REQUIRED.issubset(data.columns):
        raise SourceError(f'Unexpected NRT header from {url}: {list(data.columns)}')
    if data.empty:
        raise SourceError(f'NRT feed {url} has a header but zero rows: the satellite feed appears '
                          'to be down. Refusing to treat this as a fire-free week.')
    data['source_product'] = Path(rel).stem
    return data


# --------------------------------------------------------------------------- weather
def overpass_hours_utc(lon: float) -> tuple[int, int]:
    """Approximate UTC hours of the VIIRS (NOAA-20/S-NPP) ~01:30 and ~13:30 local-solar overpasses."""
    return tuple(int(round((h - lon / 15.0) % 24)) % 24 for h in (1.5, 13.5))


def weather_json_to_daily(payload: dict, cell_id: str, lon: float | None = None) -> pd.DataFrame:
    """Aggregate hourly UTC weather to complete UTC days. Incomplete days become NaN, never zero."""
    hourly = payload.get('hourly', {})
    needed = set(REQUIRED_HOURLY + ['time'])
    if not needed.issubset(hourly):
        raise ValueError(f'Open-Meteo missing columns {needed - set(hourly)}')
    cols = ['time'] + [k for k in HOURLY if k in hourly]
    h = pd.DataFrame({k: hourly[k] for k in cols})
    for k in HOURLY:
        h[k] = pd.to_numeric(h[k], errors='coerce') if k in h else np.nan
    h['timestamp'] = pd.to_datetime(h['time'], utc=True)
    if h['timestamp'].duplicated().any():
        raise ValueError('Open-Meteo returned duplicate hourly timestamps; refusing to count duplicates as hours')
    h['date'] = h['timestamp'].dt.strftime('%Y-%m-%d')
    h['hour'] = h['timestamp'].dt.hour
    t, rh = h['temperature_2m'], h['relative_humidity_2m']
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))            # kPa, Tetens
    h['vpd_kpa'] = es * (1 - rh.clip(0, 100) / 100.0)
    if lon is not None:
        oh = overpass_hours_utc(lon)
        h['cloud_overpass'] = h['cloud_cover'].where(h['hour'].isin(oh))
    else:
        h['cloud_overpass'] = np.nan
    g = h.groupby('date', sort=True)
    result = g.agg(
        temp_c=('temperature_2m', 'mean'), tmax_c=('temperature_2m', 'max'),
        rh_pct=('relative_humidity_2m', 'mean'), rhmin_pct=('relative_humidity_2m', 'min'),
        precip_mm=('precipitation', 'sum'),
        wind_ms=('wind_speed_10m', 'mean'), windmax_ms=('wind_speed_10m', 'max'),
        vpd_kpa=('vpd_kpa', 'mean'), cloud_pct=('cloud_cover', 'mean'),
        cloud_overpass_pct=('cloud_overpass', 'mean'),
        n_hours=('temperature_2m', 'count'),
    ).reset_index()
    complete = g[REQUIRED_HOURLY].count().eq(24).all(axis=1)
    bad = ~result['date'].isin(complete[complete].index)
    result.loc[bad, ['temp_c', 'tmax_c', 'rh_pct', 'rhmin_pct', 'precip_mm', 'wind_ms',
                     'windmax_ms', 'vpd_kpa']] = np.nan
    cloud_complete = g['cloud_cover'].count().eq(24)
    result.loc[~result['date'].isin(cloud_complete[cloud_complete].index), 'cloud_pct'] = np.nan
    result['cell_id'] = cell_id
    return result[['cell_id', 'date', 'temp_c', 'tmax_c', 'rh_pct', 'rhmin_pct', 'precip_mm',
                   'wind_ms', 'windmax_ms', 'vpd_kpa', 'cloud_pct', 'cloud_overpass_pct', 'n_hours']]


def _weather_chunks(start: date, end: date):
    y = start.year
    while y <= end.year:
        a = max(start, date(y, 1, 1))
        b = min(end, date(y, 12, 31))
        yield a, b
        y += 1


def fetch_weather_historical(cell_id: str, lat: float, lon: float, start: date, end: date,
                             rawdir: Path, manifest: list, api: str = 'era5',
                             pace_seconds: float = 6.0) -> pd.DataFrame:
    """ERA5 (``api='era5'``) or archived operational forecasts (``api='historical_forecast'``),
    fetched in calendar-year chunks with per-chunk caching so interrupted runs resume."""
    if api == 'era5':
        base, extra, label = ARCHIVE_API, {'models': 'era5'}, 'ERA5'
    elif api == 'historical_forecast':
        base, extra, label = HIST_FORECAST_API, {}, 'HISTFC'
    else:
        raise ValueError(api)
    frames = []
    for a, b in _weather_chunks(start, end):
        fname = f'{cell_id}_{label}_{a:%Y%m%d}_{b:%Y%m%d}.json'
        dst = rawdir / 'weather' / fname
        hit = cached_entry(dst)
        if hit:
            content = dst.read_bytes()
            manifest.append(hit)
        else:
            params = {'latitude': lat, 'longitude': lon, 'start_date': str(a), 'end_date': str(b),
                      'hourly': ','.join(HOURLY), 'timezone': 'GMT', 'wind_speed_unit': 'ms',
                      'cell_selection': 'nearest', **extra}
            r = _get(base, params=params, timeout=(15, 45))
            content = r.content
            payload = json.loads(content)
            if payload.get('error'):
                raise SourceError(f'Open-Meteo error for {cell_id} {a}..{b}: {payload.get("reason")}')
            save_raw(rawdir / 'weather', fname, content, f'Open-Meteo {label}', r.url, manifest,
                     product=f'open-meteo {api}', cell_id=cell_id, first_day=str(a), last_day=str(b),
                     units=payload.get('hourly_units'), grid_latitude=payload.get('latitude'),
                     grid_longitude=payload.get('longitude'), elevation_m=payload.get('elevation'))
            time.sleep(pace_seconds)
        payload = json.loads(content)
        if payload.get('error'):
            raise SourceError(f'Cached Open-Meteo error payload for {cell_id}: {payload.get("reason")}')
        frames.append(weather_json_to_daily(payload, cell_id, lon=lon))
    out = pd.concat(frames, ignore_index=True)
    if out['date'].duplicated().any():
        raise SourceError(f'Overlapping weather chunks for {cell_id}')
    return out


def fetch_weather_live(cell_id: str, lat: float, lon: float, rawdir: Path, manifest: list,
                       past_days: int = 40) -> pd.DataFrame:
    params = {'latitude': lat, 'longitude': lon, 'hourly': ','.join(HOURLY),
              'past_days': past_days, 'forecast_days': 1, 'timezone': 'GMT', 'wind_speed_unit': 'ms'}
    r = _get(FORECAST_API, params=params)
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
    save_raw(rawdir / 'weather', f'{cell_id}_live_{stamp}.json', r.content, 'Open-Meteo forecast',
             r.url, manifest, product='open-meteo forecast (best match, past_days)', cell_id=cell_id)
    payload = r.json()
    if payload.get('error'):
        raise SourceError(payload.get('reason', 'Open-Meteo error'))
    return weather_json_to_daily(payload, cell_id, lon=lon)


# --------------------------------------------------------------------------- Natural Earth
def fetch_natural_earth(name: str, rawdir: Path, manifest: list) -> Path:
    url = NE_GEOJSON.format(name=name)
    dst = rawdir / 'geo' / f'{name}.geojson'
    hit = cached_entry(dst)
    if hit:
        manifest.append(hit)
        return dst
    r = _get(url, timeout=300)
    return save_raw(rawdir / 'geo', f'{name}.geojson', r.content, 'Natural Earth (public domain)',
                    url, manifest, product=f'Natural Earth 5.1.2 {name}')


def write_manifest(path: Path, manifest: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')


# --------------------------------------------------------------------------- v0.3: NASA POWER, WorldCover, DEM
POWER_API = 'https://power.larc.nasa.gov/api/temporal/daily/point'
POWER_PARAMS = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'QV2M', 'PRECTOTCORR', 'WS10M', 'WS10M_MAX',
                'GWETTOP', 'GWETROOT', 'CLOUD_AMT']
WORLDCOVER = ('https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/'
              'ESA_WorldCover_10m_2020_v100_{tile}_Map.tif')
WORLDCOVER_CLASSES = {10: 'tree', 20: 'shrub', 30: 'grass', 40: 'crop', 50: 'built', 60: 'bare',
                      70: 'snow', 80: 'water', 90: 'wetland', 95: 'mangrove', 100: 'moss'}
ELEVATION_API = 'https://api.open-meteo.com/v1/elevation'


def power_json_to_daily(payload: dict, cell_id: str) -> pd.DataFrame:
    """NASA POWER daily (UTC) JSON -> one row per UTC day. Fill value -999 becomes NaN, never 0."""
    hdr = payload.get('header', {})
    if hdr.get('time_standard') not in (None, 'UTC'):
        raise SourceError(f'POWER returned time standard {hdr.get("time_standard")}, expected UTC')
    par = payload.get('properties', {}).get('parameter', {})
    missing = set(POWER_PARAMS) - set(par)
    if missing:
        raise SourceError(f'POWER payload missing parameters {missing}')
    fill = hdr.get('fill_value', -999.0)
    df = pd.DataFrame({k: pd.Series(v) for k, v in par.items()})
    df = df.replace(fill, np.nan)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    if df.index.duplicated().any():
        raise SourceError('POWER returned duplicate days')
    df = df.sort_index()
    out = pd.DataFrame({
        'cell_id': cell_id, 'date': df.index.strftime('%Y-%m-%d'),
        't2m': df['T2M'], 't2m_max': df['T2M_MAX'], 't2m_min': df['T2M_MIN'], 'rh2m': df['RH2M'],
        'precip_mm': df['PRECTOTCORR'], 'ws10m': df['WS10M'], 'ws10m_max': df['WS10M_MAX'],
        'gwettop': df['GWETTOP'], 'gwetroot': df['GWETROOT'], 'cloud_pct': df['CLOUD_AMT'],
    }).reset_index(drop=True)
    es = 0.6108 * np.exp(17.27 * out['t2m_max'] / (out['t2m_max'] + 237.3))
    # Daily-max VPD approximation (Tmax with daily-mean RH); documented in the data dictionary.
    out['vpd_kpa'] = es * (1 - out['rh2m'].clip(0, 100) / 100.0)
    return out


def fetch_power_daily(cell_id: str, lat: float, lon: float, start: date, end: date, rawdir: Path,
                      manifest: list, live: bool = False, pace_seconds: float = 0.5) -> pd.DataFrame:
    params = {'parameters': ','.join(POWER_PARAMS), 'community': 'AG', 'longitude': f'{lon:.4f}',
              'latitude': f'{lat:.4f}', 'start': f'{start:%Y%m%d}', 'end': f'{end:%Y%m%d}',
              'format': 'JSON', 'time-standard': 'UTC'}
    if live:
        stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
        fname, hit = f'{cell_id}_POWER_live_{start:%Y%m%d}_{end:%Y%m%d}_{stamp}.json', None
    else:
        fname = f'{cell_id}_POWER_{start:%Y%m%d}_{end:%Y%m%d}.json'
        hit = cached_entry(rawdir / 'weather' / fname)
    dst = rawdir / 'weather' / fname
    if hit:
        content = dst.read_bytes()
        manifest.append(hit)
    else:
        r = _get(POWER_API, params=params, timeout=(15, 90))
        content = r.content
        payload = json.loads(content)
        if 'properties' not in payload:
            raise SourceError(f'POWER error for {cell_id}: {str(payload)[:300]}')
        save_raw(rawdir / 'weather', fname, content, 'NASA POWER daily', r.url, manifest,
                 product='NASA POWER v2 daily point (UTC)', cell_id=cell_id, first_day=str(start),
                 last_day=str(end), power_sources=payload['header'].get('sources'),
                 units={k: v.get('units') for k, v in payload.get('parameters', {}).items()},
                 grid_point=payload.get('geometry', {}).get('coordinates'))
        time.sleep(pace_seconds)
    out = power_json_to_daily(json.loads(content), cell_id)
    out['power_sources'] = ','.join(json.loads(content)['header'].get('sources', []))
    return out


def worldcover_tile(lat: float, lon: float) -> str:
    la = int(np.floor(lat / 3.0) * 3)
    lo = int(np.floor(lon / 3.0) * 3)
    return f'{"N" if la >= 0 else "S"}{abs(la):02d}{"E" if lo >= 0 else "W"}{abs(lo):03d}'


def worldcover_fractions(cells: pd.DataFrame, step: float, rawdir: Path, manifest: list,
                         size: int = 600) -> pd.DataFrame:
    """Class fractions per cell from ESA WorldCover 2020, read through COG overviews (≈90 m effective).

    The per-cell sampled class counts are saved as the raw artefact (with hash) so results are auditable.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    dst = rawdir / 'landcover' / f'worldcover2020_counts_{len(cells)}cells_{cells.cell_key.iloc[0]}.json'
    hit = cached_entry(dst)
    if hit:
        manifest.append(hit)
        counts = json.loads(dst.read_text())
    else:
        counts, urls = {}, set()
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                   GDAL_HTTP_MAX_RETRY='4', GDAL_HTTP_RETRY_DELAY='3')
        with rasterio.Env(**env):
            for r in cells.itertuples():
                url = WORLDCOVER.format(tile=worldcover_tile(r.latitude, r.longitude))
                urls.add(url)
                with rasterio.open('/vsicurl/' + url) as src:
                    w = from_bounds(r.longitude - step/2, r.latitude - step/2, r.longitude + step/2,
                                    r.latitude + step/2, transform=src.transform)
                    a = src.read(1, window=w, out_shape=(size, size), resampling=Resampling.nearest)
                v, c = np.unique(a, return_counts=True)
                counts[r.cell_key] = {str(int(k)): int(n) for k, n in zip(v, c)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_raw(dst.parent, dst.name, json.dumps(counts, sort_keys=True).encode(), 'ESA WorldCover 2020 v100',
                 'https://esa-worldcover.org (CC BY 4.0); tiles: ' + ' '.join(sorted(urls)), manifest,
                 product='ESA WorldCover 10m 2020 v100, nearest-neighbour sample via COG overviews',
                 sample_grid=f'{size}x{size} per cell')
    rows = []
    for key, cnt in counts.items():
        tot = sum(cnt.values()) - cnt.get('0', 0)
        row = {'cell_key': key}
        for code, name in WORLDCOVER_CLASSES.items():
            row[f'lc_{name}'] = cnt.get(str(code), 0) / tot if tot else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


COP_DEM = ('https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{ns}{la:02d}_00_{ew}{lo:03d}_00_DEM/'
           'Copernicus_DSM_COG_30_{ns}{la:02d}_00_{ew}{lo:03d}_00_DEM.tif')


def dem_terrain(cells: pd.DataFrame, step: float, rawdir: Path, manifest: list, size: int = 300) -> pd.DataFrame:
    """Elevation mean / std / p5-p95 range per cell from Copernicus DEM GLO-90 COGs (AWS open data),
    read at ~180 m (average resampling of 90 m pixels). Per-cell statistics are stored as the hashed raw artefact."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    dst = rawdir / 'terrain' / f'copdem90_stats_{len(cells)}cells_{cells.cell_key.iloc[0]}.json'
    hit = cached_entry(dst)
    if hit:
        manifest.append(hit)
        stats = json.loads(dst.read_text())['stats']
    else:
        stats, urls = {}, set()
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                   GDAL_HTTP_MAX_RETRY='4', GDAL_HTTP_RETRY_DELAY='3')
        with rasterio.Env(**env):
            for r in cells.itertuples():
                w, s_, e, n = (r.longitude - step/2, r.latitude - step/2, r.longitude + step/2, r.latitude + step/2)
                la, lo = int(np.floor(s_)), int(np.floor(w))
                url = COP_DEM.format(ns='N' if la >= 0 else 'S', la=abs(la), ew='E' if lo >= 0 else 'W', lo=abs(lo))
                urls.add(url)
                try:
                    with rasterio.open('/vsicurl/' + url) as src:
                        b = src.bounds
                        win = from_bounds(max(w, b.left), max(s_, b.bottom), min(e, b.right), min(n, b.top),
                                          transform=src.transform).round_offsets().round_lengths()
                        a = src.read(1, window=win, out_shape=(size, size), resampling=Resampling.average).astype(float)
                except rasterio.errors.RasterioIOError as exc:
                    raise SourceError(f'Copernicus DEM tile unavailable for {r.cell_key}: {url}') from exc
                stats[r.cell_key] = {'elev_mean_m': float(a.mean()), 'elev_std_m': float(a.std()),
                                     'elev_range_m': float(np.percentile(a, 95) - np.percentile(a, 5))}
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_raw(dst.parent, dst.name, json.dumps({'stats': stats, 'tiles': sorted(urls)}, sort_keys=True).encode(),
                 'Copernicus DEM GLO-90 (AWS open data COG)', 'https://copernicus-dem-90m.s3.amazonaws.com', manifest,
                 product='Copernicus DEM GLO-90 DGED 2021/2022 release, average-resampled to ~180 m per cell',
                 licence='Copernicus DEM licence (free, attribution: produced using Copernicus WorldDEM-90)')
    return pd.DataFrame([{'cell_key': k, **v} for k, v in stats.items()])
