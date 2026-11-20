#!/usr/bin/env python3
"""
feasibility_probe.py  (COSC 3 fire / PM2.5 feasibility study, Phase 0)

Retrieves a SMALL, GENUINE sample from
  1. OpenAQ public archive on AWS S3 (anonymous HTTPS): one csv.gz per location-day
  2. NASA FIRMS area API (needs a free MAP_KEY in the env var FIRMS_MAP_KEY)
  3. ERA5 via the Open-Meteo archive API (no key; non-commercial use only)
then runs schema, timestamp and join diagnostics. Writes raw files, a manifest
with SHA-256 hashes, join_sample.csv and probe_report.json to --out.

--selftest checks the geometry and time logic on the OpenAQ documentation's
example rows plus explicitly SYNTHETIC geometry cases. It never touches the
network and never writes synthetic values into network-mode outputs.

Examples
  python feasibility_probe.py --selftest
  export FIRMS_MAP_KEY=your_key
  python feasibility_probe.py --location 2178 --tz America/Denver \
      --lat 35.1353 --lon -106.584702 --start 2022-05-01 --days 10 \
      --bbox=-110,31,-102,38

Requires python>=3.9, requests, pandas, numpy.
"""
import argparse, gzip, hashlib, io, json, os, sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0088
OPENAQ_BASE = "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz"
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ERA5_HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
               "wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m",
               "boundary_layer_height"]
FIRE_LATENCY_H = 3      # NRT fire data assumed usable 3 h after overpass
PM_MARGIN_H = 1         # PM hours used as predictors must be <= tau - 1 h
MIN_VALID_HOURS = 18    # valid daily mean needs >= 18 of 24 hours
DECAY_KM = 300.0


# ---------------- geometry and time helpers (covered by --selftest) ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 (station) to point 2 (fire), degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def upwind_alignment(bearing_to_fire, wind_from_deg):
    """+1 when the fire lies exactly where the wind blows FROM (upwind), -1 when exactly downwind."""
    return np.cos(np.radians(np.asarray(bearing_to_fire, dtype=float) - wind_from_deg))


def vector_mean_wind_from(speed, from_deg):
    th = np.radians(np.asarray(from_deg, dtype=float))
    s = np.asarray(speed, dtype=float)
    u, v = -s * np.sin(th), -s * np.cos(th)          # direction the air moves toward
    ub, vb = np.nanmean(u), np.nanmean(v)
    return float((np.degrees(np.arctan2(-ub, -vb)) + 360.0) % 360.0), float(np.hypot(ub, vb))


def standard_utc_offset(tzname, year=2022):
    """Standard-time offset = smaller of the January and July offsets (DST always adds time)."""
    tz = ZoneInfo(tzname)
    return min(datetime(year, m, 15, 12, tzinfo=tz).utcoffset() for m in (1, 7))


def fire_available(det_time_utc, tau_utc, latency_h=FIRE_LATENCY_H):
    return det_time_utc + pd.Timedelta(hours=latency_h) <= tau_utc


def valid_day(n_hours):
    return n_hours >= MIN_VALID_HOURS


def stuck_mask(values, run=6):
    s = pd.Series(values, dtype=float).reset_index(drop=True)
    grp = (s != s.shift()).cumsum()
    lens = s.groupby(grp).transform("size")
    return ((lens >= run) & (s != 0) & s.notna()).to_numpy()


def normalize_openaq(df):
    """The OpenAQ docs use both sensor_id/sensors_id and unit/units; accept either."""
    df = df.rename(columns={"sensor_id": "sensors_id", "unit": "units"})
    need = {"location_id", "sensors_id", "datetime", "lat", "lon", "parameter", "units", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"OpenAQ columns missing: {sorted(missing)}; got {list(df.columns)}")
    out = df.copy()
    out["utc_offset_str"] = out["datetime"].astype(str).str[-6:]
    out["time_utc"] = pd.to_datetime(out["datetime"].astype(str), utc=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def firms_time_utc(df):
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    return pd.to_datetime(df["acq_date"].astype(str) + " " + t.str[:2] + ":" + t.str[2:], utc=True)


# ---------------- network helpers ----------------
def sha256(b):
    return hashlib.sha256(b).hexdigest()


def http_get(url, params=None, timeout=90):
    import requests
    return requests.get(url, params=params, timeout=timeout,
                        headers={"User-Agent": "cosc3-feasibility-probe/0.2 (academic, non-commercial)"})


def save(path, content, manifest, source, url, status, redact=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    shown = url.replace(redact, "<MAP_KEY>") if redact else url
    manifest.append({"source": source, "url": shown, "status": status, "bytes": len(content),
                     "sha256": sha256(content), "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                     "path": path})


def fetch_openaq(loc, d0, d1, out, manifest):
    frames, missing = [], []
    d = d0
    while d <= d1:
        got = False
        for prefix in ("location", "loc"):           # both prefixes appear in the docs
            url = f"{OPENAQ_BASE}/locationid={loc}/year={d:%Y}/month={d:%m}/{prefix}-{loc}-{d:%Y%m%d}.csv.gz"
            try:
                r = http_get(url)
            except Exception as e:                    # noqa: BLE001
                manifest.append({"source": "openaq", "url": url, "status": f"error: {type(e).__name__}"})
                continue
            if r.status_code == 200:
                save(os.path.join(out, "raw", "openaq", os.path.basename(url)), r.content,
                     manifest, "openaq", url, 200)
                frames.append(pd.read_csv(io.BytesIO(gzip.decompress(r.content))))
                got = True
                break
        if not got:
            missing.append(str(d))
        d += timedelta(days=1)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing


def fetch_firms(key, sources, bbox, d0, ndays, out, manifest):
    frames, notes = [], []
    try:
        r = http_get(f"{FIRMS_API}/data_availability/csv/{key}/all")
        save(os.path.join(out, "raw", "firms", "data_availability.csv"), r.content, manifest,
             "firms", r.url, r.status_code, redact=key)
    except Exception as e:                            # noqa: BLE001
        notes.append(f"data_availability failed: {e}")
    for src in sources:
        k = 0
        while k < ndays:
            n = min(10, ndays - k)                    # DAY_RANGE is 1..10
            d = d0 + timedelta(days=k)
            url = f"{FIRMS_API}/area/csv/{key}/{src}/{bbox}/{n}/{d:%Y-%m-%d}"
            try:
                r = http_get(url, timeout=180)
            except Exception as e:                    # noqa: BLE001
                notes.append(f"{src} {d}: request error {type(e).__name__}")
                k += n
                continue
            fname = f"{src}_{d:%Y%m%d}_{n}d.csv"
            save(os.path.join(out, "raw", "firms", fname), r.content, manifest, "firms", url,
                 r.status_code, redact=key)
            head = r.text[:200]
            if r.status_code == 200 and head.lower().startswith("latitude"):
                df = pd.read_csv(io.StringIO(r.text))
                df["source_product"] = src
                frames.append(df)
            else:
                notes.append(f"{src} {d}: HTTP {r.status_code}: {head.strip()[:120]}")
            k += n
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), notes


def fetch_era5(lat, lon, d0, d1, out, manifest):
    hourly = list(ERA5_HOURLY)
    for attempt in range(2):
        params = {"latitude": lat, "longitude": lon, "start_date": f"{d0}", "end_date": f"{d1}",
                  "hourly": ",".join(hourly), "models": "era5", "timezone": "GMT",
                  "wind_speed_unit": "ms", "cell_selection": "nearest"}
        r = http_get(OM_ARCHIVE, params=params)
        save(os.path.join(out, "raw", "era5", f"era5_attempt{attempt}.json"), r.content, manifest,
             "open-meteo-era5", r.url, r.status_code)
        js = r.json()
        if r.status_code == 200 and "hourly" in js:
            return js, "boundary_layer_height" in hourly
        if attempt == 0 and "boundary_layer_height" in hourly:
            hourly.remove("boundary_layer_height")    # retry without BLH and record that it failed
            continue
        raise RuntimeError(f"Open-Meteo error {r.status_code}: {js}")
    raise RuntimeError("unreachable")


# ---------------- diagnostics ----------------
def daily_lst(pm, std_off):
    pm = pm.copy()
    pm["time_lst"] = (pm["time_utc"] + std_off).dt.tz_localize(None)
    pm["day_lst"] = pm["time_lst"].dt.floor("D")
    g = pm.groupby("day_lst")["value"]
    return pd.DataFrame({"mean": g.mean(), "n_hours": g.count()})


def run_network(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    manifest, report = [], {"started_utc": datetime.now(timezone.utc).isoformat(), "args": vars(a)}
    d_start = date.fromisoformat(a.start)
    d0, d1 = d_start - timedelta(days=3), d_start + timedelta(days=a.days + 1)
    std_off = standard_utc_offset(a.tz, d_start.year)

    # 1. OpenAQ
    raw, missing_days = fetch_openaq(a.location, d0, d1, out, manifest)
    rep = {"days_requested": (d1 - d0).days + 1, "days_missing_file": missing_days}
    if raw.empty:
        rep["status"] = "FAIL: no OpenAQ files retrieved"
        pm = pd.DataFrame()
    else:
        aq = normalize_openaq(raw)
        rep.update({"rows": len(aq), "columns": list(raw.columns),
                    "parameters": aq["parameter"].value_counts().to_dict(),
                    "utc_offsets_seen": aq["utc_offset_str"].value_counts().to_dict(),
                    "coords_unique": aq[["lat", "lon"]].drop_duplicates().values.tolist()})
        pm = aq[aq["parameter"].str.lower().isin(["pm25", "pm2.5"])].copy()
        if pm.empty:
            rep["status"] = "WARN: location has no PM2.5 sensor in this window; choose another location"
        else:
            pm = pm.sort_values("time_utc")
            dup = int(pm.duplicated(["sensors_id", "time_utc"]).sum())
            pm["stuck"] = stuck_mask(pm["value"].to_numpy())
            v = pm["value"]
            rep.update({"pm25_units": pm["units"].value_counts().to_dict(),
                        "pm25_sensors": pm["sensors_id"].nunique(), "pm25_rows": len(pm),
                        "pm25_duplicates_sensor_time": dup, "pm25_negative": int((v < 0).sum()),
                        "pm25_max": float(v.max()), "pm25_stuck_hours": int(pm["stuck"].sum()),
                        "pm25_placeholder_like": int(v.isin([-999, 999, 985, 1000]).sum())})
            if pm["sensors_id"].nunique() > 1:
                keep = pm["sensors_id"].value_counts().idxmax()
                rep["note_multi_sensor"] = f"kept sensor {keep} (most rows)"
                pm = pm[pm["sensors_id"] == keep]
            pm = pm[(pm["value"] >= -5) & (pm["value"] <= 1500) & (~pm["stuck"])].copy()
            pm["value"] = pm["value"].clip(lower=0)
            dl = daily_lst(pm, std_off)
            rep["daily_lst"] = {str(k.date()): {"mean": round(r["mean"], 2), "n_hours": int(r["n_hours"])}
                                for k, r in dl.iterrows()}
            rep["status"] = "PASS"
    report["openaq"] = rep

    # 2. ERA5
    lat, lon = a.lat, a.lon
    if (lat is None or lon is None) and not pm.empty:
        lat, lon = float(pm["lat"].iloc[0]), float(pm["lon"].iloc[0])
    met = pd.DataFrame()
    try:
        js, blh_ok = fetch_era5(lat, lon, d0, d1, out, manifest)
        met = pd.DataFrame(js["hourly"])
        met["time_utc"] = pd.to_datetime(met["time"], utc=True)
        report["era5"] = {"status": "PASS", "requested": [lat, lon],
                          "grid_cell": [js.get("latitude"), js.get("longitude")],
                          "cell_offset_km": float(haversine_km(lat, lon, js["latitude"], js["longitude"])),
                          "utc_offset_seconds": js.get("utc_offset_seconds"),
                          "hours": len(met), "units": js.get("hourly_units"),
                          "missing_by_var": met.isna().sum().to_dict(),
                          "boundary_layer_height_available": blh_ok}
    except Exception as e:                            # noqa: BLE001
        report["era5"] = {"status": f"FAIL: {e}"}

    # 3. FIRMS
    key = os.environ.get("FIRMS_MAP_KEY")
    fires = pd.DataFrame()
    if not key:
        report["firms"] = {"status": "SKIPPED: set FIRMS_MAP_KEY (free at firms.modaps.eosdis.nasa.gov/api/map_key/)"}
    else:
        fires, notes = fetch_firms(key, a.firms_sources.split(","), a.bbox, d0, (d1 - d0).days + 1,
                                   out, manifest)
        rep = {"notes": notes}
        if fires.empty:
            rep["status"] = "FAIL or empty: no detections returned (check notes, bbox, source names)"
        else:
            fires["time_utc"] = firms_time_utc(fires)
            fires["dist_km"] = haversine_km(lat, lon, fires["latitude"], fires["longitude"])
            rep.update({"status": "PASS", "rows": len(fires), "columns": list(fires.columns),
                        "has_type_field": "type" in fires.columns,
                        "confidence": fires["confidence"].astype(str).value_counts().to_dict(),
                        "versions": fires["version"].astype(str).value_counts().to_dict(),
                        "satellites": fires["satellite"].astype(str).value_counts().to_dict(),
                        "frp_quantiles_MW": fires["frp"].quantile([.5, .9, .99]).round(2).to_dict(),
                        "utc_hour_hist": fires["time_utc"].dt.hour.value_counts().sort_index().to_dict(),
                        "within_300km": int((fires["dist_km"] <= 300).sum()),
                        "within_1000km": int((fires["dist_km"] <= 1000).sum())})
        report["firms"] = rep

    # 4. Join: one row per issue day, tau = 00:00 local standard time
    rows = []
    if not pm.empty:
        dl = daily_lst(pm, std_off)
        for k in range(a.days):
            d = pd.Timestamp(d_start + timedelta(days=k))
            tau = (d - std_off).tz_localize("UTC")
            prev, tgt = d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)
            row = {"issue_day_lst": str(d.date()), "tau_utc": str(tau)}
            hist = pm[pm["time_utc"] <= tau - pd.Timedelta(hours=PM_MARGIN_H)]
            row["max_pm_time_used_utc"] = str(hist["time_utc"].max()) if len(hist) else None
            for name, day in (("pm_d1", prev), ("y1_target_mean", tgt)):
                if day in dl.index:
                    row[name] = round(float(dl.loc[day, "mean"]), 2)
                    row[name + "_n_hours"] = int(dl.loc[day, "n_hours"])
                    row[name + "_valid"] = valid_day(int(dl.loc[day, "n_hours"]))
            if not met.empty:
                w = met[(met["time_utc"] >= tau - pd.Timedelta(hours=24)) & (met["time_utc"] < tau)]
                row["met_hours_in_window"] = len(w)
                wd, ws = vector_mean_wind_from(w["wind_speed_100m"], w["wind_direction_100m"])
                row["wind100_from_deg"], row["wind100_ms"] = round(wd, 1), round(ws, 2)
            if not fires.empty:
                ok = fires[fire_available(fires["time_utc"], tau)
                           & (fires["time_utc"] >= tau - pd.Timedelta(hours=24))
                           & (fires["dist_km"] <= 1000)]
                ok = ok[ok["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])]
                row["fires_used_n"] = len(ok)
                row["latest_fire_used_utc"] = str(ok["time_utc"].max()) if len(ok) else None
                wexp = np.exp(-ok["dist_km"] / DECAY_KM)
                row["log1p_frp_exp300"] = round(float(np.log1p((ok["frp"] * wexp).sum())), 3)
                if not met.empty:
                    al = upwind_alignment(bearing_deg(lat, lon, ok["latitude"], ok["longitude"]), wd)
                    row["log1p_frp_up300"] = round(float(np.log1p((ok["frp"] * wexp * np.clip(al, 0, None) ** 2).sum())), 3)
                    row["log1p_frp_down300"] = round(float(np.log1p((ok["frp"] * wexp * np.clip(-al, 0, None) ** 2).sum())), 3)
                if len(ok):
                    assert ok["time_utc"].max() + pd.Timedelta(hours=FIRE_LATENCY_H) <= tau, "fire leakage"
            if len(hist):
                assert hist["time_utc"].max() <= tau - pd.Timedelta(hours=PM_MARGIN_H), "PM leakage"
            rows.append(row)
    js = pd.DataFrame(rows)
    js.to_csv(os.path.join(out, "join_sample.csv"), index=False)
    report["join"] = {"rows": len(js), "leakage_assertions": "passed" if len(js) else "not run",
                      "note": "tau = local standard midnight; PM history <= tau-1h; fires need +3h latency"}
    report["manifest"] = manifest
    with open(os.path.join(out, "probe_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else None)
                      for k, v in report.items() if k in ("openaq", "era5", "firms")}, indent=2))
    print(f"wrote {out}/probe_report.json and {out}/join_sample.csv ({len(js)} rows)")


def selftest():
    res = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond), str(detail)))

    # Rows copied verbatim from the example file on docs.openaq.org/aws/about (documentation, not a download)
    doc = ("location_id,sensors_id,location,datetime,lat,lon,parameter,units,value\n"
           "2178,3919,Del Norte-2178,2023-02-22T01:00:00-07:00,35.1353,-106.584702,pm10,µg/m³,14.0\n"
           "2178,3919,Del Norte-2178,2023-02-22T02:00:00-07:00,35.1353,-106.584702,pm10,µg/m³,4.0\n"
           "2178,3919,Del Norte-2178,2023-02-22T03:00:00-07:00,35.1353,-106.584702,pm10,µg/m³,7.0\n")
    df = normalize_openaq(pd.read_csv(io.StringIO(doc)))
    check("OpenAQ doc rows: -07:00 local converts to 08:00 UTC", str(df["time_utc"].iloc[0]) == "2023-02-22 08:00:00+00:00",
          df["time_utc"].iloc[0])
    alt = doc.replace("sensors_id", "sensor_id").replace(",units,", ",unit,")
    check("OpenAQ: accepts sensor_id/unit header variant", len(normalize_openaq(pd.read_csv(io.StringIO(alt)))) == 3)
    # SYNTHETIC geometry and timing cases below
    check("haversine: 1 degree of latitude is about 111.19 km", abs(haversine_km(0, 0, 1, 0) - 111.19) < 0.05,
          round(float(haversine_km(0, 0, 1, 0)), 3))
    check("bearing to a fire due north is 0", abs(float(bearing_deg(0, 0, 1, 0))) < 1e-9)
    check("bearing to a fire due east is 90", abs(float(bearing_deg(0, 0, 0, 1)) - 90) < 1e-9)
    check("fire north, wind from north: alignment +1 (upwind)", abs(float(upwind_alignment(0, 0)) - 1) < 1e-12)
    check("fire north, wind from south: alignment -1 (downwind)", abs(float(upwind_alignment(0, 180)) + 1) < 1e-12)
    wd, _ = vector_mean_wind_from([5, 5], [350, 10])
    check("vector mean of winds from 350 and 10 degrees is 0 degrees", min(wd, 360 - wd) < 1e-9, wd)
    tau = pd.Timestamp("2022-05-03 07:00", tz="UTC")
    check("detection 2 h before issue is NOT usable (3 h latency)", not fire_available(tau - pd.Timedelta(hours=2), tau))
    check("detection 4 h before issue IS usable", fire_available(tau - pd.Timedelta(hours=4), tau))
    check("America/Denver standard offset is -7 h", standard_utc_offset("America/Denver") == timedelta(hours=-7))
    check("Australia/Sydney standard offset is +10 h", standard_utc_offset("Australia/Sydney") == timedelta(hours=10))
    check("Asia/Karachi standard offset is +5 h", standard_utc_offset("Asia/Karachi") == timedelta(hours=5))
    check("17 valid hours is not a valid day; 18 is", (not valid_day(17)) and valid_day(18))
    check("stuck run of 6 identical values is flagged", int(stuck_mask([10, 12, 12, 12, 12, 12, 12, 9]).sum()) == 6)
    tsyn = pd.DataFrame({"acq_date": ["2022-05-02"], "acq_time": [905]})
    check("FIRMS acq_time 905 parses as 09:05 UTC", str(firms_time_utc(tsyn).iloc[0]) == "2022-05-02 09:05:00+00:00")
    w = max(len(n) for n, _, _ in res)
    for n, ok, d in res:
        print(f"{'PASS' if ok else 'FAIL'}  {n.ljust(w)}  {d}")
    print(f"\n{sum(ok for _, ok, _ in res)}/{len(res)} checks passed (offline; no network data used)")
    return all(ok for _, ok, _ in res)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--location", type=int, default=2178, help="OpenAQ location id")
    p.add_argument("--tz", default="America/Denver", help="IANA time zone of the station")
    p.add_argument("--lat", type=float, default=35.1353, help="station latitude (default: OpenAQ 2178)")
    p.add_argument("--lon", type=float, default=-106.584702, help="station longitude")
    p.add_argument("--start", default="2022-05-01", help="first issue day (local), YYYY-MM-DD")
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--bbox", default="-110,31,-102,38", help="west,south,east,north for FIRMS")
    p.add_argument("--firms-sources", default="VIIRS_SNPP_SP,VIIRS_NOAA20_SP")
    p.add_argument("--out", default="probe_out")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run_network(a)


if __name__ == "__main__":
    main()
