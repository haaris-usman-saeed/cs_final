"""Land fraction and true cell area for latitude/longitude grid cells."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def cell_area_km2(south: float, north: float, west: float, east: float) -> float:
    """Exact spherical area of a lat/lon rectangle (cells shrink toward the poles)."""
    return (EARTH_RADIUS_KM ** 2 * math.radians(east - west)
            * (math.sin(math.radians(north)) - math.sin(math.radians(south))))


def global_cell_id(lat_center: float, lon_center: float, step: float) -> str:
    """Grid-independent identifier, stable across different study bounding boxes."""
    iy = int(math.floor((lat_center + 90) / step))
    ix = int(math.floor((lon_center + 180) / step))
    return f'g{step:g}_y{iy}_x{ix}'


def _load_union(paths: list[Path]):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    geoms = []
    for p in paths:
        for f in json.loads(Path(p).read_text())['features']:
            geoms.append(shape(f['geometry']))
    return unary_union(geoms) if geoms else None


def land_fraction(cells: pd.DataFrame, step: float, land_geojson: Path,
                  lakes_geojson: Path | None = None, samples: int = 20) -> pd.Series:
    """Fraction of an evenly spaced (area-weighted) point lattice that is land and not a lake."""
    from shapely import contains_xy, box
    from shapely.prepared import prep  # noqa: F401  (kept for older shapely)
    land = _load_union([land_geojson])
    lakes = _load_union([lakes_geojson]) if lakes_geojson else None
    out = []
    offs = (np.arange(samples) + 0.5) / samples - 0.5
    for row in cells.itertuples():
        bb = box(row.longitude - step / 2, row.latitude - step / 2,
                 row.longitude + step / 2, row.latitude + step / 2)
        lnd = land.intersection(bb)
        lk = lakes.intersection(bb) if lakes is not None else None
        xs, ys = np.meshgrid(row.longitude + offs * step, row.latitude + offs * step)
        w = np.cos(np.radians(ys))          # area weight across the lattice
        on_land = contains_xy(lnd, xs, ys) if not lnd.is_empty else np.zeros_like(xs, bool)
        if lk is not None and not lk.is_empty:
            on_land &= ~contains_xy(lk, xs, ys)
        out.append(float((w * on_land).sum() / w.sum()))
    return pd.Series(out, index=cells.index, name='land_fraction')
