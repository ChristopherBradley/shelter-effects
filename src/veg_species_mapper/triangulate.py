"""Triangulate a ground location from two camera positions + bearings to the target.

Given two panoramas that both 'see' the same plant, each gives an absolute bearing
(compass_angle of the image + yaw of the crop that detected the species). The two
bearing rays intersect at the plant's ground location.

Uses a local east-north tangent plane (flat-earth), which is accurate to well under
a metre over the tens-of-metres baselines involved here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Bearing:
    lat: float
    lon: float
    bearing_deg: float  # absolute, clockwise from north


@dataclass
class Fix:
    lat: float
    lon: float
    # distances from each camera to the fix (m); huge/negative => rays diverge
    range_a_m: float
    range_b_m: float
    ok: bool


def _to_local(lat, lon, lat0, lon0) -> tuple[float, float]:
    e = math.radians(lon - lon0) * 6_371_000.0 * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * 6_371_000.0
    return e, n


def _to_geo(e, n, lat0, lon0) -> tuple[float, float]:
    lat = lat0 + math.degrees(n / 6_371_000.0)
    lon = lon0 + math.degrees(e / (6_371_000.0 * math.cos(math.radians(lat0))))
    return lat, lon


def triangulate(a: Bearing, b: Bearing) -> Fix:
    """Intersect two bearing rays. Returns a Fix (ok=False if rays are near-parallel
    or the intersection falls behind a camera)."""
    lat0, lon0 = a.lat, a.lon
    ax, ay = _to_local(a.lat, a.lon, lat0, lon0)
    bx, by = _to_local(b.lat, b.lon, lat0, lon0)

    # Direction unit vectors (east, north) from clockwise-from-north bearing.
    da = (math.sin(math.radians(a.bearing_deg)), math.cos(math.radians(a.bearing_deg)))
    db = (math.sin(math.radians(b.bearing_deg)), math.cos(math.radians(b.bearing_deg)))

    # Solve  [ax]+ta*da = [bx]+tb*db
    denom = da[0] * (-db[1]) - da[1] * (-db[0])
    if abs(denom) < 1e-9:
        return Fix(0, 0, math.inf, math.inf, ok=False)
    rx, ry = bx - ax, by - ay
    ta = (rx * (-db[1]) - ry * (-db[0])) / denom
    tb = (da[0] * ry - da[1] * rx) / denom

    ix, iy = ax + ta * da[0], ay + ta * da[1]
    lat, lon = _to_geo(ix, iy, lat0, lon0)
    ok = ta > 0 and tb > 0  # intersection in front of both cameras
    return Fix(lat, lon, ta, tb, ok=ok)
