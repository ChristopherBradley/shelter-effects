"""Mapillary API v4 client: find street-level images near a point and download them.

Mapillary imagery is openly licensed (CC-BY-SA) and the API returns camera GPS
*and* compass pose, which is what makes downstream triangulation possible.

Get a free token at https://www.mapillary.com/dashboard/developers
(create an app -> client token, looks like 'MLY|<digits>|<hex>').
Set it as MAPILLARY_TOKEN in your environment / .env file.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

GRAPH = "https://graph.mapillary.com"

# Fields worth pulling. computed_* are Mapillary's pose-refined estimates and are
# preferred over the raw geometry/compass_angle when present.
IMAGE_FIELDS = (
    "id,captured_at,camera_type,compass_angle,computed_compass_angle,"
    "geometry,computed_geometry,thumb_2048_url,thumb_original_url,sequence"
)


@dataclass
class MapillaryImage:
    id: str
    lon: float
    lat: float
    compass_angle: float  # degrees clockwise from north
    camera_type: str      # 'spherical'/'equirectangular' == true 360
    captured_at: int
    thumb_url: str
    sequence: str | None = None

    @property
    def is_panoramic(self) -> bool:
        return self.camera_type in ("spherical", "equirectangular")

    @classmethod
    def from_feature(cls, f: dict) -> "MapillaryImage":
        geom = f.get("computed_geometry") or f.get("geometry")
        lon, lat = geom["coordinates"]
        compass = f.get("computed_compass_angle")
        if compass is None:
            compass = f.get("compass_angle", 0.0)
        return cls(
            id=str(f["id"]),
            lon=lon,
            lat=lat,
            compass_angle=float(compass),
            camera_type=f.get("camera_type", "perspective"),
            captured_at=int(f.get("captured_at", 0)),
            thumb_url=f.get("thumb_2048_url") or f.get("thumb_original_url", ""),
            sequence=f.get("sequence"),
        )


def _token() -> str:
    tok = os.environ.get("MAPILLARY_TOKEN")
    if not tok:
        raise RuntimeError(
            "MAPILLARY_TOKEN not set. Get one at "
            "https://www.mapillary.com/dashboard/developers and put it in .env"
        )
    return tok


def bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a square ~radius_m around a point."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search_images(
    lat: float,
    lon: float,
    radius_m: float = 30.0,
    limit: int = 50,
    panoramic_only: bool = True,
) -> list[MapillaryImage]:
    """Find images within radius_m of (lat, lon), nearest first."""
    bbox = bbox_around(lat, lon, radius_m)
    params = {
        "fields": IMAGE_FIELDS,
        "bbox": ",".join(str(c) for c in bbox),
        "limit": limit,
        "access_token": _token(),
    }
    resp = requests.get(f"{GRAPH}/images", params=params, timeout=30)
    resp.raise_for_status()
    feats = resp.json().get("data", [])
    imgs = [MapillaryImage.from_feature(f) for f in feats if (f.get("computed_geometry") or f.get("geometry"))]
    if panoramic_only:
        imgs = [i for i in imgs if i.is_panoramic]
    imgs.sort(key=lambda i: haversine_m(lat, lon, i.lat, i.lon))
    return imgs


def download(img: MapillaryImage, dest_dir: str | Path) -> Path:
    """Download an image's panorama JPEG to dest_dir, return the path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{img.id}.jpg"
    if out.exists():
        return out
    if not img.thumb_url:
        raise RuntimeError(f"image {img.id} has no thumb url")
    r = requests.get(img.thumb_url, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out
