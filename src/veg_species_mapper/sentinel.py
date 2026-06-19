"""Fetch Sentinel-2 surface reflectance at a point from Microsoft Planetary Computer.

This is the 'pull the satellite signature for the labelled location' step. It's the
well-trodden part of the pipeline, kept separate because its dependencies
(pystac-client, planetary-computer, rasterio) are heavier and may lag on brand-new
Python versions. Install with:  pip install -r requirements-sentinel.txt

Planetary Computer's Sentinel-2 L2A collection is free and needs no account.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class S2Signature:
    datetime: str
    cloud_pct: float
    bands: dict[str, float]  # reflectance 0..1 by band name, e.g. {'B04': 0.08, ...}


def point_signature(
    lat: float,
    lon: float,
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    max_cloud: float = 20.0,
    bands: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12"),
) -> list[S2Signature]:
    """Return a Sentinel-2 reflectance time series at (lat, lon), least-cloudy first.

    Each signature is the single pixel containing the point for one scene -- enough
    to prove the geotagged label lines up with a usable satellite signal.
    """
    try:
        import planetary_computer as pc
        import rasterio
        from pystac_client import Client
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Sentinel deps missing. Run: pip install -r requirements-sentinel.txt"
        ) from e

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = sorted(search.items(), key=lambda it: it.properties["eo:cloud_cover"])

    out: list[S2Signature] = []
    for item in items:
        vals: dict[str, float] = {}
        for b in bands:
            asset = item.assets.get(b)
            if asset is None:
                continue
            with rasterio.open(asset.href) as ds:
                row, col = ds.index(*_reproject(lon, lat, ds.crs))
                arr = ds.read(1, window=((row, row + 1), (col, col + 1)))
                vals[b] = float(arr[0, 0]) / 10000.0  # L2A scale factor
        out.append(
            S2Signature(
                datetime=item.properties["datetime"],
                cloud_pct=float(item.properties["eo:cloud_cover"]),
                bands=vals,
            )
        )
    return out


def _reproject(lon: float, lat: float, dst_crs):
    from rasterio.warp import transform

    xs, ys = transform("EPSG:4326", dst_crs, [lon], [lat])
    return xs[0], ys[0]
