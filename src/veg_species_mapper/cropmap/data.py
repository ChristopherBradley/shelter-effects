"""Load Sentinel-2 monthly composites + CDL labels onto a common grid.

The expensive part (downloading + compositing S2) is cached to a GeoTIFF per AOI/
year so model iterations are fast. Delete data/cache/*.tif to force a refresh.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from odc.stac import load as odc_load
from pystac_client import Client

from .legend import cdl_to_class

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Reflectance bands kept (skip ultra-blue/water-vapour). 10 m + 20 m, resampled to 10 m.
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# SCL values to drop (nodata/saturated/shadow/cloud/cirrus/snow).
SCL_DROP = {0, 1, 3, 8, 9, 10, 11}
CACHE = Path("data/cache")


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _client() -> Client:
    return Client.open(STAC, modifier=pc.sign_inplace)


def _aoi_key(bbox, year, months) -> str:
    raw = f"{bbox}-{year}-{months}".encode()
    return hashlib.md5(raw).hexdigest()[:10]


def s2_monthly_composite(
    bbox: tuple[float, float, float, float],
    year: int = 2021,
    months: tuple[int, ...] = (5, 6, 7, 8, 9),
    resolution: int = 10,
    max_cloud: int = 40,
    use_cache: bool = True,
) -> xr.DataArray:
    """Return a DataArray (band, y, x) where band = '<B>_m<MM>' monthly medians,
    cloud-masked via SCL. Cached to GeoTIFF."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _aoi_key(bbox, year, months)
    cache_tif = CACHE / f"s2_{key}.tif"
    if use_cache and cache_tif.exists():
        da = rioxarray.open_rasterio(cache_tif)
        names = da.attrs.get("long_name")
        if isinstance(names, (list, tuple)):
            da = da.assign_coords(band=list(names))
        return da

    lon_c = (bbox[0] + bbox[2]) / 2
    lat_c = (bbox[1] + bbox[3]) / 2
    epsg = utm_epsg(lon_c, lat_c)

    cat = _client()
    items = list(
        cat.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=f"{year}-{min(months):02d}-01/{year}-{max(months):02d}-30",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        ).items()
    )
    if not items:
        raise RuntimeError("no Sentinel-2 scenes for AOI/period")
    print(f"    loading {len(items)} S2 scenes -> EPSG:{epsg} @ {resolution} m ...")

    ds = odc_load(
        items,
        bands=S2_BANDS + ["SCL"],
        bbox=list(bbox),
        crs=f"EPSG:{epsg}",
        resolution=resolution,
        groupby="solar_day",
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        resampling="bilinear",
    )

    scl = ds["SCL"]
    valid = ~scl.isin(list(SCL_DROP))
    ds = ds[S2_BANDS].where(valid)

    month_idx = ds.time.dt.month
    comps = []
    band_names = []
    for m in months:
        sel = ds.where(month_idx == m, drop=False)
        med = sel.median(dim="time", skipna=True)
        for b in S2_BANDS:
            comps.append((med[b] / 10000.0).astype("float32"))
            band_names.append(f"{b}_m{m:02d}")

    cube = xr.concat(comps, dim="band").assign_coords(band=band_names)
    cube = cube.rio.write_crs(f"EPSG:{epsg}")
    print("    computing monthly medians (downloads COGs) ...")
    cube = cube.compute()
    # cache
    cube_to_save = cube.copy()
    cube_to_save.attrs["long_name"] = band_names
    cube_to_save.rio.to_raster(cache_tif)
    print(f"    cached -> {cache_tif}")
    return cube


def cdl_labels(
    bbox: tuple[float, float, float, float],
    like: xr.DataArray,
    year: int = 2021,
) -> xr.DataArray:
    """Load CDL cropland for the AOI, remap to reduced classes, aligned to `like` grid."""
    cat = _client()
    items = list(
        cat.search(
            collections=["usda-cdl"],
            bbox=list(bbox),
            datetime=f"{year}-01-01/{year}-12-31",
        ).items()
    )
    items = [it for it in items if "cropland" in it.assets and f"cropland_{year}" in it.id]
    if not items:
        raise RuntimeError(f"no CDL cropland for {year}")
    from rasterio.enums import Resampling
    href = items[0].assets["cropland"].href  # signed by client modifier
    raw = rioxarray.open_rasterio(href).squeeze()
    # reproject CDL (EPSG:5070, 30 m) onto the S2 grid (nearest = preserve classes)
    like2d = like.isel(band=0) if "band" in like.dims else like
    cdl = raw.rio.reproject_match(like2d, resampling=Resampling.nearest)
    classes = cdl_to_class(cdl.values.astype("int32"))
    out = xr.DataArray(classes, coords={"y": like2d.y, "x": like2d.x}, dims=("y", "x"))
    out = out.rio.write_crs(like.rio.crs)
    return out
