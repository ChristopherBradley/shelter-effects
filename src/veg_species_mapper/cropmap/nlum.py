"""Build crop-type training labels from NLUM v7 commodity probability surfaces.

For each 250 m cell we take the commodity with the highest probability (argmax) and
its probability as a confidence. High-confidence cells become training labels.
The result is reprojected (nearest) onto the Sentinel-2 grid.

Because NLUM is itself a modelled probability surface, accuracy against it measures
*agreement with NLUM*, not independent field truth. Independent validation would use
the NVT trial points (sensitive data) — out of scope for now.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from .legend_au import NLUM_KEY_TO_CLASS

NLUM_DIR = Path("data/public_data/NLUM_v7_250_AgProbabilitySurfaces_2020_21_geo_package_20241128")
NODATA = 32767
SCALE = 10000.0  # NLUM probabilities are 0..10000


def _find(key: str) -> str | None:
    hits = glob.glob(str(NLUM_DIR / f"*{key}*.tif"))
    return hits[0] if hits else None


def nlum_label_confidence(aoi_lonlat):
    """Return (label_da, conf_da) in EPSG:4283 @250 m over the AOI.
    label = reduced class id (0 where no ag), conf = max probability in 0..1."""
    keys = NLUM_KEY_TO_CLASS
    stack, class_of_layer = [], []
    ref = None
    for key, cls in keys:
        f = _find(key)
        if not f:
            continue
        r = rioxarray.open_rasterio(f).squeeze().rio.clip_box(*aoi_lonlat)
        if ref is None:
            ref = r
        a = r.values.astype("float32")
        a[a == NODATA] = -1.0
        stack.append(a)
        class_of_layer.append(cls)
    if not stack:
        raise RuntimeError("no NLUM layers found — check data/public_data path")
    S = np.stack(stack)                      # (layer, y, x)
    arg = S.argmax(axis=0)
    mx = S.max(axis=0)
    valid = mx > 0
    class_arr = np.zeros(arg.shape, dtype="uint8")
    cls_lut = np.array(class_of_layer, dtype="uint8")
    class_arr[valid] = cls_lut[arg[valid]]
    conf = np.where(valid, mx / SCALE, 0.0).astype("float32")

    label_da = xr.DataArray(class_arr, coords={"y": ref.y, "x": ref.x}, dims=("y", "x"))
    conf_da = xr.DataArray(conf, coords={"y": ref.y, "x": ref.x}, dims=("y", "x"))
    label_da = label_da.rio.write_crs(ref.rio.crs)
    conf_da = conf_da.rio.write_crs(ref.rio.crs)
    return label_da, conf_da


def labels_on_grid(aoi_lonlat, like: xr.DataArray):
    """Reproject NLUM label + confidence onto the S2 grid `like` (band,y,x)."""
    from rasterio.enums import Resampling
    label_da, conf_da = nlum_label_confidence(aoi_lonlat)
    like2d = like.isel(band=0) if "band" in like.dims else like
    lab = label_da.rio.reproject_match(like2d, resampling=Resampling.nearest)
    con = conf_da.rio.reproject_match(like2d, resampling=Resampling.nearest)
    y = xr.DataArray(lab.values.astype("uint8"),
                     coords={"y": like2d.y, "x": like2d.x}, dims=("y", "x")).rio.write_crs(like.rio.crs)
    c = xr.DataArray(con.values.astype("float32"),
                     coords={"y": like2d.y, "x": like2d.x}, dims=("y", "x")).rio.write_crs(like.rio.crs)
    return y, c
