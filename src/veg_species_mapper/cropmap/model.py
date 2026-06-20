"""Feature engineering, spatially-aware sampling/splitting, training, evaluation."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .data import S2_BANDS


def add_indices(cube: xr.DataArray, months) -> xr.DataArray:
    """Append per-month NDVI, NDWI and a red-edge NDVI (NDRE) to the band stack."""
    bn = list(cube.band.values)
    extra, names = [], []

    def g(b, m):
        return cube.sel(band=f"{b}_m{m:02d}")

    for m in months:
        if f"B08_m{m:02d}" in bn and f"B04_m{m:02d}" in bn:
            nir, red = g("B08", m), g("B04", m)
            extra.append((nir - red) / (nir + red + 1e-6)); names.append(f"NDVI_m{m:02d}")
        if f"B08_m{m:02d}" in bn and f"B11_m{m:02d}" in bn:
            nir, swir = g("B08", m), g("B11", m)
            extra.append((nir - swir) / (nir + swir + 1e-6)); names.append(f"NDWI_m{m:02d}")
        if f"B08_m{m:02d}" in bn and f"B05_m{m:02d}" in bn:
            nir, re = g("B08", m), g("B05", m)
            extra.append((nir - re) / (nir + re + 1e-6)); names.append(f"NDRE_m{m:02d}")

    if not extra:
        return cube
    add = xr.concat(extra, dim="band").assign_coords(band=names)
    return xr.concat([cube, add], dim="band")


def flowering_index(cube: xr.DataArray, months, kind: str = "ndyi") -> xr.DataArray:
    """Append a per-month canola-flowering index. Year-specific (tracks rotation).
      ndyi = (green-blue)/(green+blue)                  -- simple yellowness
      cfi  = NDVI * (red + 2*green - blue)              -- yellowness x greenness
             (per paddock-ts; suppresses bare-soil yellow that NDYI flags)
    """
    bn = list(cube.band.values)
    extra, names = [], []

    def g(b, m):
        return cube.sel(band=f"{b}_m{m:02d}")

    for m in months:
        need = [f"B0{x}_m{m:02d}" for x in (2, 3, 4)] + [f"B08_m{m:02d}"]
        if not all(b in bn for b in need):
            continue
        blu, grn, red, nir = g("B02", m), g("B03", m), g("B04", m), g("B08", m)
        if kind == "cfi":
            ndvi = (nir - red) / (nir + red + 1e-6)
            extra.append(ndvi * (red + 2 * grn - blu)); names.append(f"CFI_m{m:02d}")
        else:
            extra.append((grn - blu) / (grn + blu + 1e-6)); names.append(f"NDYI_m{m:02d}")

    if not extra:
        return cube
    add = xr.concat(extra, dim="band").assign_coords(band=names)
    out = xr.concat([cube, add], dim="band")
    return out


def to_feature_matrix(cube: xr.DataArray):
    """(band,y,x) -> X (n_pix, n_band), and the (y,x) shape. NaNs -> 0 (masked later)."""
    arr = cube.transpose("band", "y", "x").values
    nb, ny, nx = arr.shape
    X = arr.reshape(nb, ny * nx).T
    return X, (ny, nx), list(cube.band.values)


def valid_mask(X: np.ndarray) -> np.ndarray:
    """Pixels with finite features in all bands (strict)."""
    return np.isfinite(X).all(axis=1)


def majority_filter(labels: np.ndarray, size: int = 5, nodata: int = 255) -> np.ndarray:
    """Modal (majority) smoothing of a categorical label raster — cleans within-field
    salt-and-pepper from per-pixel classification. Vectorised: per class, count
    neighbours in a size×size window, take the per-pixel argmax count."""
    from scipy.ndimage import uniform_filter
    present = [c for c in np.unique(labels) if c != nodata]
    best_cnt = np.full(labels.shape, -1.0, dtype="float32")
    best_cls = np.full(labels.shape, nodata, dtype="uint8")
    for c in present:
        cnt = uniform_filter((labels == c).astype("float32"), size=size, mode="nearest")
        upd = cnt > best_cnt
        best_cls[upd] = c
        best_cnt[upd] = cnt[upd]
    best_cls[labels == nodata] = nodata
    return best_cls


def finite_and_fill(X: np.ndarray):
    """Robust alternative to valid_mask for national scale: keep any pixel with at
    least one finite feature (e.g. a cloudy month leaves some bands NaN), and fill the
    remaining NaN/inf with 0 in place. Tree models tolerate the 0-fill, and the same
    fill is applied at train and predict time, so it stays consistent.
    Returns (X_filled, valid_mask)."""
    finite = np.isfinite(X).any(axis=1)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, finite


def spatial_block_split(shape, block_px: int = 50, test_frac: float = 0.3, seed: int = 0):
    """Assign each pixel to a square block, then whole blocks to train/test.
    Returns a flat boolean array `is_test` over y*x. Avoids the optimistic bias of
    random pixel splits (neighbouring pixels are highly correlated)."""
    ny, nx = shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    block_id = (yy // block_px) * (nx // block_px + 1) + (xx // block_px)
    block_id = block_id.ravel()
    rng = np.random.default_rng(seed)
    uniq = np.unique(block_id)
    test_blocks = set(rng.choice(uniq, size=int(len(uniq) * test_frac), replace=False).tolist())
    return np.isin(block_id, list(test_blocks))


def sample_training(y, is_test, valid, n_per_class=3000, seed=0, drop_classes=(0,)):
    """Indices of training pixels: from train blocks & valid pixels, balanced per class."""
    rng = np.random.default_rng(seed)
    elig = valid & (~is_test)
    idx_all = []
    for cls in np.unique(y[elig]):
        if cls in drop_classes:
            continue
        cls_idx = np.where(elig & (y == cls))[0]
        if len(cls_idx) == 0:
            continue
        take = min(n_per_class, len(cls_idx))
        idx_all.append(rng.choice(cls_idx, size=take, replace=False))
    return np.concatenate(idx_all) if idx_all else np.array([], dtype=int)


def evaluate(model, X, y, is_test, valid, drop_classes=(0,)):
    """Accuracy metrics on spatially held-out test pixels."""
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 classification_report, confusion_matrix)
    mask = is_test & valid & (~np.isin(y, list(drop_classes)))
    Xte, yte = X[mask], y[mask]
    pred = model.predict(Xte)
    labels = sorted(np.unique(yte).tolist())
    return {
        "n_test": int(mask.sum()),
        "overall_accuracy": float(accuracy_score(yte, pred)),
        "kappa": float(cohen_kappa_score(yte, pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
        "report": classification_report(yte, pred, labels=labels,
                                        output_dict=True, zero_division=0),
    }
