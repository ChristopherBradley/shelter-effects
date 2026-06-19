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
    out = xr.concat([cube, add], dim="band")
    return out


def to_feature_matrix(cube: xr.DataArray):
    """(band,y,x) -> X (n_pix, n_band), and the (y,x) shape. NaNs -> 0 (masked later)."""
    arr = cube.transpose("band", "y", "x").values
    nb, ny, nx = arr.shape
    X = arr.reshape(nb, ny * nx).T
    return X, (ny, nx), list(cube.band.values)


def valid_mask(X: np.ndarray) -> np.ndarray:
    """Pixels with finite features in all bands."""
    return np.isfinite(X).all(axis=1)


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
