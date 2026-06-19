"""Shared feature-building, model persistence, and apply-to-AOI prediction.

Centralised so training and later prediction use identical features. A saved model
bundle carries everything needed to predict a brand-new AOI (year/months/indices/
band order/legend), so the model can be run anywhere in Australia without retraining.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data, model, io_geo


def build_features(bbox, year, months, use_indices):
    """Return (cube, X, bands, finite_mask) for an AOI."""
    cube = data.s2_monthly_composite(bbox, year=year, months=tuple(months))
    if use_indices:
        cube = model.add_indices(cube, tuple(months))
    X, shape, bands = model.to_feature_matrix(cube)
    X, finite = model.finite_and_fill(X)
    return cube, X, bands, finite, shape


def save_model(path, clf, bands, classes, meta):
    import joblib
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "bands": list(bands), "classes": classes, "meta": meta}, path)
    return path


def load_model(path):
    import joblib
    return joblib.load(path)


def _align_columns(X, bands, want_bands):
    """Reorder/select feature columns to match the model's training band order."""
    if list(bands) == list(want_bands):
        return X
    idx = {b: i for i, b in enumerate(bands)}
    missing = [b for b in want_bands if b not in idx]
    if missing:
        raise ValueError(f"AOI features missing bands the model needs: {missing[:5]} ...")
    cols = [idx[b] for b in want_bands]
    return X[:, cols]


def predict_aoi(bundle, bbox, out_prefix, write_confidence=True):
    """Apply a saved model bundle to an AOI; write class + confidence GeoTIFFs and a PNG.
    Returns (prediction_2d, confidence_2d, cube)."""
    meta = bundle["meta"]
    cube, X, bands, finite, shape = build_features(
        bbox, meta["year"], meta["months"], meta["use_indices"])
    Xa = _align_columns(X, bands, bundle["bands"])
    clf, classes = bundle["clf"], bundle["classes"]

    pred = np.full(shape[0] * shape[1], 255, dtype="uint8")
    pred[finite] = clf.predict(Xa[finite]); pred = pred.reshape(shape)
    proba = np.zeros(shape[0] * shape[1], dtype="float32")
    if write_confidence and hasattr(clf, "predict_proba"):
        proba[finite] = clf.predict_proba(Xa[finite]).max(axis=1)
    proba = proba.reshape(shape)

    out_prefix = Path(out_prefix)
    io_geo.write_class_geotiff(pred, cube, f"{out_prefix}_prediction.tif", classes=classes)
    if write_confidence:
        io_geo.write_float_geotiff(proba, cube, f"{out_prefix}_confidence.tif")
    present = sorted({v for v in np.unique(pred).tolist() if v != 255})
    io_geo.save_class_map_png(pred, present, f"Predicted crops — {out_prefix.name}",
                              f"{out_prefix}_prediction.png", classes=classes)
    return pred, proba, cube
