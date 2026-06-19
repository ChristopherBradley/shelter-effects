"""Proof-of-concept weed species-distribution model from ALA presences + Sentinel-2.

Presence (ALA occurrences) vs background (random pixels) classifier -> probability-of-
presence raster. Demonstrates the ALA->satellite route for weed mapping.

  python scripts/ala_weed_sdm.py --q "genus:Rubus" --name blackberry

Honest by design: presence-only data with sampling bias, so this finds "where the
established weed signal looks like the recorded sites", evaluated by spatial CV (AUC).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veg_species_mapper import ala
from veg_species_mapper.cropmap import data as s2data, model, io_geo

OUT = ROOT / "outputs"
DATA = ROOT / "data"
AOI = (148.98, -35.40, 149.12, -35.25)
YEAR = 2021
MONTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


def presence_mask(df, cube):
    """Boolean (ny,nx) mask of pixels containing >=1 ALA occurrence."""
    from pyproj import Transformer
    like = cube.isel(band=0)
    crs = like.rio.crs; transform = like.rio.transform()
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform(df.lon.values, df.lat.values)
    inv = ~transform
    cols, rows = inv * (np.array(xs), np.array(ys))
    cols = np.floor(cols).astype(int); rows = np.floor(rows).astype(int)
    ny, nx = like.shape
    m = np.zeros((ny, nx), bool)
    ok = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
    m[rows[ok], cols[ok]] = True
    return m


def save_prob_png(prob, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(prob, cmap="magma", vmin=0, vmax=1)
    ax.set_title(title); ax.axis("off")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="probability of presence")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="genus:Rubus", help="ALA biocache query")
    ap.add_argument("--name", default="blackberry")
    ap.add_argument("--bg", type=int, default=6000, help="background sample count")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print(f"[1] ALA presences for '{args.q}' ...")
    df = ala.fetch_occurrences(args.q, AOI, DATA / f"ala_{args.name}.csv")
    print(f"    {len(df)} occurrence points")

    print("[2] Sentinel-2 full-year composite ...")
    cube = s2data.s2_monthly_composite(AOI, year=YEAR, months=MONTHS)
    cube = model.add_indices(cube, MONTHS)
    X, shape, bands = model.to_feature_matrix(cube)
    X, finite = model.finite_and_fill(X)

    pres = presence_mask(df, cube).ravel()
    pres_idx = np.where(pres & finite)[0]
    print(f"    presence pixels (with valid S2): {len(pres_idx)}")
    if len(pres_idx) < 30:
        print("    too few presence pixels; aborting."); return

    rng = np.random.default_rng(0)
    bg_pool = np.where(finite & ~pres)[0]
    bg_idx = rng.choice(bg_pool, size=min(args.bg, len(bg_pool)), replace=False)

    # spatial-block CV (avoid presence/background autocorrelation leakage)
    is_test = model.spatial_block_split(shape, block_px=40, test_frac=0.3)
    idx = np.concatenate([pres_idx, bg_idx])
    y = np.concatenate([np.ones(len(pres_idx)), np.zeros(len(bg_idx))]).astype(int)
    test = is_test.ravel()[idx]
    Xtr, ytr = X[idx[~test]], y[~test]
    Xte, yte = X[idx[test]], y[test]

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te) if len(np.unique(yte)) > 1 else float("nan")
    ap_score = average_precision_score(yte, p_te)
    print(f"[3] spatial-CV AUC={auc:.3f}  avg-precision={ap_score:.3f} "
          f"(test n={len(yte)}, {int(yte.sum())} presence)")

    print("[4] predicting probability surface ...")
    prob = np.zeros(shape[0] * shape[1], "float32")
    prob[finite] = clf.predict_proba(X[finite])[:, 1]
    prob = prob.reshape(shape)
    io_geo.write_float_geotiff(prob, cube.isel(band=0), OUT / f"ala_{args.name}_probability.tif")
    save_prob_png(prob, OUT / f"ala_{args.name}_probability.png",
                  f"{args.name}: modelled probability of presence (ALA+S2)")
    (OUT / f"ala_{args.name}_metrics.json").write_text(json.dumps(
        {"q": args.q, "n_presence_points": int(len(df)), "n_presence_px": int(len(pres_idx)),
         "n_background": int(len(bg_idx)), "spatial_cv_auc": round(float(auc), 4),
         "avg_precision": round(float(ap_score), 4), "n_features": len(bands)}, indent=2))
    print(f"Wrote ala_{args.name}_probability.tif (AUC {auc:.3f})")


if __name__ == "__main__":
    main()
