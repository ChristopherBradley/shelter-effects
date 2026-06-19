"""Tree-species mapping for the ACT: ACT Tree Assets (+ NLUM pine plantation) -> labels,
Sentinel-2 full-year composite -> features, classifier -> evergreen-native / conifer /
deciduous-exotic / casuarina map. For attributing species to shelterbelts.

  # build + preview labels only (no Sentinel-2 needed):
  python scripts/tree_train.py --labels-only
  # full model (needs Planetary Computer up):
  python scripts/tree_train.py --model hgb

Full-year monthly composites are used (not just winter) so deciduous leaf-off phenology
separates exotic trees from evergreen eucalypts/conifers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veg_species_mapper import treespecies as ts
from veg_species_mapper.cropmap import io_geo
from veg_species_mapper.cropmap.data import utm_epsg

OUT = ROOT / "outputs"
DATA = ROOT / "data"

# AOI: Stromlo pine plantation + Weston Creek suburbs (pine + eucalypt + exotic mix)
AOI = (148.98, -35.36, 149.12, -35.28)
YEAR = 2021
MONTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
NLUM_PLANTATION = DATA / "public_data" / "NLUM_v7_250_AgProbabilitySurfaces_2020_21_geo_package_20241128" / "NLUM_v7_probSurf_2021_340_24_PLANTATION_FR.tif"


def make_grid(bbox, res=10):
    """Standalone 10 m grid over bbox in local UTM. Returns an xarray template + meta."""
    from rasterio.transform import from_origin
    from pyproj import Transformer
    minlon, minlat, maxlon, maxlat = bbox
    epsg = utm_epsg((minlon + maxlon) / 2, (minlat + maxlat) / 2)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([minlon, maxlon, minlon, maxlon], [minlat, minlat, maxlat, maxlat])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    nx = int(np.ceil((maxx - minx) / res)); ny = int(np.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    x = minx + (np.arange(nx) + 0.5) * res
    y = maxy - (np.arange(ny) + 0.5) * res
    da = xr.DataArray(np.zeros((ny, nx), "uint8"), coords={"y": y, "x": x}, dims=("y", "x"))
    da = da.rio.write_crs(f"EPSG:{epsg}")
    return da, transform, f"EPSG:{epsg}", (ny, nx)


def add_plantation_pine(label, template):
    """Overlay NLUM plantation-forestry (confident) as Conifer/Pine (class 2)."""
    if not NLUM_PLANTATION.exists():
        print("    (NLUM plantation layer not found; skipping pine overlay)")
        return label
    import rioxarray
    from rasterio.enums import Resampling
    raw = rioxarray.open_rasterio(NLUM_PLANTATION).squeeze()
    raw = raw.where(raw != 32767)
    plant = raw.rio.reproject_match(template, resampling=Resampling.bilinear)
    conf = plant.values / 10000.0
    label[conf > 0.5] = 2
    return label


def build_labels(template, transform, crs, shape):
    print("[1] Fetching ACT tree points (cached) ...")
    df = ts.fetch_act_trees(AOI, DATA / "act_trees_aoi.csv")
    print(f"    {len(df)} trees; class counts: "
          + str({ts.NAME_BY_ID[c]: int((df.cls == c).sum()) for c in sorted(df.cls.unique())}))
    label, total = ts.rasterize_tree_labels(df, transform, crs, shape, min_trees=3)
    print(f"    tree-labelled pixels (>=3 trees): {int((label>0).sum())}")
    label = add_plantation_pine(label, template)
    comp = {ts.NAME_BY_ID[c]: int((label == c).sum()) for c in sorted(np.unique(label)) if c}
    print(f"    final label composition: {comp}")
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--model", default="hgb", choices=["rf", "hgb"])
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--tag", default="tree_act")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    template, transform, crs, shape = make_grid(AOI)
    print(f"Grid {shape} @10 m, {crs}")
    label = build_labels(template, transform, crs, shape)

    io_geo.write_class_geotiff(label, template, OUT / f"{args.tag}_labels.tif", classes=ts.CLASSES)
    present = sorted({c for c in np.unique(label).tolist()})
    io_geo.save_class_map_png(label, present, "ACT tree-species labels (inventory + NLUM pine)",
                              OUT / f"{args.tag}_labels.png", classes=ts.CLASSES)
    print(f"Wrote {args.tag}_labels.tif / .png")

    if args.labels_only:
        print("labels-only: done."); return

    # ---- full model (needs Planetary Computer) ----
    print("[2] Building Sentinel-2 full-year composite (needs PC) ...")
    from veg_species_mapper.cropmap import data as s2data, model
    cube = s2data.s2_monthly_composite(AOI, year=YEAR, months=MONTHS)
    cube = model.add_indices(cube, MONTHS)
    # align labels to S2 grid
    lab_da = template.copy(data=label)
    from rasterio.enums import Resampling
    lab_on_s2 = lab_da.rio.reproject_match(cube.isel(band=0), resampling=Resampling.nearest)
    X, sh, bands = model.to_feature_matrix(cube)
    X, finite = model.finite_and_fill(X)
    y = lab_on_s2.values.ravel().astype("uint8")
    y[y > 4] = 0  # reproject fill/nodata -> treat as non-tree (excluded from training)
    elig = finite & (y != 0)
    is_test = model.spatial_block_split(sh, block_px=40, test_frac=0.3)
    tr_idx = model.sample_training(y, is_test, elig, n_per_class=args.n_per_class, drop_classes=(0,))
    print(f"    training {len(tr_idx)} samples, classes {sorted(np.unique(y[tr_idx]).tolist())}")

    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0,
                                     class_weight="balanced_subsample", min_samples_leaf=2)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
    clf.fit(X[tr_idx], y[tr_idx])
    metrics = model.evaluate(clf, X, y, is_test, elig)
    print(f"    spatial OA={metrics['overall_accuracy']:.3f} kappa={metrics['kappa']:.3f}")

    pred = np.full(sh[0]*sh[1], 255, "uint8")
    pred[finite] = clf.predict(X[finite]); pred = pred.reshape(sh)
    io_geo.write_class_geotiff(pred, cube.isel(band=0), OUT / f"{args.tag}_prediction.tif", classes=ts.CLASSES)
    io_geo.save_class_map_png(pred, present, f"ACT tree species predicted ({args.tag})",
                              OUT / f"{args.tag}_prediction.png", classes=ts.CLASSES)
    io_geo.save_confusion_png(metrics["confusion_matrix"], metrics["labels"],
                              OUT / f"{args.tag}_confusion.png", classes=ts.CLASSES)
    from veg_species_mapper.cropmap import pipeline
    pipeline.save_model(OUT / "models" / f"{args.tag}.joblib", clf, bands, ts.CLASSES,
                        {"year": YEAR, "months": list(MONTHS), "use_indices": True, "tag": args.tag})
    print(f"Wrote {args.tag}_prediction.tif + saved model")


if __name__ == "__main__":
    main()
