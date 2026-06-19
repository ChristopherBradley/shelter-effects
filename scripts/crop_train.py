"""Train a crop-type classifier from Sentinel-2 monthly composites (labels: USDA CDL),
evaluate with spatial holdout, predict on the training AOI AND an unseen AOI, and
write GeoTIFFs + maps + accuracy.

Usage:
  python scripts/crop_train.py --model rf
  python scripts/crop_train.py --model hgb --indices --tag run2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veg_species_mapper.cropmap import data, model, io_geo
from veg_species_mapper.cropmap.legend import NAME_BY_ID

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Train AOI and a spatially separate "unseen" AOI (central North Dakota).
AOI_TRAIN = (-99.28, 47.57, -99.16, 47.67)
AOI_UNSEEN = (-99.14, 47.57, -99.02, 47.67)
YEAR = 2021
MONTHS = (5, 6, 7, 8, 9)


def build(bbox, use_indices):
    cube = data.s2_monthly_composite(bbox, year=YEAR, months=MONTHS)
    if use_indices:
        cube = model.add_indices(cube, MONTHS)
    labels = data.cdl_labels(bbox, like=cube, year=YEAR)
    X, shape, bands = model.to_feature_matrix(cube)
    y = labels.values.ravel().astype("uint8")
    valid = model.valid_mask(X)
    return cube, labels, X, y, valid, shape, bands


def make_model(name):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0,
                                      min_samples_leaf=2, class_weight="balanced_subsample")
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                              max_leaf_nodes=63, l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(name)


def log_results(tag, model_name, n_features, n_train, metrics_spatial, metrics_naive,
                unseen_metrics, secs):
    line = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": tag, "model": model_name, "n_features": n_features, "n_train": n_train,
        "spatial_OA": round(metrics_spatial["overall_accuracy"], 4),
        "spatial_kappa": round(metrics_spatial["kappa"], 4),
        "naive_OA": round(metrics_naive["overall_accuracy"], 4),
        "unseen_OA": round(unseen_metrics["overall_accuracy"], 4),
        "unseen_kappa": round(unseen_metrics["kappa"], 4),
        "seconds": round(secs, 1),
    }
    (OUT / f"accuracy_{tag}.json").write_text(json.dumps(
        {**line, "spatial_full": metrics_spatial, "unseen_full": unseen_metrics}, indent=2))
    md = OUT / "RESULTS.md"
    if not md.exists():
        md.write_text("# Crop-type model iterations\n\n"
                      "OA = overall accuracy. **spatial** = held-out spatial blocks (honest); "
                      "naive = random-pixel (optimistic); unseen = separate AOI-B.\n\n"
                      "| time (UTC) | tag | model | feats | n_train | spatial OA | kappa | naive OA | unseen OA | unseen kappa | sec |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(md, "a") as f:
        f.write(f"| {line['time']} | {tag} | {model_name} | {n_features} | {n_train} | "
                f"**{line['spatial_OA']}** | {line['spatial_kappa']} | {line['naive_OA']} | "
                f"{line['unseen_OA']} | {line['unseen_kappa']} | {line['seconds']} |\n")
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf", choices=["rf", "hgb"])
    ap.add_argument("--indices", action="store_true", help="add NDVI/NDWI/NDRE features")
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--block-px", type=int, default=50)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.model}{'_idx' if args.indices else ''}"
    t0 = time.time()

    print(f"=== Run '{tag}': model={args.model} indices={args.indices} ===")
    print("[A] Building TRAIN AOI features + labels ...")
    cubeA, labelsA, XA, yA, validA, shapeA, bands = build(AOI_TRAIN, args.indices)
    print(f"    grid={shapeA} features={len(bands)} valid_px={int(validA.sum())}")

    # spatial split + balanced sampling
    is_test = model.spatial_block_split(shapeA, block_px=args.block_px, test_frac=0.3)
    train_idx = model.sample_training(yA, is_test, validA, n_per_class=args.n_per_class)
    print(f"    train samples={len(train_idx)} (balanced, {len(np.unique(yA[train_idx]))} classes)")

    clf = make_model(args.model)
    clf.fit(XA[train_idx], yA[train_idx])

    # honest (spatial) + optimistic (naive random) metrics
    metrics_spatial = model.evaluate(clf, XA, yA, is_test, validA)
    rng = np.random.default_rng(1)
    naive_test = rng.random(len(yA)) < 0.3
    metrics_naive = model.evaluate(clf, XA, yA, naive_test, validA)
    print(f"    spatial OA={metrics_spatial['overall_accuracy']:.3f} "
          f"kappa={metrics_spatial['kappa']:.3f} | naive OA={metrics_naive['overall_accuracy']:.3f}")

    # predict TRAIN AOI
    predA = np.full(shapeA[0] * shapeA[1], 255, dtype="uint8")
    predA[validA] = clf.predict(XA[validA])
    predA = predA.reshape(shapeA)
    proba = np.zeros(shapeA[0] * shapeA[1], dtype="float32")
    if hasattr(clf, "predict_proba"):
        proba[validA] = clf.predict_proba(XA[validA]).max(axis=1)
    proba = proba.reshape(shapeA)

    # === UNSEEN AOI ===
    print("[B] Building UNSEEN AOI features + labels ...")
    cubeB, labelsB, XB, yB, validB, shapeB, _ = build(AOI_UNSEEN, args.indices)
    predB = np.full(shapeB[0] * shapeB[1], 255, dtype="uint8")
    predB[validB] = clf.predict(XB[validB])
    predB = predB.reshape(shapeB)
    all_testB = validB  # whole unseen AOI is out-of-sample
    unseen_metrics = model.evaluate(clf, XB, yB, all_testB, validB)
    print(f"    UNSEEN OA={unseen_metrics['overall_accuracy']:.3f} "
          f"kappa={unseen_metrics['kappa']:.3f}")

    # === outputs ===
    print("[C] Writing GeoTIFFs, maps, metrics ...")
    io_geo.write_class_geotiff(labelsA.values, cubeA, OUT / f"train_labels_{tag}.tif")
    io_geo.write_class_geotiff(predA, cubeA, OUT / f"train_prediction_{tag}.tif")
    io_geo.write_float_geotiff(proba, cubeA, OUT / f"train_confidence_{tag}.tif")
    io_geo.write_class_geotiff(labelsB.values, cubeB, OUT / f"unseen_labels_{tag}.tif")
    io_geo.write_class_geotiff(predB, cubeB, OUT / f"unseen_prediction_{tag}.tif")

    present = sorted(set(np.unique(predA).tolist()) | set(np.unique(labelsA.values).tolist()))
    present = [c for c in present if c != 255]
    io_geo.save_class_map_png(predA, present, f"Predicted crops — TRAIN AOI ({tag})",
                              OUT / f"train_prediction_{tag}.png")
    io_geo.save_class_map_png(labelsA.values, present, "CDL reference — TRAIN AOI",
                              OUT / f"train_labels_{tag}.png")
    io_geo.save_class_map_png(predB, present, f"Predicted crops — UNSEEN AOI ({tag})",
                              OUT / f"unseen_prediction_{tag}.png")
    io_geo.save_class_map_png(labelsB.values, present, "CDL reference — UNSEEN AOI",
                              OUT / f"unseen_labels_{tag}.png")
    io_geo.save_confusion_png(metrics_spatial["confusion_matrix"], metrics_spatial["labels"],
                              OUT / f"confusion_{tag}.png")
    io_geo.save_training_sources_map(cubeA, train_idx, yA, shapeA, "USDA CDL 2021",
                                     OUT / f"training_sources_{tag}.png")

    line = log_results(tag, args.model, len(bands), len(train_idx),
                       metrics_spatial, metrics_naive, unseen_metrics, time.time() - t0)
    print("[done]", json.dumps(line))
    # concise per-class F1 (spatial)
    rep = metrics_spatial["report"]
    print("    per-class F1 (spatial holdout):")
    for k, v in rep.items():
        if k.isdigit():
            print(f"      {NAME_BY_ID.get(int(k), k):16s} F1={v['f1-score']:.2f} (n={int(v['support'])})")


if __name__ == "__main__":
    main()
