"""Australian crop-type classifier: Sentinel-2 winter-season composites, labelled by
high-confidence NLUM v7 commodity probabilities. Spatial-holdout accuracy; predicts
the training AOI and a separate unseen AOI; writes GeoTIFFs + maps + accuracy.

Usage:
  python scripts/crop_train_au.py --model hgb --indices --conf 0.7 --tag au_v1
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

from veg_species_mapper.cropmap import data, model, io_geo, nlum
from veg_species_mapper.cropmap.legend_au import CLASSES, NAME_BY_ID

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Riverina / SW Slopes NSW (mixed winter cropping).
AOI_TRAIN = (147.25, -34.50, 147.42, -34.35)
AOI_UNSEEN = (147.05, -34.50, 147.22, -34.35)
S2_YEAR = 2020
MONTHS = (5, 6, 7, 8, 9, 10, 11)   # Australian winter crop season


def build(bbox, use_indices):
    cube = data.s2_monthly_composite(bbox, year=S2_YEAR, months=MONTHS)
    if use_indices:
        cube = model.add_indices(cube, MONTHS)
    ylab, conf = nlum.labels_on_grid(bbox, like=cube)
    X, shape, bands = model.to_feature_matrix(cube)
    y = ylab.values.ravel().astype("uint8")
    c = conf.values.ravel().astype("float32")
    finite = model.valid_mask(X)
    return cube, ylab, X, y, c, finite, shape, bands


def make_model(name):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0,
                                      min_samples_leaf=2, class_weight="balanced_subsample")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb", choices=["rf", "hgb"])
    ap.add_argument("--indices", action="store_true")
    ap.add_argument("--conf", type=float, default=0.7, help="min NLUM confidence for labels")
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--block-px", type=int, default=40)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"au_{args.model}{'_idx' if args.indices else ''}"
    t0 = time.time()

    print(f"=== AU run '{tag}': model={args.model} indices={args.indices} conf>={args.conf} ===")
    print("[A] TRAIN AOI: building S2 winter composite + NLUM labels ...")
    cubeA, labA, XA, yA, cA, finA, shapeA, bands = build(AOI_TRAIN, args.indices)
    eligA = finA & (cA >= args.conf) & (yA != 0)
    print(f"    grid={shapeA} feats={len(bands)} finite={int(finA.sum())} "
          f"confident-labelled={int(eligA.sum())}")
    comp = {NAME_BY_ID[c]: int((yA[eligA] == c).sum()) for c in np.unique(yA[eligA])}
    print("    confident label composition:", comp)

    is_test = model.spatial_block_split(shapeA, block_px=args.block_px, test_frac=0.3)
    train_idx = model.sample_training(yA, is_test, eligA, n_per_class=args.n_per_class, drop_classes=(0,))
    print(f"    training samples={len(train_idx)} across {len(np.unique(yA[train_idx]))} classes")

    clf = make_model(args.model)
    clf.fit(XA[train_idx], yA[train_idx])

    metrics_spatial = model.evaluate(clf, XA, yA, is_test, eligA)
    rng = np.random.default_rng(1)
    naive_test = rng.random(len(yA)) < 0.3
    metrics_naive = model.evaluate(clf, XA, yA, naive_test, eligA)
    print(f"    spatial OA={metrics_spatial['overall_accuracy']:.3f} "
          f"kappa={metrics_spatial['kappa']:.3f} | naive OA={metrics_naive['overall_accuracy']:.3f}")

    predA = np.full(shapeA[0] * shapeA[1], 255, dtype="uint8")
    predA[finA] = clf.predict(XA[finA]); predA = predA.reshape(shapeA)
    proba = np.zeros(shapeA[0] * shapeA[1], dtype="float32")
    if hasattr(clf, "predict_proba"):
        proba[finA] = clf.predict_proba(XA[finA]).max(axis=1)
    proba = proba.reshape(shapeA)

    print("[B] UNSEEN AOI ...")
    cubeB, labB, XB, yB, cB, finB, shapeB, _ = build(AOI_UNSEEN, args.indices)
    eligB = finB & (cB >= args.conf) & (yB != 0)
    predB = np.full(shapeB[0] * shapeB[1], 255, dtype="uint8")
    predB[finB] = clf.predict(XB[finB]); predB = predB.reshape(shapeB)
    unseen_metrics = model.evaluate(clf, XB, yB, finB, eligB)
    print(f"    UNSEEN OA={unseen_metrics['overall_accuracy']:.3f} kappa={unseen_metrics['kappa']:.3f}")

    print("[C] Writing outputs ...")
    io_geo.write_class_geotiff(labA.values, cubeA, OUT / f"{tag}_train_labels.tif", classes=CLASSES)
    io_geo.write_class_geotiff(predA, cubeA, OUT / f"{tag}_train_prediction.tif", classes=CLASSES)
    io_geo.write_float_geotiff(proba, cubeA, OUT / f"{tag}_train_confidence.tif")
    io_geo.write_class_geotiff(labB.values, cubeB, OUT / f"{tag}_unseen_labels.tif", classes=CLASSES)
    io_geo.write_class_geotiff(predB, cubeB, OUT / f"{tag}_unseen_prediction.tif", classes=CLASSES)

    present = sorted({c for c in np.unique(predA).tolist() + np.unique(labA.values).tolist() if c != 255})
    io_geo.save_class_map_png(predA, present, f"AU predicted crops — TRAIN ({tag})", OUT / f"{tag}_train_prediction.png", classes=CLASSES)
    io_geo.save_class_map_png(labA.values, present, "NLUM labels — TRAIN AOI", OUT / f"{tag}_train_labels.png", classes=CLASSES)
    io_geo.save_class_map_png(predB, present, f"AU predicted crops — UNSEEN ({tag})", OUT / f"{tag}_unseen_prediction.png", classes=CLASSES)
    io_geo.save_class_map_png(labB.values, present, "NLUM labels — UNSEEN AOI", OUT / f"{tag}_unseen_labels.png", classes=CLASSES)
    io_geo.save_confusion_png(metrics_spatial["confusion_matrix"], metrics_spatial["labels"], OUT / f"{tag}_confusion.png", classes=CLASSES)
    io_geo.save_training_sources_map(train_idx, yA, shapeA, "NLUM v7 (2020-21)", OUT / f"{tag}_training_sources.png", classes=CLASSES)

    line = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag,
        "model": args.model, "indices": args.indices, "conf": args.conf,
        "n_features": len(bands), "n_train": len(train_idx),
        "spatial_OA": round(metrics_spatial["overall_accuracy"], 4),
        "spatial_kappa": round(metrics_spatial["kappa"], 4),
        "naive_OA": round(metrics_naive["overall_accuracy"], 4),
        "unseen_OA": round(unseen_metrics["overall_accuracy"], 4),
        "unseen_kappa": round(unseen_metrics["kappa"], 4),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / f"accuracy_{tag}.json").write_text(json.dumps(
        {**line, "label_composition": comp, "spatial_full": metrics_spatial,
         "unseen_full": unseen_metrics}, indent=2))
    md = OUT / "RESULTS_AU.md"
    if not md.exists():
        md.write_text("# Australian crop-type model iterations (NLUM-labelled)\n\n"
                      "spatial OA = held-out spatial blocks (honest); naive = random pixels; "
                      "unseen = separate AOI. Accuracy is agreement with NLUM, not field truth.\n\n"
                      "| time | tag | model | idx | conf | feats | n_train | spatial OA | kappa | naive OA | unseen OA | sec |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(md, "a") as f:
        f.write(f"| {line['time']} | {tag} | {args.model} | {args.indices} | {args.conf} | "
                f"{len(bands)} | {len(train_idx)} | **{line['spatial_OA']}** | {line['spatial_kappa']} | "
                f"{line['naive_OA']} | {line['unseen_OA']} | {line['seconds']} |\n")
    print("[done]", json.dumps(line))
    for k, v in metrics_spatial["report"].items():
        if k.isdigit():
            print(f"      {NAME_BY_ID.get(int(k), k):20s} F1={v['f1-score']:.2f} (n={int(v['support'])})")


if __name__ == "__main__":
    main()
