"""Multi-region Australian crop-type classifier (NLUM-labelled Sentinel-2).

Trains on confident NLUM pixels pooled from several cropping regions (to cover all
target classes: cereals / canola / legumes / hay / pasture), evaluates on spatially
held-out blocks AND a completely held-out region, and writes per-region GeoTIFFs +
maps + a per-region training-distribution map.

Memory-light two-pass design: pass 1 samples each region then frees the big arrays;
pass 2 re-loads each region from the S2 cache (fast) to predict and write rasters.

Usage:
  python scripts/crop_train_au_multi.py --model hgb --indices --conf 0.7 --tag au_multi
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veg_species_mapper.cropmap import data, model, io_geo, nlum, pipeline
from veg_species_mapper.cropmap.legend_au import CLASSES, NAME_BY_ID

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# SE-Australia training regions chosen for class coverage (from NLUM probe):
#   Temora    -> winter cereals, canola, grazing
#   Wimmera   -> winter cereals, legumes, grazing
#   Corowa    -> winter cereals, canola, hay, grazing
TRAIN_REGIONS = {
    "Temora":  (147.25, -34.50, 147.42, -34.35),
    "Wimmera": (142.30, -36.55, 142.47, -36.40),
    "Corowa":  (146.75, -35.95, 146.92, -35.80),
}
UNSEEN_REGIONS = {"Ardlethan_E": (147.45, -34.50, 147.62, -34.35)}

# National preset: spread across WA / SA / Vic / NSW for cross-continent generalization.
NATIONAL_TRAIN = {
    "Temora_NSW":      (147.25, -34.50, 147.42, -34.35),
    "Wimmera_VIC":     (142.30, -36.55, 142.47, -36.40),
    "Corowa_NSW":      (146.75, -35.95, 146.92, -35.80),
    "Merredin_WA":     (118.20, -31.55, 118.37, -31.40),
    "Clare_SA":        (138.55, -33.85, 138.72, -33.70),
    "Cootamundra_NSW": (148.00, -34.70, 148.17, -34.55),
    "Mallee_VIC":      (142.40, -35.30, 142.57, -35.15),
}
NATIONAL_UNSEEN = {
    "Ardlethan_NSW": (147.45, -34.50, 147.62, -34.35),
    "Kojonup_WA":    (117.10, -33.85, 117.27, -33.70),
}
S2_YEAR = 2020
MONTHS = (5, 6, 7, 8, 9, 10, 11)


def build(bbox, use_indices):
    cube = data.s2_monthly_composite(bbox, year=S2_YEAR, months=MONTHS)
    if use_indices:
        cube = model.add_indices(cube, MONTHS)
    ylab, conf = nlum.labels_on_grid(bbox, like=cube)
    X, shape, bands = model.to_feature_matrix(cube)
    y = ylab.values.ravel().astype("uint8")
    c = conf.values.ravel().astype("float32")
    X, finite = model.finite_and_fill(X)
    return cube, ylab, X, y, c, finite, shape, bands


def make_model(name):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0,
                                      min_samples_leaf=2, class_weight="balanced_subsample")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.07,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb", choices=["rf", "hgb"])
    ap.add_argument("--indices", action="store_true")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--n-per-class", type=int, default=4000)
    ap.add_argument("--block-px", type=int, default=40)
    ap.add_argument("--national", action="store_true", help="use WA/SA/Vic/NSW preset")
    ap.add_argument("--tag", default="au_multi")
    args = ap.parse_args()
    tag = args.tag
    train_regions = NATIONAL_TRAIN if args.national else TRAIN_REGIONS
    unseen_regions = NATIONAL_UNSEEN if args.national else UNSEEN_REGIONS
    t0 = time.time()
    print(f"=== AU multi-region '{tag}': model={args.model} indices={args.indices} "
          f"conf>={args.conf} national={args.national} ===")

    # ---- pass 1: sample each training region ----
    Xtr, ytr = [], []
    test_pool_X, test_pool_y = [], []
    sources_plot = {}   # region -> (rows, cols, classes, shape)
    comp_total = {}
    feat_bands = None
    for name, bbox in train_regions.items():
        print(f"[1] region {name}: building ...")
        cube, ylab, X, y, c, finite, shape, bands = build(bbox, args.indices)
        feat_bands = bands
        elig = finite & (c >= args.conf) & (y != 0)
        is_test = model.spatial_block_split(shape, block_px=args.block_px, test_frac=0.3)
        tr_idx = model.sample_training(y, is_test, elig, n_per_class=args.n_per_class, drop_classes=(0,))
        Xtr.append(X[tr_idx]); ytr.append(y[tr_idx])
        # pooled spatial-holdout test samples (cap to keep memory modest)
        te_idx = np.where(elig & is_test)[0]
        if len(te_idx) > 60000:
            te_idx = np.random.default_rng(0).choice(te_idx, 60000, replace=False)
        test_pool_X.append(X[te_idx]); test_pool_y.append(y[te_idx])
        rows, cols = np.unravel_index(tr_idx, shape)
        sources_plot[name] = (rows, cols, y[tr_idx], shape)
        for cls in np.unique(y[tr_idx]):
            comp_total[NAME_BY_ID[cls]] = comp_total.get(NAME_BY_ID[cls], 0) + int((y[tr_idx] == cls).sum())
        print(f"    {name}: train={len(tr_idx)} classes={sorted(np.unique(y[tr_idx]).tolist())}")
        del cube, X, y, c, finite, ylab; gc.collect()

    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    print(f"[2] training {args.model} on {len(Xtr)} pooled samples, composition={comp_total}")
    clf = make_model(args.model)
    clf.fit(Xtr, ytr)

    # persist the model so it can be applied anywhere later without retraining
    meta = {"year": S2_YEAR, "months": list(MONTHS), "use_indices": args.indices,
            "conf": args.conf, "train_regions": list(train_regions),
            "model": args.model, "tag": tag}
    mpath = pipeline.save_model(OUT / "models" / f"{tag}.joblib", clf, feat_bands, CLASSES, meta)
    print(f"    saved model -> {mpath}")

    # pooled spatial-holdout metrics
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
    Xte = np.concatenate(test_pool_X); yte = np.concatenate(test_pool_y)
    pte = clf.predict(Xte)
    labels = sorted(np.unique(yte).tolist())
    metrics_spatial = {
        "n_test": int(len(yte)),
        "overall_accuracy": float(accuracy_score(yte, pte)),
        "kappa": float(cohen_kappa_score(yte, pte)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(yte, pte, labels=labels).tolist(),
        "report": classification_report(yte, pte, labels=labels, output_dict=True, zero_division=0),
    }
    print(f"    POOLED spatial OA={metrics_spatial['overall_accuracy']:.3f} kappa={metrics_spatial['kappa']:.3f}")
    del Xtr, ytr, Xte, yte, test_pool_X, test_pool_y; gc.collect()

    # ---- pass 2: predict each region (re-load from cache) + write rasters ----
    def predict_and_write(name, bbox, is_unseen=False):
        cube, ylab, X, y, c, finite, shape, _ = build(bbox, args.indices)
        if int(finite.sum()) == 0:
            print(f"    [skip] {name}: no valid Sentinel-2 pixels (cloud/coverage gap)")
            del cube, X, y, c, finite, ylab; gc.collect()
            return None
        pred = np.full(shape[0] * shape[1], 255, dtype="uint8")
        pred[finite] = clf.predict(X[finite]); pred = pred.reshape(shape)
        proba = np.zeros(shape[0] * shape[1], dtype="float32")
        if hasattr(clf, "predict_proba"):
            proba[finite] = clf.predict_proba(X[finite]).max(axis=1)
        proba = proba.reshape(shape)
        io_geo.write_class_geotiff(ylab.values, cube, OUT / f"{tag}_{name}_labels.tif", classes=CLASSES)
        io_geo.write_class_geotiff(pred, cube, OUT / f"{tag}_{name}_prediction.tif", classes=CLASSES)
        io_geo.write_float_geotiff(proba, cube, OUT / f"{tag}_{name}_confidence.tif")
        present = sorted({v for v in np.unique(pred).tolist() + np.unique(ylab.values).tolist() if v != 255})
        io_geo.save_class_map_png(pred, present, f"Predicted — {name} ({'UNSEEN' if is_unseen else 'train region'})",
                                  OUT / f"{tag}_{name}_prediction.png", classes=CLASSES)
        io_geo.save_class_map_png(ylab.values, present, f"NLUM labels — {name}",
                                  OUT / f"{tag}_{name}_labels.png", classes=CLASSES)
        # metrics on confident pixels
        elig = finite & (c >= args.conf) & (y != 0)
        m = None
        if elig.sum() > 0:
            yy, pp = y[elig], pred.ravel()[elig]
            m = {"overall_accuracy": float(accuracy_score(yy, pp)),
                 "kappa": float(cohen_kappa_score(yy, pp)), "n": int(elig.sum())}
        del cube, X, y, c, finite, ylab; gc.collect()
        return m

    print("[3] writing per-train-region rasters ...")
    for name, bbox in train_regions.items():
        predict_and_write(name, bbox)

    print("[4] UNSEEN regions ...")
    unseen_results = {}
    for uname, ubbox in unseen_regions.items():
        m = predict_and_write(uname, ubbox, is_unseen=True) or {"overall_accuracy": float("nan"), "kappa": float("nan")}
        unseen_results[uname] = m
        print(f"    UNSEEN {uname} OA={m['overall_accuracy']:.3f} kappa={m['kappa']:.3f}")
    # representative unseen metric = mean OA across held-out regions
    _oas = [m["overall_accuracy"] for m in unseen_results.values() if m["overall_accuracy"] == m["overall_accuracy"]]
    unseen_metrics = {"overall_accuracy": float(np.mean(_oas)) if _oas else float("nan"),
                      "kappa": float(np.mean([m["kappa"] for m in unseen_results.values()])) if _oas else float("nan"),
                      "per_region": unseen_results}

    # confusion + multi-region sources map
    io_geo.save_confusion_png(metrics_spatial["confusion_matrix"], metrics_spatial["labels"],
                              OUT / f"{tag}_confusion.png", classes=CLASSES,
                              title="Confusion (pooled spatial holdout)")
    _multiregion_sources_map(sources_plot, OUT / f"{tag}_training_sources.png")

    line = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag,
        "model": args.model, "indices": args.indices, "conf": args.conf,
        "train_regions": list(train_regions), "unseen_regions": list(unseen_regions),
        "unseen_per_region": {k: round(v["overall_accuracy"], 4) for k, v in unseen_results.items()},
        "n_train": int(clf.n_features_in_ and len(sources_plot) and sum(len(v[2]) for v in sources_plot.values())),
        "spatial_OA": round(metrics_spatial["overall_accuracy"], 4),
        "spatial_kappa": round(metrics_spatial["kappa"], 4),
        "unseen_OA": round(unseen_metrics["overall_accuracy"], 4),
        "unseen_kappa": round(unseen_metrics["kappa"], 4),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / f"accuracy_{tag}.json").write_text(json.dumps(
        {**line, "label_composition": comp_total, "spatial_full": metrics_spatial}, indent=2))
    md = OUT / "RESULTS_AU.md"
    header_needed = not md.exists()
    with open(md, "a") as f:
        if header_needed:
            f.write("# Australian crop-type model iterations (NLUM-labelled)\n\n")
        f.write(f"| {line['time']} | {tag} | {args.model} | idx={args.indices} | conf={args.conf} | "
                f"n_train={line['n_train']} | spatial OA **{line['spatial_OA']}** | kappa {line['spatial_kappa']} | "
                f"unseen mean OA {line['unseen_OA']} ({line['unseen_per_region']}) | {line['seconds']}s |\n")
    print("[done]", json.dumps(line))
    for k, v in metrics_spatial["report"].items():
        if k.isdigit():
            print(f"      {NAME_BY_ID.get(int(k), k):20s} F1={v['f1-score']:.2f} (n={int(v['support'])})")


def _multiregion_sources_map(sources_plot, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(sources_plot)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), squeeze=False)
    for ax, (name, (rows, cols, cls, shape)) in zip(axes[0], sources_plot.items()):
        for c in sorted(np.unique(cls)):
            sel = cls == c
            ax.scatter(cols[sel], rows[sel], s=2, color=np.array(CLASSES[c][1]) / 255,
                       label=NAME_BY_ID.get(c, str(c)))
        ax.set_title(f"{name}  (n={len(rows)})"); ax.invert_yaxis()
        ax.set_aspect("equal"); ax.axis("off")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False, markerscale=3)
    fig.suptitle("Training-sample distribution by region (source: NLUM v7 2020-21)", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


if __name__ == "__main__":
    main()
