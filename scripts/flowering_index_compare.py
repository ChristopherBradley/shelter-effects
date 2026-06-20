"""A/B test canola-flowering features: base vs +NDYI vs +CFI (and both).

Train-only on cached national regions (fast, no prediction rasters). Same spatial
train/test split and sampled pixels across variants for a fair comparison. Reports
per-class F1 with sample sizes, highlighting Canola.

  python scripts/flowering_index_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veg_species_mapper.cropmap import data, model, nlum
from veg_species_mapper.cropmap.legend_au import NAME_BY_ID

REGIONS = {
    "Temora_NSW": (147.25, -34.50, 147.42, -34.35),
    "Wimmera_VIC": (142.30, -36.55, 142.47, -36.40),
    "Corowa_NSW": (146.75, -35.95, 146.92, -35.80),
    "Merredin_WA": (118.20, -31.55, 118.37, -31.40),
    "Clare_SA": (138.55, -33.85, 138.72, -33.70),
    "Cootamundra_NSW": (148.00, -34.70, 148.17, -34.55),
    "Mallee_VIC": (142.40, -35.30, 142.57, -35.15),
}
YEAR, MONTHS = 2020, (5, 6, 7, 8, 9, 10, 11)
VARIANTS = ["base", "ndyi", "cfi", "both"]
CONF, NPC, BLOCK = 0.7, 4000, 40


def variant_cube(base, raw_cube):
    out = {}
    out["base"] = base
    out["ndyi"] = model.flowering_index(base, MONTHS, "ndyi")
    out["cfi"] = model.flowering_index(base, MONTHS, "cfi")
    out["both"] = model.flowering_index(out["ndyi"], MONTHS, "cfi")
    return out


def main():
    pooled = {v: {"Xtr": [], "ytr": [], "Xte": [], "yte": []} for v in VARIANTS}
    for name, bbox in REGIONS.items():
        print(f"[build] {name} ...")
        raw = data.s2_monthly_composite(bbox, year=YEAR, months=MONTHS)
        base = model.add_indices(raw, MONTHS)
        ylab, conf = nlum.labels_on_grid(bbox, like=base)
        # one split + sample, shared across variants
        Xb, shape, _ = model.to_feature_matrix(base)
        Xb, finite = model.finite_and_fill(Xb)
        y = ylab.values.ravel().astype("uint8")
        c = conf.values.ravel().astype("float32")
        elig = finite & (c >= CONF) & (y != 0)
        is_test = model.spatial_block_split(shape, block_px=BLOCK, test_frac=0.3)
        tr = model.sample_training(y, is_test, elig, n_per_class=NPC, drop_classes=(0,))
        te = np.where(elig & is_test)[0]
        if len(te) > 40000:
            te = np.random.default_rng(0).choice(te, 40000, replace=False)
        cubes = variant_cube(base, raw)
        for v in VARIANTS:
            Xv, _, _ = model.to_feature_matrix(cubes[v])
            Xv, _ = model.finite_and_fill(Xv)
            pooled[v]["Xtr"].append(Xv[tr]); pooled[v]["ytr"].append(y[tr])
            pooled[v]["Xte"].append(Xv[te]); pooled[v]["yte"].append(y[te])
            del Xv

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    results = {}
    nfeat = {}
    for v in VARIANTS:
        Xtr = np.concatenate(pooled[v]["Xtr"]); ytr = np.concatenate(pooled[v]["ytr"])
        Xte = np.concatenate(pooled[v]["Xte"]); yte = np.concatenate(pooled[v]["yte"])
        nfeat[v] = Xtr.shape[1]
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        rep = classification_report(yte, pred, output_dict=True, zero_division=0)
        results[v] = {"OA": accuracy_score(yte, pred), "report": rep, "yte": yte}

    classes = sorted({int(k) for v in VARIANTS for k in results[v]["report"] if k.isdigit()})
    print("\n=== Flowering-index A/B (spatial-holdout F1 per class) ===")
    hdr = f"{'class':20s} " + "".join(f"{v:>9s}" for v in VARIANTS) + f"{'test n':>9s}"
    print(hdr)
    yte0 = results["base"]["yte"]
    for c in classes:
        n = int((yte0 == c).sum())
        row = f"{NAME_BY_ID.get(c, c):20s} "
        for v in VARIANTS:
            f1 = results[v]["report"].get(str(c), {}).get("f1-score", 0.0)
            row += f"{f1:9.2f}"
        row += f"{n:9d}"
        print(row + ("   <- CANOLA" if c == 2 else ""))
    print("\noverall OA: " + "  ".join(f"{v}={results[v]['OA']:.3f}(feat {nfeat[v]})" for v in VARIANTS))


if __name__ == "__main__":
    main()
