"""Does a per-scene CFI seasonal-max amplitude feature lift canola at the MODEL level?

base features vs base + CFI_max (Jul-Nov per-scene amplitude), pooled over canola-bearing
regions, spatial-holdout. Reports per-class F1 (Canola highlighted).

  python scripts/canola_cfi_model_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from veg_species_mapper.cropmap import data, model, nlum
from veg_species_mapper.cropmap.legend_au import NAME_BY_ID

REGIONS = {
    "Temora_NSW": (147.25, -34.50, 147.42, -34.35),
    "Corowa_NSW": (146.75, -35.95, 146.92, -35.80),
    "Clare_SA": (138.55, -33.85, 138.72, -33.70),
    "Cootamundra_NSW": (148.00, -34.70, 148.17, -34.55),
}
YEAR, MONTHS = 2020, (5, 6, 7, 8, 9, 10, 11)
FLOWER_MONTHS = (7, 8, 9, 10, 11)
CONF, NPC, BLOCK = 0.7, 4000, 40
VARIANTS = ["base", "cfi_amp"]


def main():
    pooled = {v: {"Xtr": [], "ytr": [], "Xte": [], "yte": []} for v in VARIANTS}
    for name, bbox in REGIONS.items():
        print(f"[build] {name} ...")
        base = model.add_indices(data.s2_monthly_composite(bbox, year=YEAR, months=MONTHS), MONTHS)
        amp = data.flowering_amplitude(bbox, year=YEAR, months=FLOWER_MONTHS, index="cfi")
        amp = amp.rio.reproject_match(base.isel(band=0)).expand_dims(band=["CFI_max"])
        cfi_cube = xr.concat([base, amp], dim="band")
        ylab, conf = nlum.labels_on_grid(bbox, like=base)

        Xb, shape, _ = model.to_feature_matrix(base)
        Xb, finite = model.finite_and_fill(Xb)
        y = ylab.values.ravel().astype("uint8"); c = conf.values.ravel().astype("float32")
        elig = finite & (c >= CONF) & (y != 0)
        is_test = model.spatial_block_split(shape, block_px=BLOCK, test_frac=0.3)
        tr = model.sample_training(y, is_test, elig, n_per_class=NPC, drop_classes=(0,))
        te = np.where(elig & is_test)[0]
        if len(te) > 40000:
            te = np.random.default_rng(0).choice(te, 40000, replace=False)
        for v, cube in (("base", base), ("cfi_amp", cfi_cube)):
            Xv, _, _ = model.to_feature_matrix(cube)
            Xv, _ = model.finite_and_fill(Xv)
            pooled[v]["Xtr"].append(Xv[tr]); pooled[v]["ytr"].append(y[tr])
            pooled[v]["Xte"].append(Xv[te]); pooled[v]["yte"].append(y[te]); del Xv

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    res = {}
    for v in VARIANTS:
        Xtr = np.concatenate(pooled[v]["Xtr"]); ytr = np.concatenate(pooled[v]["ytr"])
        Xte = np.concatenate(pooled[v]["Xte"]); yte = np.concatenate(pooled[v]["yte"])
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        res[v] = {"OA": accuracy_score(yte, pred),
                  "rep": classification_report(yte, pred, output_dict=True, zero_division=0),
                  "yte": yte, "nf": Xtr.shape[1]}

    classes = sorted({int(k) for v in VARIANTS for k in res[v]["rep"] if k.isdigit()})
    print("\n=== base vs +CFI_max amplitude (spatial-holdout F1) ===")
    print(f"{'class':20s} {'base':>8s} {'cfi_amp':>9s} {'test n':>8s}")
    yte0 = res["base"]["yte"]
    for c in classes:
        n = int((yte0 == c).sum())
        b = res["base"]["rep"].get(str(c), {}).get("f1-score", 0)
        a = res["cfi_amp"]["rep"].get(str(c), {}).get("f1-score", 0)
        tag = "   <- CANOLA" if c == 2 else ""
        print(f"{NAME_BY_ID.get(c,c):20s} {b:8.2f} {a:9.2f} {n:8d}{tag}")
    print(f"\nOA: base={res['base']['OA']:.3f}(feat {res['base']['nf']})  "
          f"cfi_amp={res['cfi_amp']['OA']:.3f}(feat {res['cfi_amp']['nf']})")


if __name__ == "__main__":
    main()
