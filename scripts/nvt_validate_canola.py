"""Validate canola flowering indices against NVT trial sites (real ground truth).

For a year, sample each NVT trial site (canola = positive; wheat/barley = negative) with
the candidate index computed server-side on GEE, then report ROC AUC. Compares CFI_max
(seasonal magnitude) vs a flowering-SPIKE anomaly (peak minus baseline), to see which is
actually canola-specific. Processed per-state to bound each EE computation.

NVT GPS can sit at a paddock corner, so we reduce over a small buffer (default 70 m).

  python scripts/nvt_validate_canola.py --year 2020 --buffer 70 --reducer p80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import mask_s2, cfi, PROJECT

NVT = ROOT / "data" / "sensitive_data" / "trials.gpkg"
STATE_BBOX = {  # rough bounds to bound the S2 filter per state
    "NSW": [140.9, -37.6, 153.7, -28.1], "VIC": [140.9, -39.3, 150.1, -33.9],
    "SA": [129, -38.2, 141.1, -31.5], "WA": [114, -35.3, 124, -27.5],
}


def ndyi(img):
    g, b = img.select("B3"), img.select("B2")
    return g.subtract(b).divide(g.add(b).add(1e-6)).rename("NDYI")


def build_indices(bbox, year):
    aoi = ee.Geometry.Rectangle(bbox)
    def coll(d0, d1):
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
                .filterDate(f"{year}-{d0}", f"{year}-{d1}")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(mask_s2))
    flower = coll("08-01", "10-15")
    cfimax = flower.map(cfi).max().rename("CFI_max")
    # flowering spike = peak yellowness (Aug-Oct) minus green-baseline yellowness (Jun + Nov)
    base = coll("06-01", "07-15").merge(coll("11-01", "11-30"))
    spike = flower.map(ndyi).max().subtract(base.map(ndyi).median()).rename("CFI_spike")
    return ee.Image.cat([cfimax, spike])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--buffer", type=float, default=70)
    ap.add_argument("--reducer", default="p80", choices=["mean", "max", "p80"])
    ap.add_argument("--max-neg", type=int, default=250, help="cap negatives per state")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    red = {"mean": ee.Reducer.mean(), "max": ee.Reducer.max(),
           "p80": ee.Reducer.percentile([80])}[args.reducer]

    g = gpd.read_file(NVT)
    g = g.dropna(subset=["Trial GPS Lat", "Trial GPS Long"])
    g = g[g["Year"] == args.year]
    g["pos"] = g["Crop.Name"].str.contains("anola", na=False)
    g = g[g["Crop.Name"].isin(["Canola", "Wheat", "Barley"])]
    g = g.drop_duplicates(subset=["Trial GPS Lat", "Trial GPS Long", "Crop.Name"])

    rows = []
    for state, bbox in STATE_BBOX.items():
        sub = g[g["State"] == state]
        pos = sub[sub["pos"]]; neg = sub[~sub["pos"]].head(args.max_neg)
        sites = (list(zip(pos["Trial GPS Long"], pos["Trial GPS Lat"], [1] * len(pos)))
                 + list(zip(neg["Trial GPS Long"], neg["Trial GPS Lat"], [0] * len(neg))))
        if not pos.size:
            continue
        feats = [ee.Feature(ee.Geometry.Point([lo, la]).buffer(args.buffer), {"pos": p})
                 for lo, la, p in sites]
        fc = ee.FeatureCollection(feats)
        img = build_indices(bbox, args.year)
        sampled = img.reduceRegions(collection=fc, reducer=red, scale=10).getInfo()
        for f in sampled["features"]:
            pr = f["properties"]
            rows.append((state, pr.get("pos"), pr.get("CFI_max") or pr.get("CFI_max_p80"),
                         pr.get("CFI_spike") or pr.get("CFI_spike_p80")))
        print(f"  {state}: {len(pos)} canola + {len(neg)} cereal sampled")

    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    df = pd.DataFrame(rows, columns=["state", "pos", "cfi_max", "cfi_spike"]).dropna()
    out_csv = ROOT / "data" / f"nvt_canola_validation_{args.year}.csv"  # gitignored
    df.to_csv(out_csv, index=False)
    print(f"\nNVT canola validation {args.year} (n={len(df)}, canola={int(df.pos.sum())}):")
    for col in ["cfi_max", "cfi_spike"]:
        auc = roc_auc_score(df["pos"], df[col])
        print(f"  {col:10s} AUC = {auc:.3f}")
    # operating threshold for cfi_max
    fpr, tpr, thr = roc_curve(df["pos"], df["cfi_max"])
    j = int(np.argmax(tpr - fpr))
    prec, rec, pthr = precision_recall_curve(df["pos"], df["cfi_max"])
    hi = np.where(prec[:-1] >= 0.90)[0]
    print(f"  CFI_max Youden thr={thr[j]:.3f} (TPR={tpr[j]:.2f}, FPR={fpr[j]:.2f})")
    if len(hi):
        i = hi[0]
        print(f"  CFI_max thr={pthr[i]:.3f} gives precision {prec[i]:.2f} at recall {rec[i]:.2f}")
    for st in df.state.unique():
        d = df[df.state == st]
        if d.pos.nunique() > 1:
            print(f"    {st}: CFI_max {roc_auc_score(d.pos,d.cfi_max):.2f} | "
                  f"spike {roc_auc_score(d.pos,d.cfi_spike):.2f} (n={len(d)}, canola={int(d.pos.sum())})")


if __name__ == "__main__":
    main()
