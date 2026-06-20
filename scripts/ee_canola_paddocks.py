"""Canola paddock extractor (GEE compute + local NLUM validation), Temora 2020.

1. Compute CFI_max (flowering peak) server-side on Earth Engine.
2. Download the small CFI_max raster; locally pick the threshold that best separates
   NLUM canola from the rest (ROC / Youden's J) and report precision/recall.
3. Apply the threshold on EE, vectorise to canola paddock polygons, report count/area,
   and save thumbnails (CFI_max and the canola mask).

  python scripts/ee_canola_paddocks.py --bbox 147.25 -34.50 147.42 -34.35 --year 2020 --name temora
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import ee
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from veg_species_mapper.cropmap import nlum
from veg_species_mapper.cropmap.data import utm_epsg
from ee_canola import cfi_max, PROJECT  # reuse CFI logic
OUT = ROOT / "outputs"


def download_image(img, aoi, epsg, scale=10):
    url = img.getDownloadURL({"region": aoi, "scale": scale, "crs": f"EPSG:{epsg}",
                              "format": "GEO_TIFF"})
    r = requests.get(url, timeout=180); r.raise_for_status()
    f = Path(tempfile.mktemp(suffix=".tif")); f.write_bytes(r.content)
    import rioxarray
    return rioxarray.open_rasterio(f).squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--name", default="temora")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    bbox = tuple(args.bbox)
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    cmax, aoi, n = cfi_max(bbox, args.year)
    print(f"[1] CFI_max on {n.getInfo()} scenes; downloading raster for validation ...")
    cfimg = download_image(cmax, aoi, epsg, scale=10).rio.write_crs(f"EPSG:{epsg}")

    print("[2] validating threshold vs NLUM ...")
    ylab, conf = nlum.labels_on_grid(bbox, like=cfimg)
    y = ylab.values.ravel(); c = conf.values.ravel(); v = cfimg.values.ravel().astype("float32")
    ok = np.isfinite(v) & (c >= 0.7) & (y != 0)
    target = (y[ok] == 2).astype(int)          # canola vs all other confident crops
    score = v[ok]
    from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
    auc = roc_auc_score(target, score)
    fpr, tpr, thr = roc_curve(target, score)
    j = np.argmax(tpr - fpr); best = float(thr[j])
    pred = (score >= best).astype(int)
    prec = precision_score(target, pred, zero_division=0)
    rec = recall_score(target, pred, zero_division=0)
    print(f"    canola-vs-rest AUC={auc:.3f}  best CFI_max thr={best:.3f}  "
          f"precision={prec:.2f} recall={rec:.2f}  (canola px={int(target.sum())}/{len(target)})")

    print("[3] vectorising canola paddocks on EE at threshold ...")
    mask = cmax.gt(best).selfMask().rename("canola")
    area_ha = mask.multiply(ee.Image.pixelArea()).reduceRegion(
        ee.Reducer.sum(), aoi, 10, maxPixels=1e9).getInfo().get("canola", 0) / 1e4
    vecs = mask.reduceToVectors(geometry=aoi, scale=10, geometryType="polygon",
                                maxPixels=1e9, bestEffort=True)
    # keep paddock-sized polygons (>0.5 ha) to drop speckle
    vecs = vecs.map(lambda f: f.set("ha", f.area(10).divide(1e4))).filter(ee.Filter.gt("ha", 0.5))
    print(f"    canola area ~{area_ha:.0f} ha; paddock polygons (>0.5 ha): {vecs.size().getInfo()}")

    thumb = mask.visualize(palette=["ffff00"]).blend(
        ee.Image().paint(vecs, 1, 1).visualize(palette=["ff0000"]))
    url = thumb.getThumbURL({"dimensions": 700, "region": aoi})
    (OUT / f"ee_canola_paddocks_{args.name}_{args.year}.png").write_bytes(requests.get(url, timeout=90).content)
    print(f"    thumbnail -> outputs/ee_canola_paddocks_{args.name}_{args.year}.png")


if __name__ == "__main__":
    main()
