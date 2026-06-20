"""Earth Engine canola-flowering (CFI_max) map for an AOI/year, server-side.

CFI = NDVI*(red + 2*green - blue); we take the per-pixel MAX over the flowering window
(the flowering peak). High CFI_max = likely canola that year. Pulls a thumbnail to view
and prints percentile stats. No image downloads — runs on GEE free (noncommercial).

  python scripts/ee_canola.py --bbox 147.25 -34.50 147.42 -34.35 --year 2020 --name temora

Project: ee-christopher-bradley (compute only; existing assets are READ-ONLY, untouched).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ee
import requests

OUT = Path(__file__).resolve().parent.parent / "outputs"
PROJECT = "ee-christopher-bradley"
SCL_BAD = [0, 1, 3, 8, 9, 10, 11]


def mask_s2(img):
    scl = img.select("SCL")
    bad = scl.remap(SCL_BAD, [1] * len(SCL_BAD), 0)
    return img.updateMask(bad.Not()).divide(10000)


def cfi(img):
    nir, red = img.select("B8"), img.select("B4")
    grn, blu = img.select("B3"), img.select("B2")
    ndvi = nir.subtract(red).divide(nir.add(red).add(1e-6))
    return ndvi.multiply(red.add(grn.multiply(2)).subtract(blu)).rename("CFI")


def cfi_max(bbox, year, flower=("07-01", "11-30")):
    aoi = ee.Geometry.Rectangle(list(bbox))
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(f"{year}-{flower[0]}", f"{year}-{flower[1]}")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(mask_s2))
    return col.map(cfi).max().clip(aoi), aoi, col.size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--name", default="aoi")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)

    cmax, aoi, n = cfi_max(args.bbox, args.year)
    print(f"scenes: {n.getInfo()}")
    stats = cmax.reduceRegion(ee.Reducer.percentile([5, 50, 90, 99]), aoi, 20, maxPixels=1e9).getInfo()
    print("CFI_max percentiles:", {k: round(v, 3) for k, v in stats.items() if v is not None})
    url = cmax.getThumbURL({"min": 0, "max": 0.35, "palette": ["000044", "228822", "ffff00"],
                            "dimensions": 700, "region": aoi})
    out = OUT / f"ee_canola_cfi_{args.name}_{args.year}.png"
    out.write_bytes(requests.get(url, timeout=90).content)
    print(f"thumbnail -> {out}")


if __name__ == "__main__":
    main()
