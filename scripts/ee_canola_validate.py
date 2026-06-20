"""Visual validation of canola extraction against the imagery itself: overlay the
extracted canola paddock polygons on a true-colour Sentinel-2 flowering composite.
Canola flowers bright yellow, so polygons should land on visibly yellow fields.

  python scripts/ee_canola_validate.py --bbox 147.25 -34.50 147.42 -34.35 --year 2020 --thr 0.2 --name temora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import cfi_max, mask_s2, PROJECT
OUT = ROOT / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--thr", type=float, default=0.20)
    ap.add_argument("--name", default="temora")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    bbox = tuple(args.bbox)

    cmax, aoi, _ = cfi_max(bbox, args.year)
    mask = cmax.gt(args.thr).selfMask()
    vecs = (mask.rename("c").reduceToVectors(geometry=aoi, scale=10, geometryType="polygon",
            maxPixels=1e9, bestEffort=True)
            .map(lambda f: f.set("ha", f.area(10).divide(1e4))).filter(ee.Filter.gt("ha", 0.5)))

    # true-colour median over the flowering window (canola shows yellow)
    tc = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(f"{args.year}-08-15", f"{args.year}-10-10")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)).map(mask_s2)
          .select(["B4", "B3", "B2"]).median().clip(aoi))
    base = tc.visualize(min=0, max=0.3)
    outline = ee.Image().paint(vecs, 1, 2).visualize(palette=["ff0000"])
    overlay = base.blend(outline)

    for img, suffix in [(base, "truecolour"), (overlay, "overlay")]:
        url = img.getThumbURL({"dimensions": 750, "region": aoi})
        (OUT / f"ee_canola_validate_{args.name}_{suffix}.png").write_bytes(
            requests.get(url, timeout=90).content)
    print(f"canola polygons (>0.5ha): {vecs.size().getInfo()}")
    print(f"saved outputs/ee_canola_validate_{args.name}_truecolour.png and _overlay.png")


if __name__ == "__main__":
    main()
