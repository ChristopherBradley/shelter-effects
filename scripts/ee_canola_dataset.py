"""Generate a canola-paddock boundary dataset (GeoJSON) for an AOI/year on GEE.

CFI_max > threshold (NVT-validated default 0.23 = ~91% precision) -> vectorise to
paddock polygons (>min_ha) -> save GeoJSON + thumbnail + stats. Year-tagged, so it
handles canola<->wheat rotation. Built for the shelter-effect analysis.

  python scripts/ee_canola_dataset.py --bbox 147.25 -34.50 147.42 -34.35 --year 2020 \
      --thr 0.23 --name temora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ee
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import cfi_max, PROJECT
OUT = ROOT / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--thr", type=float, default=0.23, help="CFI_max threshold (NVT-validated)")
    ap.add_argument("--min-ha", type=float, default=0.5)
    ap.add_argument("--name", default="aoi")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)

    cmax, aoi, n = cfi_max(tuple(args.bbox), args.year)
    mask = cmax.gt(args.thr).selfMask().rename("canola")
    vecs = (mask.reduceToVectors(geometry=aoi, scale=10, geometryType="polygon",
                                 maxPixels=1e9, bestEffort=True)
            .map(lambda f: f.set({"ha": f.area(10).divide(1e4), "year": args.year}))
            .filter(ee.Filter.gt("ha", args.min_ha)))
    area_ha = mask.multiply(ee.Image.pixelArea()).reduceRegion(
        ee.Reducer.sum(), aoi, 10, maxPixels=1e9).getInfo().get("canola", 0) / 1e4
    nfeat = vecs.size().getInfo()
    print(f"[{args.name} {args.year}] scenes={n.getInfo()} thr={args.thr} "
          f"-> {nfeat} canola paddocks, ~{area_ha:.0f} ha")

    # GeoJSON (getInfo OK for small AOIs; use Export.table.toDrive at regional scale)
    if nfeat <= 4000:
        gj = vecs.getInfo()
        out = OUT / f"canola_paddocks_{args.name}_{args.year}.geojson"
        out.write_text(json.dumps(gj))
        print(f"  saved {out.name}")
    else:
        print("  >4000 polygons: use Export.table.toDrive for the full dataset")

    thumb = (cmax.gt(args.thr).selfMask().visualize(palette=["ffd000"])
             .blend(ee.Image().paint(vecs, 1, 2).visualize(palette=["ff0000"])))
    (OUT / f"canola_paddocks_{args.name}_{args.year}.png").write_bytes(
        requests.get(thumb.getThumbURL({"dimensions": 700, "region": aoi}), timeout=90).content)
    print(f"  thumbnail saved")


if __name__ == "__main__":
    main()
