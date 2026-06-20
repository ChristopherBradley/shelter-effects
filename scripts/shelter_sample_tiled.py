"""Tiled shelter-effect sampling across the Australian cropping zone.

Samples many small tiles (each fits the GEE interactive limit). Within a tile, sheltered
vs unsheltered pixels share climate/soil/region -> a naturally matched comparison; pooled
across tiles we also get the environmental gradient. Saves incrementally so partial runs
are usable.

  python scripts/shelter_sample_tiled.py --zone se --tile 0.4 --step 0.9 --npc 300
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import ee
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import PROJECT
from shelter_sample import build_stack
DATA = ROOT / "data"

ZONES = {  # southern wheat-sheep belt, split into two big boxes
    "se": [138.0, -38.0, 150.5, -28.0],   # SA-east / VIC / NSW / sthn QLD
    "sw": [115.0, -34.5, 122.5, -28.0],   # SW Western Australia
}


def tile_centers(bbox, step):
    lon0, lat0, lon1, lat1 = bbox
    lons = [lon0 + step * (i + 0.5) for i in range(int((lon1 - lon0) / step))]
    lats = [lat0 + step * (i + 0.5) for i in range(int((lat1 - lat0) / step))]
    return [(lo, la) for la in lats for lo in lons]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="se", choices=list(ZONES))
    ap.add_argument("--tile", type=float, default=0.4, help="tile half-extent? no: full deg")
    ap.add_argument("--step", type=float, default=0.9)
    ap.add_argument("--npc", type=int, default=300)
    ap.add_argument("--scale", type=int, default=20)
    ap.add_argument("--max-tiles", type=int, default=120)
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    stack = build_stack()

    out = DATA / f"shelter_samples_tiled_{args.zone}.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(prev["tile"].unique())
        print(f"resuming: {len(prev)} rows, {len(done)} tiles done")

    centers = tile_centers(ZONES[args.zone], args.step)
    import random; random.Random(0).shuffle(centers)
    centers = centers[: args.max_tiles]
    n_new = 0
    for i, (lo, la) in enumerate(centers):
        tid = f"{lo:.2f}_{la:.2f}"
        if tid in done:
            continue
        h = args.tile / 2
        region = ee.Geometry.Rectangle([lo - h, la - h, lo + h, la + h])
        try:
            samp = stack.stratifiedSample(numPoints=args.npc, classBand="cls", region=region,
                                          scale=args.scale, seed=42, geometries=True,
                                          dropNulls=True, tileScale=4)
            csv = requests.get(samp.getDownloadURL(filetype="CSV"), timeout=290).content
            df = pd.read_csv(io.BytesIO(csv))
            if len(df) == 0:
                print(f"  [{i+1}/{len(centers)}] {tid}: empty"); continue
            df["tile"] = tid; df["tile_lon"] = lo; df["tile_lat"] = la
            df.to_csv(out, mode="a", header=not out.exists(), index=False)
            n_new += 1
            print(f"  [{i+1}/{len(centers)}] {tid}: +{len(df)} rows (total tiles +{n_new})")
        except Exception as e:
            print(f"  [{i+1}/{len(centers)}] {tid}: skip ({str(e)[:50]})")
        time.sleep(0.5)
    print(f"done: {n_new} new tiles -> {out}")


if __name__ == "__main__":
    main()
