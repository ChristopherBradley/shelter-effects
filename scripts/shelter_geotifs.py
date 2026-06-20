"""Export shelter-effect GeoTIFFs for a showcase region (default Riverina) from GEE.

Produces ~10 rasters for QGIS: peak EVI (drought/wet), drought anomaly, shelter class,
distance-to-tree, tree cover, growing-season rainfall, PDSI, and an empirical
'shelter-benefit' map (the distance-decay curve applied to the distance raster).

  python scripts/shelter_geotifs.py --bbox 147.05 -35.1 147.55 -34.65 --name riverina
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import PROJECT
from shelter_sample import evi_season, climate, SHELTER, TREE
OUT = ROOT / "outputs" / "shelter_geotifs"


def export(img, aoi, name, scale=20, crs="EPSG:4326"):
    url = img.getDownloadURL({"region": aoi, "scale": scale, "crs": crs, "format": "GEO_TIFF"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tif").write_bytes(requests.get(url, timeout=300).content)
    print(f"  {name}.tif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=[147.05, -35.1, 147.55, -34.65])
    ap.add_argument("--name", default="riverina")
    ap.add_argument("--scale", type=int, default=20)
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    aoi = ee.Geometry.Rectangle(args.bbox)
    n = args.name

    evi19, evi20, evi21 = evi_season(2019), evi_season(2020), evi_season(2021)
    anomaly = evi19.subtract(evi20.add(evi21).divide(2)).rename("evi_drought_anomaly")
    shelter = ee.ImageCollection(SHELTER).mosaic().select("b1").clip(aoi)
    tree = ee.ImageCollection(TREE).mosaic().select("b1")
    dist = tree.gt(50).distance(ee.Kernel.euclidean(400, "meters"), False).unmask(400).rename("dist_tree")

    # empirical shelter-benefit curve (from analysis, pasture-ish): piecewise on distance.
    # penalty near trees, peak ~+0.02 at ~120 m, decay to 0 by ~350 m.
    d = dist
    benefit = (ee.Image(0)
               .where(d.lt(120), d.subtract(50).multiply(0.02 / 70))      # rise -0.014..+0.02
               .where(d.gte(120).And(d.lt(350)), ee.Image(0.02).subtract(d.subtract(120).multiply(0.02 / 230)))
               .where(d.gte(350), 0)).rename("shelter_benefit").clip(aoi)

    c19 = climate(2019)
    layers = {
        f"{n}_evi_2019_drought": evi19.clip(aoi),
        f"{n}_evi_2021_wet": evi21.clip(aoi),
        f"{n}_evi_drought_anomaly": anomaly.clip(aoi),
        f"{n}_shelter_class": shelter,
        f"{n}_dist_to_tree": dist.clip(aoi),
        f"{n}_tree_cover": tree.clip(aoi),
        f"{n}_rain_gs_2019": c19.select("rain_2019").clip(aoi),
        f"{n}_pdsi_2019": c19.select("pdsi_2019").clip(aoi),
        f"{n}_shelter_benefit": benefit,
    }
    print(f"exporting {len(layers)} geotiffs for {n} @ {args.scale} m ...")
    for name, img in layers.items():
        try:
            export(img, aoi, name, args.scale)
        except Exception as e:
            print(f"  {name}: FAILED {str(e)[:60]}")


if __name__ == "__main__":
    main()
