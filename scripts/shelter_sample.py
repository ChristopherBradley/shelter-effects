"""Sample the shelter-effect dataset on Earth Engine: productivity (EVI) + shelter
treatment + confounders, at stratified points across the agricultural zone.

Treatment (from ee-christopher-bradley shelter asset, percent method):
  1 = Unsheltered Cropland (41)   2 = Sheltered Cropland (42)
  3 = Unsheltered Pasture (31)     4 = Sheltered Pasture (32)
Response: growing-season (Apr-Nov) 95th-pct EVI per year (peak productivity proxy).
Confounders: growing-season rainfall, max temp, PDSI (drought), climatic water deficit
(per year); elevation, slope, aspect; soil clay/SOC/pH/bulk-density (OpenLandMap).

Years include a drought (2019) and wet La Nina years (2020, 2021) to test whether the
shelter benefit is larger under drought. Pulls the sample as CSV (no Drive needed).

  python scripts/shelter_sample.py --region riverina --npc 1000 --name riverina
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import mask_s2, PROJECT
DATA = ROOT / "data"
SHELTER = "projects/ee-christopher-bradley/assets/Aus_ag2020_default-percentmethod"
WIND = "projects/ee-christopher-bradley/assets/Aus_ag2020_default-windmethod"

REGIONS = {
    "tiny": [147.10, -34.60, 147.50, -34.25],
    "riverina": [146.0, -36.0, 149.0, -34.0],
    "se_aus": [140.0, -38.0, 152.0, -29.0],
    "sw_wa": [115.0, -34.5, 122.0, -29.0],
}
YEARS = [2019, 2020, 2021]  # 2019 drought; 2020-21 wet La Nina


def evi_season(year):
    # peak-window (Aug 15 - Oct 31) median EVI = crop/pasture peak productivity proxy.
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-08-15", f"{year}-10-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)).map(mask_s2))
    def evi(img):
        return img.expression("2.5*(N-R)/(N+6*R-7.5*B+1)",
                              {"N": img.select("B8"), "R": img.select("B4"),
                               "B": img.select("B2")}).rename("EVI")
    return s2.map(evi).median().rename(f"evi_{year}")


def climate(year):
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(
        f"{year}-04-01", f"{year}-11-30")
    return ee.Image.cat([
        tc.select("pr").sum().rename(f"rain_{year}"),
        tc.select("tmmx").mean().multiply(0.1).rename(f"tmax_{year}"),
        tc.select("pdsi").mean().multiply(0.01).rename(f"pdsi_{year}"),
        tc.select("def").sum().multiply(0.1).rename(f"cwd_{year}"),
    ])


def static_covariates():
    dem = ee.Image("USGS/SRTMGL1_003")
    terr = ee.Terrain.products(dem)
    soil = {
        "clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
        "soc": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
        "ph": "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
        "bd": "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02",
    }
    simgs = [ee.Image(v).select("b0").rename(k) for k, v in soil.items()]
    return ee.Image.cat([dem.rename("elev"), terr.select("slope"),
                         terr.select("aspect")] + simgs)


TREE = "projects/ee-christopher-bradley/assets/Aus_2020_noxy_predictions"


def build_stack():
    shelter = ee.ImageCollection(SHELTER).mosaic().select("b1")
    wind = ee.ImageCollection(WIND).mosaic().select("b1")
    cls = shelter.remap([41, 42, 31, 32], [1, 2, 3, 4]).rename("cls")
    cls_wind = wind.remap([41, 42, 31, 32], [1, 2, 3, 4]).rename("cls_wind")
    # continuous shelter: distance (m) to nearest tree (>50% cover), capped 400 m,
    # so we can resolve the competition-near vs benefit-far productivity curve.
    treemask = ee.ImageCollection(TREE).mosaic().select("b1").gt(50)
    dist = treemask.distance(ee.Kernel.euclidean(400, "meters"), False).unmask(400).rename("dist_tree")
    bands = [cls, cls_wind, dist]
    for y in YEARS:
        bands += [evi_season(y), climate(y)]
    bands.append(static_covariates())
    return ee.Image.cat(bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="riverina", choices=list(REGIONS))
    ap.add_argument("--npc", type=int, default=1000, help="points per shelter class")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    name = args.name or args.region
    region = ee.Geometry.Rectangle(REGIONS[args.region])

    img = build_stack()
    print(f"sampling {args.npc}/class over {args.region} @ {args.scale} m ...")
    samp = img.stratifiedSample(numPoints=args.npc, classBand="cls", region=region,
                                scale=args.scale, seed=42, geometries=True, dropNulls=True,
                                tileScale=4)
    n = samp.size().getInfo()
    print(f"  {n} samples")
    url = samp.getDownloadURL(filetype="CSV")
    out = DATA / f"shelter_samples_{name}.csv"
    out.write_bytes(requests.get(url, timeout=300).content)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
