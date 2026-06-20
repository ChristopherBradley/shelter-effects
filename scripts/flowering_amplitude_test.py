"""Definitive flowering-index test: does a PER-SCENE seasonal-max amplitude of CFI vs
NDYI separate canola from cereals?

The earlier A/B computed indices from monthly-median composites, which smooths the brief
flowering peak (and a tree model already has the raw bands). Here we load per-scene S2 over
the flowering window, compute the index per acquisition, take the pixel-wise seasonal MAX
(the flowering amplitude), and report single-feature ROC AUC for canola-vs-cereal — the
cleanest measure of "how noticeable" each index is.

  python scripts/flowering_amplitude_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rioxarray  # noqa: F401 (registers .rio)
from odc.stac import load as odc_load
from pystac_client import Client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from veg_species_mapper.cropmap import nlum
from veg_species_mapper.cropmap.data import utm_epsg

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
# canola-bearing regions (from the national set)
REGIONS = {
    "Temora_NSW": (147.25, -34.50, 147.42, -34.35),
    "Clare_SA": (138.55, -33.85, 138.72, -33.70),
}
# Southern-AU canola flowers ~Aug-Oct
WINDOW = ("2020-08-01", "2020-10-31")
SCL_DROP = {0, 1, 3, 8, 9, 10, 11}


def per_scene_amplitudes(bbox):
    epsg = utm_epsg((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(bbox),
                 datetime=f"{WINDOW[0]}/{WINDOW[1]}",
                 query={"eo:cloud_cover": {"lt": 60}}).items())
    ds = odc_load(items, bands=["B02", "B03", "B04", "B08", "SCL"], bbox=list(bbox),
                  crs=f"EPSG:{epsg}", resolution=10, groupby="solar_day",
                  chunks={"x": 2048, "y": 2048}, dtype="uint16", resampling="bilinear")
    valid = ~ds["SCL"].isin(list(SCL_DROP))
    b = {k: (ds[k].where(valid) / 10000.0) for k in ["B02", "B03", "B04", "B08"]}
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 1e-6)
    ndyi = (b["B03"] - b["B02"]) / (b["B03"] + b["B02"] + 1e-6)
    cfi = ndvi * (b["B04"] + 2 * b["B03"] - b["B02"])
    print(f"    {len(items)} scenes; computing seasonal max ...")
    nm = ndyi.max("time").compute().rio.write_crs(f"EPSG:{epsg}")
    cm = cfi.max("time").compute().rio.write_crs(f"EPSG:{epsg}")
    return nm, cm


def main():
    from sklearn.metrics import roc_auc_score
    print("Canola-vs-cereal single-feature separability (seasonal-max amplitude):\n")
    print(f"{'region':14s} {'NDYI_max AUC':>13s} {'CFI_max AUC':>12s} {'n_canola':>9s} {'n_cereal':>9s}")
    for name, bbox in REGIONS.items():
        print(f"[{name}] loading per-scene flowering window ...")
        ndyi_max, cfi_max = per_scene_amplitudes(bbox)
        ylab, conf = nlum.labels_on_grid(bbox, like=ndyi_max)
        y = ylab.values.ravel(); c = conf.values.ravel()
        ndyi = ndyi_max.values.ravel(); cfi = cfi_max.values.ravel()
        ok = np.isfinite(ndyi) & np.isfinite(cfi) & (c >= 0.7)
        canola = ok & (y == 2); cereal = ok & (y == 1)
        m = canola | cereal
        target = canola[m].astype(int)
        auc_ndyi = roc_auc_score(target, ndyi[m])
        auc_cfi = roc_auc_score(target, cfi[m])
        print(f"{name:14s} {auc_ndyi:13.3f} {auc_cfi:12.3f} {int(canola.sum()):9d} {int(cereal.sum()):9d}")
    print("\n(AUC = how well the single index alone ranks canola above cereal; 0.5=useless, 1=perfect)")


if __name__ == "__main__":
    main()
