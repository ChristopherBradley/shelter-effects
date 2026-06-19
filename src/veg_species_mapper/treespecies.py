"""Tree-species labels for the ACT from the ACT Tree Assets inventory (+ NLUM plantation
forestry for pine). Produces a per-pixel tree-group label raster to train a Sentinel-2
classifier (evergreen-native vs conifer vs deciduous-exotic), for attributing species to
shelterbelts.

Label sources:
  - ACT Tree Assets (Socrata 9qch-rvqr): 786k urban/open-space trees with genus + point.
  - NLUM v7 PLANTATION_FR probability surface: large softwood (pine) plantation stands.

Note: urban tree pixels are spectrally mixed with roads/roofs; the cleanest signal comes
from pixels where one tree group is dense (open space, reserves, plantations).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SOCRATA = "https://www.data.act.gov.au/resource/9qch-rvqr.json"
HDR = {"User-Agent": "veg-mapper-research/0.1 (chris2bradley@gmail.com)"}

# reduced tree class id -> (name, RGB)
CLASSES: dict[int, tuple[str, tuple[int, int, int]]] = {
    0: ("Non-tree/other",   (200, 200, 200)),
    1: ("Eucalypt (native)", (60, 150, 70)),
    2: ("Conifer/Pine",      (20, 80, 120)),
    3: ("Deciduous exotic",  (210, 140, 40)),
    4: ("Casuarina",         (150, 90, 160)),
}
NAME_BY_ID = {k: v[0] for k, v in CLASSES.items()}

_GENUS_TO_CLASS = {}
for g in ["Eucalyptus", "Corymbia", "Angophora"]:
    _GENUS_TO_CLASS[g] = 1
for g in ["Pinus", "Cupressus", "Cedrus", "Picea", "Cupressocyparis", "Callitris", "Sequoia"]:
    _GENUS_TO_CLASS[g] = 2
for g in ["Quercus", "Fraxinus", "Ulmus", "Populus", "Platanus", "Zelkova", "Liquidambar",
          "Prunus", "Pyrus", "Acer", "Celtis", "Pistacia", "Gleditsia", "Betula",
          "Tilia", "Robinia", "Fagus", "Sorbus", "Malus", "Salix"]:
    _GENUS_TO_CLASS[g] = 3
_GENUS_TO_CLASS["Casuarina"] = 4
_GENUS_TO_CLASS["Allocasuarina"] = 4


def genus_to_class(genus) -> int:
    if not isinstance(genus, str):
        return 0
    return _GENUS_TO_CLASS.get(genus.strip().title(), 0)


def fetch_act_trees(bbox, cache_csv: str | Path, page: int = 50000) -> pd.DataFrame:
    """Download ACT tree points (genus, lon, lat) within bbox=(minlon,minlat,maxlon,maxlat).
    Cached to CSV."""
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    minlon, minlat, maxlon, maxlat = bbox
    where = f"within_box(the_geom,{maxlat},{minlon},{minlat},{maxlon})"
    rows, offset = [], 0
    while True:
        r = requests.get(SOCRATA, params={
            "$select": "genus,the_geom", "$where": where,
            "$limit": page, "$offset": offset}, headers=HDR, timeout=120).json()
        if not r:
            break
        for d in r:
            g = d.get("the_geom")
            if g and g.get("coordinates"):
                lon, lat = g["coordinates"][:2]
                rows.append((d.get("genus"), lon, lat))
        offset += page
        if len(r) < page:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["genus", "lon", "lat"])
    df["cls"] = df["genus"].map(genus_to_class)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def rasterize_tree_labels(df: pd.DataFrame, transform, crs, shape, min_trees: int = 3):
    """Majority tree-class per pixel from points, keeping only pixels with >= min_trees.

    transform: affine (rasterio) mapping pixel->crs coords of the target grid.
    crs: target CRS (e.g. UTM). df has lon/lat (EPSG:4326).
    Returns a uint8 label raster (0 where below min_trees)."""
    from pyproj import Transformer
    import rasterio
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform(df["lon"].values, df["lat"].values)
    inv = ~transform
    cols, rows = inv * (np.array(xs), np.array(ys))
    cols = np.floor(cols).astype(int); rows = np.floor(rows).astype(int)
    ny, nx = shape
    ok = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)
    cls = df["cls"].values
    # per-pixel vote counts per class
    n_cls = max(CLASSES) + 1
    votes = np.zeros((n_cls, ny, nx), dtype="int32")
    flat = rows[ok] * nx + cols[ok]
    for c in range(1, n_cls):
        sel = ok & (cls == c)
        if sel.any():
            f = (rows[sel] * nx + cols[sel])
            np.add.at(votes[c].reshape(-1), f, 1)
    total = votes.sum(axis=0)
    label = votes.argmax(axis=0).astype("uint8")
    label[total < min_trees] = 0
    return label, total
