"""Fetch Atlas of Living Australia (ALA) occurrence points for a species in a bbox.

Used to build presence data for a weed species-distribution model (presence vs
background) trained on Sentinel-2 features.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BIOCACHE = "https://biocache-ws.ala.org.au/ws/occurrences/search"
GBIF = "https://api.gbif.org/v1"
HDR = {"User-Agent": "veg-mapper-research/0.1 (chris2bradley@gmail.com)"}


def fetch_occurrences_gbif(name: str, bbox, cache_csv, max_records: int = 8000,
                           page: int = 300) -> pd.DataFrame:
    """Fallback occurrence source via GBIF (which aggregates ALA + others on robust
    infra). `name` = a taxon name e.g. 'Rubus' or 'Rubus fruticosus'."""
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    m = requests.get(f"{GBIF}/species/match", params={"name": name}, headers=HDR, timeout=30).json()
    key = m.get("usageKey")
    minlon, minlat, maxlon, maxlat = bbox
    rows, offset = [], 0
    while offset < max_records:
        r = requests.get(f"{GBIF}/occurrence/search", params={
            "taxonKey": key, "decimalLatitude": f"{minlat},{maxlat}",
            "decimalLongitude": f"{minlon},{maxlon}", "hasCoordinate": "true",
            "limit": page, "offset": offset}, headers=HDR, timeout=90).json()
        res = r.get("results", [])
        for o in res:
            lon, lat = o.get("decimalLongitude"), o.get("decimalLatitude")
            if lon is not None and lat is not None:
                rows.append((lon, lat))
        if r.get("endOfRecords") or not res:
            break
        offset += page
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["lon", "lat"]).drop_duplicates()
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def fetch_presences(ala_q: str, gbif_name: str, bbox, cache_csv) -> pd.DataFrame:
    """Try ALA biocache first; fall back to GBIF if ALA is unavailable."""
    try:
        return fetch_occurrences(ala_q, bbox, cache_csv)
    except Exception as e:
        print(f"    ALA unavailable ({str(e)[:40]}); falling back to GBIF '{gbif_name}'")
        return fetch_occurrences_gbif(gbif_name, bbox, cache_csv)


def _get_json(params, retries=3):
    last = None
    for i in range(retries):
        r = requests.get(BIOCACHE, params=params, headers=HDR, timeout=90)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        last = f"{r.status_code} {r.headers.get('content-type')}"
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"ALA biocache non-JSON after {retries} tries: {last}")


def fetch_occurrences(q: str, bbox, cache_csv: str | Path, max_records: int = 8000,
                      page: int = 300) -> pd.DataFrame:
    """q: biocache query e.g. 'genus:Rubus' or 'raw_scientificName:...'.
    bbox=(minlon,minlat,maxlon,maxlat). Returns DataFrame(lon,lat). Cached."""
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    minlon, minlat, maxlon, maxlat = bbox
    fq = [f"decimalLatitude:[{minlat} TO {maxlat}]",
          f"decimalLongitude:[{minlon} TO {maxlon}]",
          "geospatial_kosher:true"]
    rows, start = [], 0
    while start < max_records:
        r = _get_json({"q": q, "fq": fq, "pageSize": page,
                       "startIndex": start, "facet": "false"})
        occ = r.get("occurrences", [])
        if not occ:
            break
        for o in occ:
            lon, lat = o.get("decimalLongitude"), o.get("decimalLatitude")
            if lon is not None and lat is not None:
                rows.append((lon, lat))
        total = r.get("totalRecords", 0)
        start += page
        if start >= total:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["lon", "lat"]).drop_duplicates()
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df
