"""RESTREND-style before/after test (thesis task 2d): did cropland that GAINED shelter
(moved into the benefit zone 2017->2024) improve in rainfall-adjusted productivity
relative to stable-open controls?

Change classes (central NSW, from NSW per-year tree maps + 2020 cropland):
  1 = stable open    (>250 m from trees in both 2017 and 2024)
  2 = gained shelter (>250 m in 2017, now 80-200 m = benefit zone in 2024)
Samples annual peak EVI + growing-season rainfall for early (2017-18) & late (2023-24).
Difference-in-differences, rainfall-adjusted (RESTREND). Tiled to fit the GEE limit.

  python scripts/restrend_beforeafter.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import PROJECT
from shelter_sample import evi_season
DATA = ROOT / "data"
ROOT_A = "projects/ee-christopher-bradley/assets"
REGION = [146.0, -35.0, 149.5, -31.5]
EARLY, LATE = [2017, 2018], [2023, 2024]


def dist(img):
    return img.select(0).gt(50).distance(ee.Kernel.euclidean(400, "meters"), False).unmask(400)


def rain(year):
    return (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterDate(f"{year}-04-01", f"{year}-11-30").select("pr").sum().rename(f"rain_{year}"))


def build_stack():
    d17 = dist(ee.Image(f"{ROOT_A}/NSW_2017_predicted_s4_4326"))
    d24 = dist(ee.Image(f"{ROOT_A}/NSW_2024_predicted_s4_4326"))
    crop = ee.ImageCollection(f"{ROOT_A}/Aus_ag2020_default-percentmethod").mosaic().select("b1").remap([41, 42], [1, 1]).eq(1)
    stable = d17.gt(250).And(d24.gt(250)).And(crop)
    gained = d17.gt(250).And(d24.gte(80)).And(d24.lt(200)).And(crop)
    cls = ee.Image(0).where(stable, 1).where(gained, 2).selfMask().rename("chg")
    bands = [cls]
    for y in EARLY + LATE:
        bands += [evi_season(y), rain(y)]
    return ee.Image.cat(bands)


def main():
    ee.Initialize(project=PROJECT)
    stack = build_stack()
    out = DATA / "restrend_samples.csv"
    if out.exists():
        out.unlink()
    lon0, lat0, lon1, lat1 = REGION
    step = 0.5
    tiles = [(lon0 + step * (i + 0.5), lat0 + step * (j + 0.5))
             for i in range(int((lon1 - lon0) / step)) for j in range(int((lat1 - lat0) / step))]
    print(f"{len(tiles)} tiles")
    for k, (lo, la) in enumerate(tiles):
        reg = ee.Geometry.Rectangle([lo - step / 2, la - step / 2, lo + step / 2, la + step / 2])
        try:
            s = stack.stratifiedSample(numPoints=400, classBand="chg", region=reg, scale=20,
                                       seed=1, dropNulls=True, tileScale=4)
            df = pd.read_csv(io.BytesIO(requests.get(s.getDownloadURL(filetype="CSV"), timeout=290).content))
            if len(df):
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"  [{k+1}/{len(tiles)}] +{len(df)}")
        except Exception as e:
            print(f"  [{k+1}/{len(tiles)}] skip ({str(e)[:40]})")
        time.sleep(0.3)

    # ---- analysis ----
    df = pd.read_csv(out)
    df = df[df.chg.isin([1, 2])]
    for grp, yrs in [("evi_early", EARLY), ("evi_late", LATE), ("rain_early", EARLY), ("rain_late", LATE)]:
        base = "evi" if "evi" in grp else "rain"
        df[grp] = df[[f"{base}_{y}" for y in yrs]].mean(axis=1)
    df = df[df.evi_early.between(-0.2, 1) & df.evi_late.between(-0.2, 1)].dropna(
        subset=["evi_early", "evi_late", "rain_early", "rain_late"])
    df["d_evi"] = df.evi_late - df.evi_early
    df["d_rain"] = df.rain_late - df.rain_early
    # RESTREND: remove rainfall-change effect from EVI change, then DiD
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(df[["d_rain"]], df["d_evi"])
    df["d_evi_adj"] = df["d_evi"] - lr.predict(df[["d_rain"]]) + df["d_evi"].mean()
    gained = df[df.chg == 2]["d_evi_adj"]; stable = df[df.chg == 1]["d_evi_adj"]
    did = gained.mean() - stable.mean()
    from scipy.stats import ttest_ind
    t, p = ttest_ind(gained, stable, equal_var=False)
    print(f"\nRESTREND DiD (gained-shelter vs stable-open, rainfall-adjusted EVI change):")
    print(f"  gained n={len(gained)} ΔEVI={gained.mean():+.4f} | stable n={len(stable)} ΔEVI={stable.mean():+.4f}")
    print(f"  DiD effect = {did:+.4f} EVI (~{did*6.4:+.3f} t/ha wheat), t={t:.2f}, p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["stable open", "gained shelter"], [stable.mean(), gained.mean()],
           yerr=[stable.sem(), gained.sem()], capsize=5, color=["grey", "tab:green"])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("rainfall-adjusted ΔEVI (late−early)")
    ax.set_title(f"RESTREND before/after: shelter gain effect\nDiD = {did:+.4f} EVI ({did*6.4:+.3f} t/ha), p={p:.3f}")
    fig.tight_layout(); fig.savefig(ROOT / "outputs" / "shelter" / "12_restrend_beforeafter.png", dpi=130)
    print("saved 12_restrend_beforeafter.png")


if __name__ == "__main__":
    main()
