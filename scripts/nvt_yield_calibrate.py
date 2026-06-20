"""Calibrate peak EVI to crop yield using NVT trial sites, so the shelter EVI effect can
be expressed in t/ha. Samples each NVT site's peak EVI (its trial year) on GEE and
regresses site yield against EVI, per crop.

  python scripts/nvt_yield_calibrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ee_canola import PROJECT
from shelter_sample import evi_season
NVT = ROOT / "data" / "sensitive_data" / "trials.gpkg"
OUT = ROOT / "outputs" / "shelter"


def main():
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file(NVT)
    g = g.dropna(subset=["Trial GPS Lat", "Trial GPS Long", "Single Site Yield"])
    g = g[g["Crop.Name"].isin(["Wheat", "Canola", "Barley"])]
    # site-level mean yield
    sites = (g.groupby(["TrialCode", "Year", "Crop.Name", "Trial GPS Lat", "Trial GPS Long"])
             ["Single Site Yield"].mean().reset_index()
             .rename(columns={"Trial GPS Lat": "lat", "Trial GPS Long": "lon",
                              "Single Site Yield": "yield", "Crop.Name": "crop"}))
    sites = sites[(sites["yield"] > 0) & (sites["yield"] < 12)]
    rows = []
    for year in sorted(sites["Year"].unique()):
        sy = sites[sites["Year"] == year]
        if year < 2017 or len(sy) == 0:
            continue
        evi = evi_season(int(year))
        feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]).buffer(70),
                            {"i": int(idx), "crop": r.crop, "yld": float(r["yield"])})
                 for idx, r in sy.iterrows()]
        fc = ee.FeatureCollection(feats)
        try:
            s = evi.reduceRegions(fc, ee.Reducer.percentile([80]), 10).getInfo()
        except Exception as e:
            print(f"  {year}: skip ({str(e)[:50]})"); continue
        for f in s["features"]:
            p = f["properties"]
            ev = p.get("p80")
            if ev is not None:
                rows.append((int(year), p["crop"], p["yld"], ev))
        print(f"  {int(year)}: {len(sy)} sites sampled")
    df = pd.DataFrame(rows, columns=["year", "crop", "yield", "evi"]).dropna()
    df = df[df["evi"].between(0, 1)]
    df.to_csv(ROOT / "data" / "nvt_evi_yield.csv", index=False)

    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    print("\nEVI->yield calibration:")
    for ax, crop in zip(axes, ["Wheat", "Canola", "Barley"]):
        d = df[df.crop == crop]
        if len(d) < 20:
            ax.set_title(f"{crop} (n={len(d)})"); continue
        X = d[["evi"]].values; Y = d["yield"].values
        lr = LinearRegression().fit(X, Y)
        r, _ = pearsonr(d["evi"], d["yield"])
        ax.scatter(d["evi"], d["yield"], s=8, alpha=0.4)
        xs = np.linspace(d["evi"].min(), d["evi"].max(), 50)
        ax.plot(xs, lr.predict(xs.reshape(-1, 1)), "r-")
        ax.set_title(f"{crop}: yield={lr.coef_[0]:.1f}·EVI{lr.intercept_:+.1f}\n"
                     f"r={r:.2f}, n={len(d)}, slope={lr.coef_[0]:.1f} t/ha per EVI")
        ax.set_xlabel("peak EVI"); ax.set_ylabel("yield (t/ha)")
        print(f"  {crop}: slope={lr.coef_[0]:.2f} t/ha/EVI, r={r:.2f}, n={len(d)}; "
              f"shelter +0.012 EVI -> +{lr.coef_[0]*0.012:.3f} t/ha")
    fig.suptitle("NVT yield vs peak EVI calibration", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "05_nvt_yield_calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {OUT}/05_nvt_yield_calibration.png")


if __name__ == "__main__":
    main()
