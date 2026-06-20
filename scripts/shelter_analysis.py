"""Analyse the shelter-effect sample: distance-decay, confounder-adjusted effect,
drought interaction, heterogeneous (causal-forest-style) effects. Robust to partial
data; regenerates all plots to outputs/shelter/.

  python scripts/shelter_analysis.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "shelter"
OUT.mkdir(parents=True, exist_ok=True)
DROUGHT, WET = 2019, 2021
CONF = ["rain", "tmax", "pdsi", "cwd", "elev", "slope", "aspect", "clay", "soc", "ph", "bd"]


def load():
    files = glob.glob(str(ROOT / "data" / "shelter_samples_tiled_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "tile" not in df:
        df["tile"] = "t0"
    df = df[df["cls"].isin([1, 2, 3, 4])].copy()
    df["cover"] = np.where(df["cls"].isin([1, 2]), "crop", "pasture")
    df["sheltered"] = df["cls"].isin([2, 4]).astype(int)
    years = sorted({int(c.split("_")[1]) for c in df.columns if c.startswith("evi_")})
    for y in years:
        df = df[df[f"evi_{y}"].between(-0.5, 1.2)]
    return df.dropna(subset=[f"evi_{y}" for y in years] + ["dist_tree"]), years


def tile_demean(df, col):
    return df[col] - df.groupby("tile")[col].transform("mean")


def fig_distance_decay(df, years):
    """Tile-demeaned EVI vs distance-to-tree, crop & pasture, drought vs wet."""
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 400]
    mids = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cover in zip(axes, ["crop", "pasture"]):
        d = df[df.cover == cover].copy()
        for y, c in [(DROUGHT, "tab:red"), (WET, "tab:blue")]:
            if f"evi_{y}" not in d:
                continue
            d["dem"] = tile_demean(d, f"evi_{y}")
            d["db"] = pd.cut(d["dist_tree"], bins, labels=mids)
            grp = d.groupby("db", observed=True)["dem"]
            m, se = grp.mean(), grp.sem()
            ax.errorbar(mids[:len(m)], m.values, yerr=se.values, marker="o",
                        color=c, label=f"{y} ({'drought' if y==DROUGHT else 'wet'})", capsize=3)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{cover.title()}: productivity vs distance to trees")
        ax.set_xlabel("distance to nearest tree (m)"); ax.legend()
    axes[0].set_ylabel("tile-demeaned peak EVI (Δ from local mean)")
    fig.suptitle("Shelter distance-decay curve (within-tile matched)", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "01_distance_decay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_naive_vs_adjusted(df, years):
    """Sheltered-unsheltered ΔEVI: naive vs within-tile vs covariate-adjusted."""
    from sklearn.linear_model import LinearRegression
    rows = []
    for cover in ["crop", "pasture"]:
        for y in years:
            d = df[(df.cover == cover)].dropna(subset=CONF_COLS(y) + [f"evi_{y}"]).copy()
            if d.sheltered.nunique() < 2 or len(d) < 30:
                continue
            naive = d[d.sheltered == 1][f"evi_{y}"].mean() - d[d.sheltered == 0][f"evi_{y}"].mean()
            # within-tile FE: demean y and treatment by tile
            yd = tile_demean(d, f"evi_{y}"); td = d["sheltered"] - d.groupby("tile")["sheltered"].transform("mean")
            fe = np.polyfit(td, yd, 1)[0] if td.std() > 0 else np.nan
            # covariate-adjusted (FE + confounders)
            X = np.column_stack([td] + [tile_demean(d, c) for c in CONF_COLS(y)])
            adj = LinearRegression().fit(X, yd).coef_[0]
            rows.append((cover, y, naive, fe, adj))
    res = pd.DataFrame(rows, columns=["cover", "year", "naive", "within_tile", "adjusted"])
    res.to_csv(OUT / "shelter_effect_table.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, cover in zip(axes, ["crop", "pasture"]):
        r = res[res.cover == cover]
        x = np.arange(len(r)); w = 0.27
        ax.bar(x - w, r["naive"], w, label="naive")
        ax.bar(x, r["within_tile"], w, label="within-tile")
        ax.bar(x + w, r["adjusted"], w, label="+confounders")
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(r["year"])
        ax.set_title(f"{cover.title()} shelter effect (ΔEVI)"); ax.legend()
    axes[0].set_ylabel("sheltered − unsheltered EVI")
    fig.suptitle("Naive vs matched vs confounder-adjusted shelter effect", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "02_naive_vs_adjusted.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return res


def CONF_COLS(y):
    return [f"rain_{y}", f"tmax_{y}", f"pdsi_{y}", f"cwd_{y}", "elev", "slope", "aspect",
            "clay", "soc", "ph", "bd"]


def fig_drought_interaction(res):
    """Shelter effect: drought vs wet, per cover."""
    fig, ax = plt.subplots(figsize=(7, 5))
    piv = res.pivot_table(index="cover", columns="year", values="adjusted")
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("adjusted shelter ΔEVI")
    ax.set_title("Shelter effect by year (drought 2019 vs wet 2021)")
    fig.tight_layout(); fig.savefig(OUT / "03_drought_interaction.png", dpi=130); plt.close(fig)


def fig_causal_forest(df, years):
    """T-learner CATE of shelter (controlling for X) + heterogeneity vs rainfall."""
    from sklearn.ensemble import RandomForestRegressor
    y = DROUGHT if f"evi_{DROUGHT}" in df else years[0]
    d = df.dropna(subset=CONF_COLS(y) + [f"evi_{y}"]).copy()
    feats = ["sheltered", "dist_tree"] + CONF_COLS(y)
    X = d[feats].values; Y = d[f"evi_{y}"].values
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=0).fit(X, Y)
    # CATE = pred(sheltered=1) - pred(sheltered=0)
    Xs = d[feats].copy(); Xs["sheltered"] = 1
    Xu = d[feats].copy(); Xu["sheltered"] = 0
    cate = rf.predict(Xs.values) - rf.predict(Xu.values)
    d["cate"] = cate
    # importance
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    imp.plot(kind="barh", ax=axes[0]); axes[0].set_title(f"RF feature importance (EVI {y})")
    # CATE vs rainfall
    d["rb"] = pd.qcut(d[f"rain_{y}"], 6, duplicates="drop")
    g = d.groupby("rb", observed=True)["cate"].mean()
    axes[1].plot([iv.mid for iv in g.index], g.values, "o-")
    axes[1].axhline(0, color="grey", ls="--"); axes[1].set_xlabel(f"growing-season rainfall {y} (mm)")
    axes[1].set_ylabel("estimated shelter effect (CATE, ΔEVI)")
    axes[1].set_title("Does shelter help more where it's drier?")
    fig.tight_layout(); fig.savefig(OUT / "04_causal_forest.png", dpi=130); plt.close(fig)
    return float(np.mean(cate))


def main():
    df, years = load()
    print(f"loaded {len(df)} samples, {df['tile'].nunique()} tiles, years {years}")
    print("cover x sheltered counts:\n", pd.crosstab(df.cover, df.sheltered))
    fig_distance_decay(df, years)
    res = fig_naive_vs_adjusted(df, years)
    fig_drought_interaction(res)
    mean_cate = fig_causal_forest(df, years)
    print("\nAdjusted shelter effect (ΔEVI):")
    print(res.to_string(index=False))
    print(f"\nMean CATE (drought year, RF T-learner): {mean_cate:+.4f}")
    print(f"plots -> {OUT}")


if __name__ == "__main__":
    main()
