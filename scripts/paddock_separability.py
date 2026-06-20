"""Per-crop separability at PIXEL vs PADDOCK level — the core of the 'which crop is
easiest' exploration.

Aggregating per-pixel predictions to whole paddocks averages out noise, usually lifting
accuracy a lot. This quantifies that lift per crop, so we can pick the crop that maps
most cleanly at the paddock level (the unit that matters for the shelter analysis).

Proxy paddocks = contiguous same-class regions of the NLUM label raster (a stand-in
until real FTW boundaries are dropped in). Run after a crop model has written
<tag>_<region>_prediction.tif and _labels.tif.

  python scripts/paddock_separability.py --tag au_national_ndyi \
      --regions Temora_NSW Wimmera_VIC Corowa_NSW Merredin_WA Clare_SA Cootamundra_NSW Mallee_VIC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from veg_species_mapper.cropmap.legend_au import NAME_BY_ID
OUT = ROOT / "outputs"


def load(tag, region):
    import rioxarray
    pred = rioxarray.open_rasterio(OUT / f"{tag}_{region}_prediction.tif").squeeze().values
    lab = rioxarray.open_rasterio(OUT / f"{tag}_{region}_labels.tif").squeeze().values
    return pred, lab


def paddock_stats(pred, lab, min_px=5):
    """Build proxy paddocks (contiguous same-class NLUM), return per-class pixel and
    paddock correct/total tallies."""
    from scipy.ndimage import label as cc
    classes = [c for c in np.unique(lab) if c not in (0, 255)]
    px = {c: [0, 0] for c in classes}      # [correct, total] pixels
    pad = {c: [0, 0] for c in classes}     # [correct, total] paddocks
    for c in classes:
        # pixel-level for this true class
        m = lab == c
        px[c][1] += int(m.sum()); px[c][0] += int((pred[m] == c).sum())
        # proxy paddocks of true class c
        comp, n = cc(m)
        if n == 0:
            continue
        sizes = np.bincount(comp.ravel())
        for pid in range(1, n + 1):
            if sizes[pid] < min_px:
                continue
            sel = comp == pid
            maj = np.bincount(pred[sel].astype(int), minlength=256).argmax()
            pad[c][1] += 1; pad[c][0] += int(maj == c)
    return px, pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="au_national_ndyi")
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--min-px", type=int, default=5)
    args = ap.parse_args()

    px_tot, pad_tot = {}, {}
    for r in args.regions:
        try:
            pred, lab = load(args.tag, r)
        except Exception as e:
            print(f"  skip {r}: {e}"); continue
        px, pad = paddock_stats(pred, lab, args.min_px)
        for c in px:
            px_tot.setdefault(c, [0, 0]); pad_tot.setdefault(c, [0, 0])
            px_tot[c][0] += px[c][0]; px_tot[c][1] += px[c][1]
            pad_tot[c][0] += pad[c][0]; pad_tot[c][1] += pad[c][1]

    print(f"\nPer-crop accuracy — pixel vs proxy-paddock (tag={args.tag})")
    print(f"{'crop':20s} {'pixel-acc':>10s} {'paddock-acc':>12s} {'n_paddocks':>11s} {'n_pixels':>10s}")
    rows = []
    for c in sorted(px_tot):
        pa = px_tot[c][0] / max(px_tot[c][1], 1)
        da = pad_tot[c][0] / max(pad_tot[c][1], 1)
        rows.append((NAME_BY_ID.get(c, c), pa, da, pad_tot[c][1], px_tot[c][1]))
    for name, pa, da, npad, npx in sorted(rows, key=lambda x: -x[2]):
        print(f"{name:20s} {pa:10.2f} {da:12.2f} {npad:11d} {npx:10d}")
    print("\n(paddock-acc = fraction of proxy paddocks whose pixel-majority prediction "
          "matches the paddock's NLUM class)")


if __name__ == "__main__":
    main()
