"""Street imagery -> vegetation crop -> Pl@ntNet, vs full-frame baseline.

Tests whether isolating the dominant plant (so it fills the frame) rescues plant-ID
from street-level imagery, which failed on whole frames in the first POC.

  python scripts/street_plantid.py --lat -35.2776 --lon 149.1310 --n 6

Saves crops + vegetation overlays to outputs/street/ and prints, per image, the
full-frame top ID vs the veg-crop top ID with scores.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_envfile = ROOT / ".env"
if _envfile.exists():
    for line in _envfile.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from PIL import Image
from veg_species_mapper import mapillary, species_id, veg_crop

OUT = ROOT / "outputs" / "street"


def top_str(results):
    return str(results[0]) if results else "no id"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=60)
    ap.add_argument("--n", type=int, default=6, help="how many nearby images to test")
    ap.add_argument("--thresh", type=float, default=0.12, help="ExG vegetation threshold")
    ap.add_argument("--min-area", type=float, default=0.02, help="min veg-blob fraction of frame")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pano_dir = ROOT / "data" / "street_panos"

    imgs = mapillary.search_images(args.lat, args.lon, radius_m=args.radius,
                                   panoramic_only=False, limit=200)
    if not imgs:
        print("No Mapillary imagery here."); return
    print(f"Found {len(imgs)} images; testing nearest {args.n}.\n")

    wins = 0
    for i, im in enumerate(imgs[: args.n]):
        path = mapillary.download(im, pano_dir)
        image = Image.open(path)

        full = species_id.identify(path)
        vc = veg_crop.largest_plant_crop(image, thresh=args.thresh, min_area_frac=args.min_area)
        if vc is None:
            print(f"[{im.id}] no vegetation blob found; full-frame: {top_str(full)}")
            continue

        crop_path = OUT / f"{im.id}_crop.jpg"
        vc.image.save(crop_path, quality=90)
        veg_crop.overlay_mask(image, vc.bbox, thresh=args.thresh).save(OUT / f"{im.id}_overlay.jpg", quality=80)
        crop_res = species_id.identify(crop_path)

        f0 = full[0].score if full else 0.0
        c0 = crop_res[0].score if crop_res else 0.0
        better = c0 > f0
        wins += better
        print(f"[{im.id}] veg blob {vc.area_fraction*100:.0f}% of frame, {vc.veg_fraction*100:.0f}% green")
        print(f"    full-frame : {top_str(full)}")
        print(f"    veg-crop   : {top_str(crop_res)}   {'<-- higher confidence' if better else ''}")

    print(f"\nveg-crop beat full-frame on {wins}/{min(args.n, len(imgs))} images. "
          f"Crops/overlays saved to {OUT}")


if __name__ == "__main__":
    main()
