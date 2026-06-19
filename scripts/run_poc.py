"""End-to-end proof-of-concept for the two hard parts + triangulation.

  1. Fetch nearby Mapillary 360 panoramas around a target coordinate (with pose).
  2. Crop perspective views sweeping around each panorama.
  3. Auto-label each crop with Pl@ntNet; find crops matching the target species.
  4. From two panoramas, triangulate the ground location of the detected plant.

Run:
  pip install -r requirements.txt
  export MAPILLARY_TOKEN=...   PLANTNET_API_KEY=...   (or use a .env file)
  python scripts/run_poc.py --lat -35.2885 --lon 149.0742 --species "Pinus radiata"

Add --sentinel to also pull the Sentinel-2 signature at the triangulated point
(needs requirements-sentinel.txt installed).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make src/ importable without installing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Tiny .env loader (avoids a python-dotenv dependency).
_envfile = ROOT / ".env"
if _envfile.exists():
    for line in _envfile.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from veg_species_mapper import mapillary, panorama, species_id, triangulate
from PIL import Image


def label_panorama(img: mapillary.MapillaryImage, pano_path: Path, species: str,
                   n_views: int, crops_dir: Path):
    """Crop a sweep of views, ID each, return (best_bearing, best_result) for the species."""
    pano = Image.open(pano_path)
    best = None  # (abs_bearing, IdResult, crop_path)
    for yaw in panorama.sweep_yaws(n_views):
        crop = panorama.crop_perspective(pano, yaw_deg=yaw)
        crop_path = crops_dir / f"{img.id}_yaw{int(yaw):03d}.jpg"
        crop.save(crop_path, quality=85)
        results = species_id.identify(crop_path)
        match = species_id.best_match_for(results, species)
        top = results[0] if results else None
        print(f"    yaw {yaw:5.1f}deg -> "
              f"{'TOP ' + str(top) if top else 'no id'}"
              f"{'  << MATCH ' + f'{match.score:.2f}' if match else ''}")
        if match and (best is None or match.score > best[1].score):
            abs_bearing = (img.compass_angle + yaw) % 360.0
            best = (abs_bearing, match, crop_path)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--species", default="Pinus radiata",
                    help="genus or 'Genus species' to look for, e.g. 'Eragrostis' for lovegrass")
    ap.add_argument("--radius", type=float, default=30.0, help="search radius (m)")
    ap.add_argument("--panos", type=int, default=2, help="how many nearby panoramas to use")
    ap.add_argument("--views", type=int, default=12, help="perspective crops per panorama")
    ap.add_argument("--sentinel", action="store_true", help="also pull Sentinel-2 signature")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    data = Path(args.data)
    pano_dir = data / "panos"
    crops_dir = data / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Searching Mapillary within {args.radius:.0f} m of ({args.lat}, {args.lon}) ...")
    imgs = mapillary.search_images(args.lat, args.lon, radius_m=args.radius, panoramic_only=True)
    if not imgs:
        print("  No panoramic imagery here. Try a larger --radius or a road with Mapillary 360 coverage.")
        print("  Check coverage visually: https://www.mapillary.com/app/")
        return
    print(f"  Found {len(imgs)} panoramas; using nearest {args.panos}.")

    detections: list[triangulate.Bearing] = []
    for img in imgs[: args.panos]:
        d = mapillary.haversine_m(args.lat, args.lon, img.lat, img.lon)
        print(f"\n[2-3/4] Panorama {img.id} ({d:.0f} m away, compass {img.compass_angle:.0f}deg)")
        pano_path = mapillary.download(img, pano_dir)
        best = label_panorama(img, pano_path, args.species, args.views, crops_dir)
        if best:
            abs_bearing, match, crop_path = best
            print(f"  -> '{args.species}' detected at bearing {abs_bearing:.1f}deg "
                  f"(score {match.score:.2f}, {crop_path.name})")
            detections.append(triangulate.Bearing(img.lat, img.lon, abs_bearing))
        else:
            print(f"  -> no '{args.species}' detected in this panorama.")

    if len(detections) < 2:
        print(f"\n[4/4] Need 2 detections to triangulate; got {len(detections)}. "
              "Increase --radius/--panos/--views or pick a spot with denser coverage.")
        return

    fix = triangulate.triangulate(detections[0], detections[1])
    print(f"\n[4/4] Triangulated location: ok={fix.ok}  "
          f"lat={fix.lat:.6f} lon={fix.lon:.6f}  "
          f"(ranges {fix.range_a_m:.0f} m / {fix.range_b_m:.0f} m)")
    if not fix.ok:
        print("  Rays diverged -- the two crops probably saw different plants. "
              "More panoramas / finer --views will help.")
        return

    if args.sentinel:
        print("\n[+] Fetching Sentinel-2 signature at the triangulated point ...")
        from veg_species_mapper import sentinel
        sigs = sentinel.point_signature(fix.lat, fix.lon)
        for s in sigs[:3]:
            print(f"  {s.datetime}  cloud {s.cloud_pct:.0f}%  {s.bands}")

    print("\nDone. The two hard parts (imagery + auto-label) plus triangulation are proven if "
          "you see a triangulated location above.")


if __name__ == "__main__":
    main()
