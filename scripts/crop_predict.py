"""Apply a saved crop-type model to ANY Australian AOI -> GeoTIFF, no retraining.

  python scripts/crop_predict.py \
      --model outputs/models/au_national.joblib \
      --bbox 147.45 -34.50 147.62 -34.35 --name riverina_test

Outputs outputs/predict_<name>_prediction.tif (+ _confidence.tif, _prediction.png).
The model bundle carries the year/months/features it was trained on, so prediction
uses identical features automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from veg_species_mapper.cropmap import pipeline

OUT = ROOT / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to saved .joblib model bundle")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    ap.add_argument("--name", default="aoi")
    ap.add_argument("--no-confidence", action="store_true")
    args = ap.parse_args()

    bundle = pipeline.load_model(args.model)
    meta = bundle["meta"]
    print(f"Loaded model '{meta.get('tag')}' trained on {meta.get('train_regions')} "
          f"(year {meta['year']}, months {meta['months']}, indices={meta['use_indices']})")
    print(f"Predicting AOI {tuple(args.bbox)} -> {args.name}")
    pred, proba, _ = pipeline.predict_aoi(
        bundle, tuple(args.bbox), OUT / f"predict_{args.name}",
        write_confidence=not args.no_confidence)
    import numpy as np
    from veg_species_mapper.cropmap.legend_au import NAME_BY_ID
    vals, counts = np.unique(pred[pred != 255], return_counts=True)
    print("Predicted class pixel counts:")
    for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
        print(f"  {NAME_BY_ID.get(v, v):20s} {c}")
    print(f"Wrote outputs/predict_{args.name}_prediction.tif")


if __name__ == "__main__":
    main()
