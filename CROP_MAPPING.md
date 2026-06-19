# Crop-type mapping from Sentinel-2 + NLUM (Australia) / CDL (US)

Train a classifier on **Sentinel-2 monthly composites** and predict wall-to-wall crop
type. Labels come from an existing crop layer — **NLUM v7** probability surfaces for
Australia, **USDA CDL** for the US validation. No street-level imagery or manual
labelling needed: the existing crop layer *is* the label source.

```
Sentinel-2 (Planetary Computer)  ->  monthly median composites (cloud-masked, SCL)
   + per-month NDVI / NDWI / NDRE (phenology)                         [features]
NLUM v7 commodity probabilities  ->  argmax + confidence per 250 m cell [labels]
   keep high-confidence cells (>0.7) as training labels
classifier (HistGradientBoosting / RandomForest)
   -> spatial-block holdout accuracy + held-out region
   -> GeoTIFFs (labels / prediction / confidence) + maps
```

## Why this design

- **Phenology is the signal.** A May–Nov composite stack lets the model separate
  cereals / canola / legumes / pasture by their seasonal trajectory, not just one date.
- **Confidence-thresholded labels.** NLUM is a *probability* surface; we only train on
  cells where one commodity clearly dominates (>0.7), giving clean labels.
- **Honest accuracy.** Random pixel splits leak (neighbouring pixels are correlated),
  so we report **spatial-block holdout** and a **completely held-out region**. The naive
  random-split number is logged too, to show the gap.

## Run

```bash
conda activate veg_species_mapper
# Australia, multi-region (covers cereals/canola/legumes/hay/pasture):
python scripts/crop_train_au_multi.py --model hgb --indices --conf 0.7 --tag au_multi
# US validation (USDA CDL, North Dakota):
python scripts/crop_train.py --model rf --tag rf_base
```

Outputs land in `outputs/`: `*_labels.tif`, `*_prediction.tif`, `*_confidence.tif`
(categorical GeoTIFFs with colour tables — open in QGIS), `*_prediction.png` /
`*_labels.png` previews, `*_confusion.png`, `*_training_sources.png`, and
`accuracy_*.json` + `RESULTS*.md`.

## Results

See `outputs/RESULTS.md` (US) and `outputs/RESULTS_AU.md` (Australia) for the running
table. Headline so far:

| Run | Labels | Region | Spatial OA | Unseen-region OA |
|---|---|---|---|---|
| US baseline (RF) | USDA CDL | North Dakota | 0.88 | 0.85 |
| AU v1 (HGB) | NLUM v7 | Temora NSW | 0.87 | — |

## Important caveats (for the writeup)

- **NLUM lumps wheat+barley+oats** into "Winter cereals" — wheat vs barley is *not*
  separable from NLUM. Splitting them needs an independent label source (e.g. the NVT
  trial points) or finer phenology + region priors.
- **Accuracy is agreement with NLUM**, which is itself a model — not field truth.
  Independent validation would use ground observations (NVT trials, etc.).
- **Fields of The World has no Australian coverage** (24 countries, not incl. AU), so
  the object-based "label whole paddocks" step needs a different boundary source
  (ePaddocks, or running FTW's segmentation model on AU Sentinel-2).
- 250 m NLUM labels on a 10 m S2 grid: a confident 250 m cell is assumed pure; mixed
  cells are excluded by the confidence threshold.

## Reusability

`cropmap/` is region-agnostic. To map a new area: pick an AOI bbox, point at a label
source (NLUM/CDL/your own), and the same feature-building + training + GeoTIFF code runs.
