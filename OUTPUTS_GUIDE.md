# Understanding the inputs and outputs

A practical guide to what went into the crop models and how to read what came out.

## Inputs

**1. Sentinel-2 imagery (the features).** For each area, the pipeline pulls Sentinel-2
surface-reflectance scenes from Microsoft Planetary Computer over the winter growing
season (May–Nov 2020), cloud-masks them (SCL band), and builds a **monthly median
composite** per band. It then adds per-month **NDVI / NDWI / NDRE** vegetation indices.
So each pixel becomes a ~91-number "fingerprint" of how its reflectance changes through
the season — which is what separates crops (canola flowers yellow in spring, cereals
senesce by summer, pasture stays green, etc.). Composites are cached in `data/cache/`.

**2. NLUM v7 (the labels).** `data/public_data/NLUM_v7_.../` holds one 250 m probability
surface per commodity (winter cereals, canola, legumes, hay, grazing, …). For each cell
we take the **highest-probability commodity** and keep only **confident cells (>0.7)** as
training labels. So the model learns "this seasonal S2 fingerprint = canola" from places
NLUM is sure about, then predicts everywhere.

Think of it as: *NLUM tells us what's there at 250 m where it's confident; Sentinel-2
tells us the fine-grained 10 m detail; the model learns to go from S2 → crop type.*

## Outputs (in `outputs/`)

For each region `<R>` and model run `<tag>`:

| File | What it is | How to read it |
|---|---|---|
| `<tag>_<R>_prediction.tif` | Predicted crop class per 10 m pixel | Open in QGIS — it has a built-in colour table (cereals tan, canola yellow, legumes purple, grazing green, …) |
| `<tag>_<R>_labels.tif` | The NLUM "truth" the model was scored against | Compare side-by-side with prediction |
| `<tag>_<R>_confidence.tif` | Model's max class probability (0–1) | Bright = confident, dark = uncertain (often field edges/mixed pixels) |
| `predict_<name>_prediction_smooth.tif` | Prediction after a 5-px majority filter | The clean, field-coherent version — use this for nice maps |
| `<tag>_confusion.png` | Which classes get confused, on held-out data | Rows = truth, cols = predicted; diagonal = correct |
| `<tag>_training_sources.png` | Where training samples came from, by class & region | Shows the spatial spread / class balance of training data |
| `accuracy_<tag>.json`, `RESULTS_AU.md` | The numbers | See "accuracy" below |

### How to open the GeoTIFFs
```bash
conda activate veg_species_mapper
# quick look from the terminal:
gdalinfo -stats outputs/predict_kojonup_wa_smooth_prediction_smooth.tif
```
Or just drag the `.tif` into **QGIS** — the class colours and georeferencing are embedded,
so it overlays correctly on a basemap. The `.png` files are quick previews (no GIS needed).

### Reading the accuracy
- **spatial OA** (overall accuracy on spatially **held-out blocks**) is the honest number —
  it doesn't let the model "cheat" by memorising neighbouring pixels of the same field.
- **unseen-region OA** is the real generalization test: train on some regions, predict a
  region the model never saw. The national model holds ~0.73 on a held-out **WA** region.
- **naive OA** (random pixel split) is logged too, and is optimistically high — the gap
  between it and spatial OA shows how much spatial autocorrelation inflates naive scores.
- Per-class **F1** tells you which crops are reliable (cereals .89, grazing .81) vs shaky
  (hay, legumes — rarer, fewer training samples).

## Apply a model to a new area yourself
```bash
python scripts/crop_predict.py \
  --model outputs/models/au_national_v2.joblib \
  --bbox <minlon> <minlat> <maxlon> <maxlat> --name myplace --smooth 5
```
This downloads S2 for the bbox, builds the same features, and writes
`outputs/predict_myplace_prediction_smooth.tif`. No retraining needed — the model file
carries the recipe (year, months, features) with it.

## The one big caveat
The accuracy numbers are **agreement with NLUM**, which is itself a model — not
independent ground truth. A prediction can be "wrong" vs NLUM but right on the ground, or
vice-versa. Genuine validation needs field data (e.g. the NVT trial points).
