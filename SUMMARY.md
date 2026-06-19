# Overnight crop-mapping run — morning summary

## What works now

A full **crop-type mapping pipeline**: Sentinel-2 winter-season monthly composites
(cloud-masked, + NDVI/NDWI/NDRE phenology) → classifier → wall-to-wall crop GeoTIFFs.
Labels come from existing crop layers — **NLUM v7** for Australia, **USDA CDL** for a
US sanity check. No street-level imagery / manual labelling needed.

## Models trained (saved in `outputs/models/`)

| Model | Train regions | Classes | Pooled spatial OA | Held-out region OA |
|---|---|---|---|---|
| **`au_national_v2.joblib`** ⭐ recommended | 7 areas (NSW/VIC/WA/SA) | 7 | 0.831 | **Ardlethan NSW 0.74, Kojonup WA 0.73** (best generalist) |
| `au_national.joblib` | 5 areas (NSW/VIC/WA/SA) | 7 | 0.842 (κ 0.71) | Ardlethan 0.73, Kojonup WA 0.73 |
| `au_multi.joblib` | Temora + Wimmera + Corowa | 5 | 0.859 (κ 0.73) | Ardlethan 0.72 |
| (US, CDL) | North Dakota | 9 | 0.88 | 0.85 |

**Use `au_national_v2.joblib`** — it generalizes best to unseen regions and handles more
classes (hay F1 0.45 vs 0.31). `au_national` has marginally higher *pooled* OA but that
metric rewards memorizing the training regions; held-out generalization is what matters
for mapping new areas.

"spatial OA" = accuracy on spatially **held-out blocks** (honest, no leakage). The
held-out-region numbers are the real generalization test — note the **WA region (Kojonup)
predicted at 0.73 despite most training being in the eastern states** → the model
generalizes across the continent for the dominant classes.

Per-class (national, spatial holdout): Winter cereals F1 **0.89**, Grazing **0.81**,
Horticulture 0.73; Canola 0.51, Legumes 0.57, Hay 0.31 (rarer → weaker, need more samples).

## Run it / apply anywhere in Australia

```bash
conda activate veg_species_mapper
# predict ANY AOI with a saved model (no retraining), with smoothing:
python scripts/crop_predict.py --model outputs/models/au_national.joblib \
    --bbox <minlon> <minlat> <maxlon> <maxlat> --name myaoi --smooth 5
# retrain / extend:
python scripts/crop_train_au_multi.py --national --model hgb --indices --tag au_national
```

## Key outputs to look at (`outputs/`)

- `predict_kojonup_wa_smooth_prediction_smooth.{tif,png}` — clean held-out **WA** crop map.
- `au_national_<region>_prediction.tif` / `_labels.tif` — prediction vs NLUM for each region
  (train regions + held-out Ardlethan_NSW & Kojonup_WA).
- `au_national_confusion.png`, `au_national_training_sources.png` (training distribution by region).
- `*_confidence.tif` — per-pixel max-probability (model confidence).
- `RESULTS_AU.md` / `accuracy_*.json` — metrics.

## Honest limitations (for the writeup)

1. **Accuracy is agreement with NLUM**, which is itself a model — not field truth.
   Independent validation needs ground data (the NVT trial points — kept NDA-safe, never
   committed — are a candidate).
2. **NLUM can't separate wheat vs barley** (lumped "Winter cereals").
3. **Rare classes** (hay, legumes, modified pasture) are under-sampled → low F1; fix by
   adding regions where they're common.
4. **Fields of The World has no Australian coverage** — object/paddock-based labelling
   needs ePaddocks or running FTW's segmentation model on AU Sentinel-2.
5. Per-pixel model → use `--smooth` (modal filter) for field-coherent maps.

## Sensible next steps

- Add more training regions for national coverage + boost rare classes.
- Independent validation against NVT trial points (yield/variety) once cleared.
- Move to object-based (paddock) labels for the shelter-effect analysis.
- Try a temporal deep model (e.g., per-pixel LSTM/transformer on the S2 time series).
