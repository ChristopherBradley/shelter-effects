# Tree-species mapping (ACT) — pipeline + status

Goal: distinguish **eucalypt / conifer-pine / deciduous-exotic / casuarina** from
Sentinel-2, to attribute species to mapped shelterbelts.

## Labels (built, no Sentinel-2 needed)

- **ACT Tree Assets** (Socrata `9qch-rvqr`, 786k trees) → genus → tree-group class,
  rasterised to a 10 m grid keeping pixels with ≥3 trees of the majority group.
- **NLUM v7 plantation-forestry** probability surface → confident cells as Conifer/Pine.

`scripts/tree_train.py --labels-only` builds `outputs/tree_act_labels.tif/.png`.
Module: `src/veg_species_mapper/treespecies.py`. Full model (Sentinel-2 **full-year**
composite, so deciduous leaf-off separates exotics from evergreens) runs with
`scripts/tree_train.py --model hgb` once Planetary Computer is back up.

## Status & honest limitations (visible in the label map)

1. **Urban-tree sparsity.** Street trees rarely fill a 10 m pixel, so only ~8k pixels
   reach the ≥3-tree threshold, and they **trace the street grid** rather than forming
   contiguous stands. Trainable (Eucalypt 4.6k px, Deciduous 2.3k px) but the S2 signal
   at these pixels is partly road/roof — expect more confusion than the crop model.
2. **Pine is under-represented here** (224 px): ACT pine plantations are *rural* (not in
   the urban inventory), and this AOI only clips the edge of Stromlo (much of which was
   burnt in 2003 and not replanted). Better pine labels: a larger AOI over **Kowen
   Forest**, or the national **Forests of Australia** softwood-plantation layer.
3. **Urban/rural confound.** Eucalypt labels are urban, pine labels rural — a classifier
   could partly learn context rather than species. For shelterbelt attribution (rural),
   the cleaner design is: eucalypt labels from a native forest-type map, pine from the
   plantation layer — both in rural settings.

## Recommended next step (when PC is up)
- Run the full model on this AOI to get a baseline, then iterate the AOI/label sources
  toward rural eucalypt-woodland + plantation pine for a shelterbelt-relevant model.
- Trees benefit more than crops from **aerial 1 m** imagery (narrow crowns) — a fusion
  of S2 phenology + aerial texture is the likely best path, per the earlier discussion.
