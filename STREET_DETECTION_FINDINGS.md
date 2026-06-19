# Street imagery + vegetation isolation → Pl@ntNet: second attempt

Following the first POC (Pl@ntNet returns noise on whole street frames), we added a
**vegetation-isolation step before Pl@ntNet** so the plant fills the frame, per the idea
that tighter crops help.

## What was built

A torch-free isolator (`src/veg_species_mapper/veg_crop.py`): Excess-Green index
(ExG = 2G−R−B) → threshold → morphological cleanup → **largest connected vegetation blob**
→ padded square crop. Driver: `scripts/street_plantid.py` (Mapillary → crop → Pl@ntNet,
vs full-frame baseline; saves crops + green-tinted overlays to `outputs/street/`).

(We intended a SegFormer/Cityscapes segmenter, but this is an Intel-mac env and PyTorch
has no wheels past 2.2.2, which modern `transformers` rejects — so the dependency-free
colour approach was used instead.)

## Results (ACT, Mapillary)

| Location | Outcome |
|---|---|
| Northbourne Ave (winter, deciduous street trees) | ExG found **almost no vegetation** — bare/leafless trees → no green to segment |
| Yarralumla (leafier) | Vegetation found on all 6 images; veg-crop **beat full-frame on 3/6**, but top scores stayed ~**0.07–0.09** |

## Verdict: isolation helps framing, but doesn't rescue street-view plant-ID

Two hard limits, confirmed visually (`outputs/street/*_overlay.jpg`):

1. **Imaging scale is the real bottleneck, not framing.** Roadside trees occupy only
   3–8% of frame and sit 20–40 m away. Even correctly cropped, Pl@ntNet (trained on
   close-up leaves/bark/flowers) only reaches ~0.08 confidence — far below usable.
2. **Colour segmentation is seasonal.** ExG needs green foliage; ACT winter / deciduous
   street trees are bare, so nothing is segmented. A geometry-based detector would be
   needed for year-round robustness.
3. **360 panoramas spread vegetation out**, so the largest-blob bounding box still
   includes sky/road/houses — a per-tree-crown detector would crop tighter.

## Implication for the methodology

This reinforces the earlier pivot: **street-view → reliable species labels is not viable
with off-the-shelf tools** for distant roadside trees. The isolation step is a real but
insufficient improvement. To make street-view labelling work would need (a) a proper
tree-crown detector (needs a GPU/torch env), (b) imagery where the target is close and
large, and (c) likely a model trained on canopy-scale (not organ-scale) photos.

**The authoritative-label routes remain far stronger**: NLUM for crops (working, ~0.84
OA), the ACT tree inventory + NLUM plantation layer for tree species, and ALA for weeds.
Street-view is best treated as a future augmentation, not a foundation.
