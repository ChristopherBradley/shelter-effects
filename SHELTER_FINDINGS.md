# Shelter effects on productivity — findings (prototype)

Prototype analysis for thesis Goal 2 ("estimating current effects of shelterbelts on
crop & pasture production during droughts across Australia"). All compute on Earth Engine
(free, noncommercial); your assets used read-only. Design in `SHELTER_EXPERIMENT.md`.

## Data
**158,191 matched sample points across 144 tiles** spanning the SE Australian wheat–sheep
belt (SA-east/VIC/NSW/sthn QLD) and SW Western Australia. Each point: peak-season EVI
(2019 drought, 2020, 2021 wet), distance-to-nearest-tree, your shelter class (percent &
wind methods), growing-season rainfall/PDSI/temp, terrain, and soil. Sampled equally
across sheltered/unsheltered cropland & pasture, matched within 0.4° tiles.

## Result 1 — the classic shelter curve, recovered empirically *(plot 01)*
Tile-demeaned EVI vs distance to trees shows the textbook shelterbelt pattern:

| Zone | Distance | Crop ΔEVI | Pasture ΔEVI |
|---|---|---|---|
| **Competition** | <25 m | **−0.042** | **−0.022** |
| transition | ~60 m | ≈ 0 | +0.004 |
| **Peak shelter benefit** | ~100–200 m | **+0.006 to +0.009** | **+0.011 to +0.013** |
| decay to open-field | >300 m | +0.001 | ≈ 0 |

So productivity is **suppressed right next to trees** (competition/shade) and **raised a
few tree-heights out**, then returns to open-field levels. This is why a naive
sheltered-vs-unsheltered contrast is *negative* — the "sheltered" class sits ~55–70 m
from trees, squarely in/near the competition zone.

## Result 2 — drought *temporally* increases the relative value of shelter (H2) *(plot 01, 03)*
For **crops, near the trees (25–90 m), the 2019 drought curve sits above the 2021 wet
curve** — i.e. within a location the shelter zone loses *less* (or gains more) in a
drought year than in a wet year. The adjusted binary shelter effect is also **least
negative in the drought year** (crop −0.009 in 2019 vs −0.020 in 2020). Consistent with
shelter buffering moisture/heat stress when it matters most.

## Result 3 — shelter benefit is *spatially* larger in wetter regions, not drier (H3) *(plot 06)*
Across the rainfall gradient, the peak shelter benefit is **~0 in the driest tiles
(<150 mm growing-season rain) and rises to ~+0.07 EVI in wetter tiles (>300 mm)**. The
naive "shelter helps most where driest" does **not** hold spatially — arid areas simply
have little productivity to enhance. (Note this is the opposite axis to Result 2: *where*
vs *when*.) Worth foregrounding in the thesis as a genuine, non-obvious distinction.

## Result 4 — in yield units, via NVT calibration *(plot 05)*
Peak EVI calibrates to NVT site yields with r ≈ 0.6–0.7 (wheat 6.4, barley 6.3, canola
3.5 t/ha per EVI unit). Translating the curve:
- **Peak shelter benefit ≈ +0.08 t/ha** for wheat/barley (~+3% on a 2.5 t/ha crop),
  +0.04 t/ha canola.
- **Competition cost ≈ −0.27 t/ha** in the first ~25 m next to trees.
So the net paddock effect depends on the **area-weighted balance** of the narrow penalty
strip vs the broad benefit zone — which favours shelter unless belts are very dense.

## Honest caveats
- Observational: within-tile matching + confounders (soil/terrain/climate) reduce but
  don't remove land-selection bias (trees often left on poorer land). The
  distance-curve + tile-matching is the strongest available identification here.
- EVI peak is a productivity proxy; the yield numbers inherit the NVT calibration scatter.
- Shelter map is 2020; applied across years assuming belts are persistent (reasonable).
- The competition-vs-benefit balance needs **area weighting** per real paddock geometry to
  give a true per-field number (next step).

## Outputs to review
**Plots** (`outputs/shelter/`): 01 distance-decay (headline), 02 naive-vs-adjusted,
03 drought interaction, 04 causal-forest CATE + importance, 05 NVT yield calibration,
06 aridity gradient (H3), 07 sample map. **GeoTIFFs** (`outputs/shelter_geotifs/`,
Riverina showcase): EVI 2019/2021, drought anomaly, shelter class, distance-to-tree,
tree cover, rainfall, PDSI, shelter-benefit.

## Recommended next steps
1. **Area-weight** the penalty vs benefit per real paddock (FTW/your boundaries) → a true
   per-field net effect, and an optimal belt-spacing recommendation.
2. **RESTREND before/after**: use your NSW per-year tree maps (2017–2024) to find paddocks
   that gained/lost shelter and compare productivity trajectories, controlling for weather
   (Burrell 2017) — much stronger causal identification than the cross-section.
3. **Causal forest (Cambron 2024)** on the full covariate set for heterogeneous effects by
   region/soil/climate, mapped nationally.
4. **Scale**: rerun the sampler over all Australian cropping zones + more years (incl. the
   2017–2019 drought sequence) for the national, multi-year picture.
5. **Wind vs percent method**: compare leeward (wind) shelter definition — the asset is
   already sampled (`cls_wind`).
