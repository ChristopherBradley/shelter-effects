# ALA → satellite model: what's realistic to map?

Quick feasibility scan of Atlas of Living Australia (ALA) plant records in the ACT
(bbox -35.95..-35.10, 148.70..149.45), to see which species have enough geolocated
records *and* the right ecology to train a satellite model. **411,650** plant
occurrences total — data volume is not the problem; *spatial structure* is.

## The key distinction

ALA records are **point occurrences** (someone saw/collected a plant there). A satellite
model needs the species to occupy **areal stands** so that 10 m Sentinel-2 pixels are
reasonably pure. So the realistic targets are species that *cover ground in patches*, not
scattered individuals:

- **Stand-forming weeds** ✅ — dense infestations, exactly what land managers want mapped.
- **Dominant woodland trees** ✅ — form communities over hectares.
- **Scattered herbs / one-off records** ❌ — sub-pixel, won't map.

## Top stand-forming candidates in the ACT (by ALA record count)

| Species | Records | Type | Mappability |
|---|---|---|---|
| **Nassella trichotoma** (serrated tussock) | 8,694 | invasive grass | ✅ forms dense swards |
| **Hypericum perforatum** (St John's wort) | 7,562 | invasive forb | ✅ patchy infestations |
| **Rubus fruticosus** (blackberry) | 5,641 | invasive shrub | ✅ dense thickets — easiest target |
| **Eragrostis curvula** (African lovegrass) | 3,264 | invasive grass | ✅ the original weed goal — viable here |
| Casuarina cunninghamiana | 5,699 | native tree | ✅ riparian stands |
| Eucalyptus dives/pauciflora/rossii/melliodora/… | ~2,000 each | native trees | ✅ woodland communities |
| Themeda triandra, Poa sieberiana | ~2,200 | native grasses | ⚠️ mixed with other groundcover |

`genus:Eucalyptus` = 26,605 records, `genus:Acacia` = 15,600, `genus:Pinus` = only 385
(plantations are mapped as land use, not iNatted — use NLUM/Forests of Australia for pine).

## Recommended ALA target: **blackberry or serrated tussock weed mapping**

Best first model because: thousands of records, forms visually/spectrally distinct dense
patches, and is directly useful to land-management agencies (the stated goal). African
lovegrass is also viable (3,264 records) and was the original target — and note this ALA
route **sidesteps the street-view/Pl@ntNet dead-end** we found earlier (Pl@ntNet couldn't
ID lovegrass from imagery).

## How the modelling differs from crops (important)

ALA gives **presence-only** data with strong **sampling bias** (records cluster along
roads, reserves, near Canberra). So this is a **species-distribution / presence–background**
problem, not balanced pixel classification:

1. Take ALA presences for the target weed; snap to the S2 grid.
2. Generate **pseudo-absences / background** points (random, or bias-corrected by sampling
   the same accessibility surface as the presences, to avoid learning "near roads").
3. Features = the same S2 seasonal composite stack we built for crops (reuse `cropmap`).
4. Train a presence–background classifier (e.g. MaxEnt-style, or a calibrated RF/HGB),
   output a **probability-of-presence** raster, validate by spatial cross-validation and,
   ideally, a held-out field/ground dataset.

Precedent: citizen-science (iNat/GBIF) + Sentinel-2 SDMs are established (PNAS 2024; the
Sentinel-2 orchid SDM). ALA feeds GBIF, so this is well-trodden.

## Caveats to design around
- **Sampling bias** is the biggest risk — without bias correction the model maps
  "where people walk", not "where the weed is".
- **Presence-only** → no true negatives; needs background/pseudo-absence design.
- **Temporal mismatch** — records span many years; pair each with a season-appropriate
  composite, or model on a multi-year median.
- **Detection limits** — sparse/early infestations are sub-pixel; the model finds
  established patches, which is still the management-relevant signal.
