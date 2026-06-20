# Experimental design: effect of shelter on nearby crop/pasture productivity

Prototype for Goal 2 of the thesis ("estimating current effects of shelterbelts on
crop & pasture production during droughts across Australia"). Built to be run on Earth
Engine (free), validated against NVT.

## Question & hypotheses
- **H1 (competition vs shelter):** productivity is *reduced* immediately next to trees
  (competition/shade) and *raised* a few tree-heights out, then returns to open-field
  levels — the classic shelterbelt curve.
- **H2 (drought modulation):** the relative shelter benefit (and the relative
  competition cost) is **larger in drought years** than wet years.
- **H3 (heterogeneity):** the net effect varies with climate (drier → more benefit),
  soil, terrain, and cover type (crop vs pasture).

## Variables
- **Response Y:** peak-of-season (Aug–Oct) median **EVI** at 10 m, per year — a
  productivity / biomass proxy (calibrated to yield via NVT below).
- **Treatment / exposure:**
  - categorical **shelter class** from Chris's Australia-wide 2020 maps
    (percent method *and* wind method): sheltered vs unsheltered cropland/pasture.
  - continuous **distance to nearest tree** (>50% cover), capped 400 m — lets us
    resolve the competition→benefit curve rather than a single binary contrast.
- **Confounders X:** growing-season rainfall, max temp, PDSI (drought), climatic water
  deficit (per year); elevation, slope, aspect; soil clay, organic carbon, pH, bulk
  density (OpenLandMap). These address the **land-selection bias** — trees are often
  retained on poorer/steeper/rockier land, which would otherwise masquerade as a shelter
  effect.
- **Years:** 2019 (severe eastern-Australia drought) vs 2020–2021 (wet La Niña) — the
  drought contrast for H2. Shelter map is 2020 (shelterbelts are persistent, so it
  applies across years).

## Sampling
Stratified random points (300/class/tile) over many **0.4° tiles** across the southern
wheat–sheep belt (SE) and SW Western Australia. Tiling keeps each Earth Engine job under
the interactive limit **and** gives a naturally **matched design**: within a tile,
sheltered vs unsheltered pixels share climate/soil/region, so within-tile contrasts are
quasi-experimental. `scripts/shelter_sample*.py`.

## Analyses (`scripts/shelter_analysis.py`)
1. **Distance-decay curve** — tile-demeaned EVI vs distance-to-tree, by cover type and
   year. The headline test of H1.
2. **Naive vs adjusted shelter effect** — sheltered−unsheltered ΔEVI, then the same
   controlling for confounders via (a) **within-tile fixed effects** and (b) covariate
   regression. The gap between naive and adjusted quantifies the selection bias.
3. **Drought interaction (H2)** — distance-decay and shelter effect in 2019 vs 2021;
   shelter-effect vs rainfall.
4. **Heterogeneous effects (H3)** — a **causal forest / random-forest** estimate of the
   conditional effect of shelter controlling for X (à la Cambron 2024), with partial
   dependence and feature importance to see *where* shelter helps most.
5. **Yield calibration** — regress NVT site yields against EVI to convert the EVI effect
   into t/ha (and screen which crops calibrate best).

## Key caveats (being honest)
- "Sheltered" pixels sit ~55–70 m from trees (≈3–5 tree-heights) — squarely in the zone
  where competition and shelter both act, so the *binary* effect can be net-negative even
  when there's a real benefit further out. The distance curve is the more informative view.
- Observational, not randomised: confounder control + within-tile matching reduce but
  don't eliminate selection bias. RESTREND-style before/after (using the per-year tree
  maps) is the natural next step to strengthen causality.
- EVI is a productivity proxy; absolute effects depend on the NVT yield calibration.
