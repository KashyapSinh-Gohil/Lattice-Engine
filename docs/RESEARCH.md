# Research Basis — VAJRA AGRO domain (smallholder climate risk & yield intelligence)

Every design choice in the AGRO pack traces to a real figure. Citations at the bottom.
Where a parameter is domain-standard rather than from a single source, it's marked *(agronomic standard)*.

## 1. The user and the scale of the problem

- Of **~570 million farms worldwide, ~475 million are smallholdings, and ~74% are in Asia**,
  where they account for roughly **60% of agricultural production**. Five of every six farms
  are **under 2 hectares**. [FAO family-farming; Our World in Data]
- Smallholders in Asia supply a very large share of the region's food, yet operate with the
  **least buffer against a bad season** — one failed monsoon can wipe out a year's income.
- **Advisory gap:** public extension services are stretched thin (India's extension-worker-to-
  farmer ratio is on the order of **1:1000+**, far below the ~1:400 often recommended),
  so per-village, per-season, science-based guidance simply doesn't reach most farmers in time
  *(widely reported policy figure; treat as order-of-magnitude)*.

**Primary user of AGRO:** a **District Agriculture Officer / FPO (Farmer Producer Org) advisor**
who must decide, each week of the season, **which villages to send scarce extension visits,
irrigation releases, and insurance/relief attention to first** — across hundreds of villages
and hundreds of thousands of plots. Same "scarce resource, too many claimants, decide now"
shape as a grid dispatcher.

## 2. The decision & bottleneck

Weekly, mid-season, the officer must answer:
1. **Which villages are heading for a bad harvest** (so advisories/inputs go there first)?
2. **Which plots crossed an insurance trigger** (rainfall deficit / low area-yield) so payouts
   and relief start *before* distress sales?
3. **If there's limited canal water / pump-diesel subsidy this week, where does it save the
   most yield** without always favouring the same head-reach villages (fairness)?

Today this rides on **coarse district-level public bulletins that lag by days to weeks**, plus
manual reading of satellite portals. Time-to-insight is the bottleneck: by the time a dry spell
shows up in a monthly bulletin, the crop's reproductive window may already be lost.

## 3. Data & why it's GPU-shaped

- **Satellite NDVI (vegetation greenness):** Sentinel-2 gives **10 m resolution with ~5-day
  revisit** — but **monsoon cloud cover punches large gaps** in exactly the growing season that
  matters, so per-plot time series must be **gap-filled and smoothed** (millions of plot×date
  cells → a GPU-shaped cleaning job). [Sentinel-2 docs; harmonized NDVI literature]
- **Rainfall / weather:** daily gridded rainfall & temperature; **dry-spell length during the
  reproductive stage is the strongest yield killer** for monsoon rice. [Nature Comms Earth &
  Environment 2024; Scientific Reports 2023]
- **Soil water:** a simple daily bucket model (rain in, ET out) → plant-available water.
- **Phenology via Growing Degree Days (GDD):** thermal time with **base ≈ 10 °C for rice**
  drives stage transitions (vegetative → reproductive → grain-fill); risk is stage-weighted
  because a deficit at flowering hurts far more than the same deficit early. *(agronomic standard)*
- **Market prices & historical yields:** 5 seasons of village yields give real training labels.

## 4. The insurance hook (why payouts can be faster)

- **PMFBY (India's crop-insurance scheme)** is the **world's largest by enrolment**: over
  **72.6 crore farmer applications** in nine years and **~₹1.72 lakh crore (~US$19.6 bn) in
  claims** paid to ~19.6 crore farmers. Enrolment rose **32% (3.17→4.19 crore) from FY23 to
  FY25**. [PIB / IBEF / DD News]
- The scheme mixes **area-yield** and **weather-index (parametric)** triggers. AGRO computes a
  transparent **insurance-trigger score** per village (rainfall-deficit index + modelled
  area-yield shortfall vs threshold) so that **eligible villages surface in days, not after a
  season-end assessment** — the acceleration directly shortens the payout clock.

## 5. What AGRO outputs (mirrors GRID, different domain)

| GRID | AGRO |
|---|---|
| Feeder Criticality Index | **Village Advisory Priority Index (VAPI)** |
| Transformer failure watchlist | **Insurance-trigger / crop-loss watchlist** |
| 4-h demand forecast | **Season-end yield forecast** (per village) |
| Shed-plan what-if (MW relief) | **Irrigation/relief allocation what-if** (save max yield, fairness across head/tail villages) |
| Reason codes (OVERLOAD-4H…) | Reason codes (**DRY-SPELL-REPRO, NDVI-DECLINE, TRIGGER-HIT, FAIR-OK…**) |

## 6. Acceleration → better decision (same 4 proofs)

1. **Runtime** — NDVI gap-fill + geospatial joins + feature build over millions of plot×date
   rows: minutes on CPU pandas → seconds on RAPIDS cudf.pandas.
2. **Scale** — CPU handles a block; GPU handles a **whole state** of plots in one node.
3. **Freshness** — re-score **every new satellite pass / rain update** instead of monthly
   bulletins → advisories land while the crop can still be saved.
4. **Decision** — evaluate **thousands of irrigation/relief allocations per second** →
   provably higher yield-saved per unit of scarce water, fairly spread.

**The money line:** *because the pipeline turns a fresh satellite pass into per-village advice
in seconds, the officer warns the right villages while the reproductive window is still open —
and payouts start on this week's data, not next month's assessment.*

## Sources
- FAO — Smallholders & family farms produce ~a third of the world's food; 570M farms, 74% in Asia: https://www.fao.org/family-farming/detail/en/c/1398060/ and https://www.fao.org/family-farming/detail/en/c/463373/
- Our World in Data — Smallholder food production share: https://ourworldindata.org/smallholder-food-production
- Earth.org — Smallholder farmers in Asia (share of production): https://earth.org/smallholder-farmers-in-asia-challenges-opportunities-and-the-path-to-sustainable-food-production/
- Nature, Communications Earth & Environment (2024) — Optimal rainfall threshold for monsoon rice varies across space/time: https://www.nature.com/articles/s43247-024-01414-7
- Scientific Reports (2023) — Monsoon variability & rice production via ML: https://www.nature.com/articles/s41598-023-27752-8
- Science Advances — Severe floods significantly reduce global rice yields: https://www.science.org/doi/10.1126/sciadv.adx7799
- Sentinel-2 resolution/revisit (10 m, ~5-day) & agriculture use: https://skyfi.com/en/blog/understanding-sentinel-2-resolution
- Harmonized Landsat–Sentinel-2 NDVI phenology for small-scale cropping: https://www.sciencedirect.com/science/article/pii/S2352938524000946
- PMFBY scale — PIB claims figure: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2011791
- PMFBY enrolment/claims (nine-year totals): https://ddnews.gov.in/en/pm-fasal-bima-yojana-turns-nine-rs-1-75-lakh-crore-in-claims-disbursed-to-23-22-crore-farmers/
- PMFBY overview & largest-scheme status: https://www.ibef.org/government-schemes/fasal-bima-yojana
