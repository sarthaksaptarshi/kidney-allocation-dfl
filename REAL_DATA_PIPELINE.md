# Real-Data Pipeline Guide (SRTR/OPTN re-estimation — paper's main next step)

**Goal:** run the paper's survival comparison (proposed discrete-time multi-horizon model vs Cox PH) on
a large transplant registry the moment the data arrives, and report honestly whether the proposed
method actually beats Cox. The bottleneck is **data access, not code** — `code/star_pipeline.py` is
built and validated; you supply the registry extract.

## 1. Which dataset, and how to request it

No openly downloadable set is large enough with covariates (the Mendeley cohort has 436 recipients / 31
events). A real win needs a registry, all DUA-gated:

| Registry | Size | Access | Timeline |
|---|---|---|---|
| **OPTN/UNOS STAR** (recommended) | ~1M+ transplants, graft+patient survival, censoring | OPTN Data Request page → signed DUA → secure delivery | ~30 days after signed DUA |
| **SRTR SAF** | Same scope, risk-adjusted | Email `srtr@srtr.org`; ~30-day review | ~30 days |
| **NHSBT UK Transplant Registry** | 36k+ accepted offers | NHSBT application | Weeks–months |
| **ANZDATA** (AUS/NZ) | National dialysis+transplant | ANZDATA application | Weeks–months |

Most require institutional affiliation and often IRB/ethics approval. **Recommended: OPTN STAR**, kidney
file (KIDPAN). Download the OPTN STAR Data Dictionary with it — variable names vary by release.

## 2. STAR → pipeline column mapping (confirm against the Data Dictionary)

| Pipeline role | STAR variable (typical) | Notes |
|---|---|---|
| `--time` | `GTIME` | days to graft failure/last follow-up → `--time-unit days` |
| `--event` | `GSTATUS` | 1 = graft failure, 0 = censored |
| `--num` | `AGE`, `AGE_DON`, `HLAMIS` (or `DRMIS`+`AMIS`+`BMIS`), `END_CPRA`/`PRA`, `COLD_ISCH_KI`, days on dialysis, `BMI_CALC`, `BMI_DON_CALC` | pre-transplant only |
| `--cat` | `DIAB`, ABO-match flag, `DON_TY`/DCD flag, `PREV_KI` (regraft) | pre-transplant only |

**Leakage rule (as in the paper):** use offer-time / pre-transplant fields only; exclude anything
recorded after transplant.

## 3. Validate first, then run for real

```bash
# A) confirm the pipeline works (no real data needed)
python code/make_synthetic_star.py
python code/star_pipeline.py synthetic_star.csv \
    --time GTIME_Y --event GSTATUS \
    --num rec_age don_age hla_mm cpra cold_isch dial_vintage don_bmi rec_bmi \
    --cat diabetes abo_mm dcd regraft --grid 0 1 5 10

# B) run on the real STAR extract
python code/star_pipeline.py KIDPAN.csv \
    --time GTIME --event GSTATUS --time-unit days \
    --num AGE AGE_DON HLAMIS END_CPRA COLD_ISCH_KI DAYS_DIAL BMI_CALC BMI_DON_CALC \
    --cat DIAB ABO_MAT DCD PREV_KI --grid 0 1 5 10
```

## 4. What decides the "win"

The pipeline reports `proposed_minus_cox_paired`: the paired bootstrap mean and 95% CI of
C(proposed) − C(Cox), on identical resampled subjects and pairs.
- **CI entirely above 0** → proposed genuinely beats Cox (the real-data win).
- **CI includes 0** → a tie (consistent with the literature's caution that ML rarely beats Cox on the same variables).
- **CI below 0** → Cox wins.

On the synthetic validation file Cox edges the proposed model — expected for near-proportional-hazards
data, and it confirms the test is calibrated rather than that the method loses on real data.

## 5. Performance

C-index uses fast vectorized **pair sampling** (exact Harrell C is O(n²), infeasible on a registry);
default `--pairs 500000` per bootstrap. Fitting uses all rows (~15–30 min on ~400k rows; lower
`--folds`/`--pairs` to speed up). Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib.
