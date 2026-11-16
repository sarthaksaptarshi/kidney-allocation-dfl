# Datasets

The synthetic generator needs no external data. The two real-data arms use openly licensed public
datasets, which are **not redistributed here** — obtain them from the original sources below.

## 1. UCI Chronic Kidney Disease (detection arm)

- Source: UCI Machine Learning Repository, dataset 336.
- DOI: `10.24432/C5G020`
- URL: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- License: open (UCI terms).
- Access in code: loaded automatically through the `ucimlrepo` package — no manual download needed:
  ```bash
  python code/ckd_detection.py
  ```

## 2. Mendeley kidney-allograft cohort (rejection + survival arms)

- Citation: C. A. Chukwu, "Outcomes of basiliximab and alemtuzumab induction in kidney allograft
  recipients with matched immunological risk profile: a retrospective matched cohort study,"
  *Mendeley Data*, V1, 2025.
- DOI: `10.17632/3vdszzmz2n.1`
- URL: https://data.mendeley.com/datasets/3vdszzmz2n/1
- License: CC BY 4.0.
- Download the anonymised CSV, then pass its path:
  ```bash
  python code/rejection_real.py /path/to/campath.simulect_anonymised_data.csv
  python code/survival_real.py  /path/to/campath.simulect_anonymised_data.csv --endpoint dcgl
  ```

## 3. Jakarta kidney-transplant cohort (second survival arm)

- Citation: "Long-term outcomes and prognostic factors in kidney transplant recipients in Jakarta,
  Indonesia: a cohort study," *Dryad Dataset*, 2022.
- DOI: `10.5061/dryad.41ns1rnfr`
- URL: https://datadryad.org/dataset/doi:10.5061/dryad.41ns1rnfr
- License: open (Dryad terms).
- Download `survival_data.xlsx`, then prepare it (months→years; drops the post-transplant leakage
  columns Rejection / Infections / Delayed_Graft) and run the pipeline:
  ```bash
  python code/prep_jakarta.py /path/to/survival_data.xlsx      # -> jakarta_clean.csv (also included)
  python code/star_pipeline.py jakarta_clean.csv --time PT_DC_y --event DC_Graft \
      --num rec_age60 rec_overweight dial_gt24mo smoking diabetes cardiovascular \
            don_age40 related_donor don_overweight don_gfr100 \
      --cat dialysis_type crossmatch --grid 0 1 5 10
  python code/build_fig_survival2.py                           # -> out/fig_survival.png (two-cohort)
  ```
  A prepared `code/jakarta_clean.csv` is included for convenience.

No patient-level identifiers are used. Only pre-transplant variables enter the rejection and survival
models; all post-transplant trajectory variables are withheld (see the paper, Section II-H and Table IV).
