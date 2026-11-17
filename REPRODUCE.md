# Reproduction map

Python 3.13, dependencies pinned in `requirements.txt` (`pip install -r requirements.txt`).
All synthetic point estimates are seed-fixed and reproduce across BLAS builds to within
~0.03 regret points, with every significance verdict unchanged. Each script writes a metrics
JSON; the figures read those JSONs (no hard-coded numbers).

| Paper element | Script | Output |
|---|---|---|
| Fig. 1 (architecture) | `code/arch.py` | `out/fig_architecture.png` |
| Fig. 8 (decision-focused, 4 panels) | `code/generate_fig_integration.py` | reads `results_dfl/metrics_dfl.json`, `results_integration/metrics_{lambda_sweep,regime}.json` |
| Utility-only base vs enriched | `code/decision_focused.py` | `results_dfl/metrics_dfl.json` |
| Integration baseline (lambda=0.12) | `code/integration_experiment.py` | `results_integration/metrics_integration.json` |
| Lambda sweep (panel c) | `code/integration_lambda_sweep.py` | `results_integration/metrics_lambda_sweep.json` |
| Regime sweep (panel d) | `code/integration_regime.py` | `results_integration/metrics_regime.json` |
| Table VI + between-cell contrasts | `code/factorial_2x2.py` | `results_integration/metrics_factorial.json` |
| MLP capacity check + convergence | `code/mlp_capacity.py` | `results_integration/metrics_mlp.json` |
| Survival-to-allocation loop closure | `code/loop_closure.py` | `results_integration/metrics_loop.json` |
| Two-cohort survival (Fig. for real data) | `code/build_fig_survival2.py` | reads `results_survival/` + `results_star/`; `out/fig_survival.png` |
| Jakarta survival C-index | `code/star_pipeline.py jakarta_clean.csv --time PT_DC_y --event DC_Graft ...` | `results_star/metrics_star.json` |
| Mendeley survival C-index (leak-free) | `code/survival_real.py <campath.csv> --endpoint dcgl` | `results_survival/metrics_survival.json` |
| CKD detection | `code/ckd_detection.py` | `out/` |

## Notes
- Preprocessing (imputation, standardization) is fit **inside** each cross-validation fold in every
  real-data script (`star_pipeline.py`, `survival_real.py`, `build_fig_survival2.py`); no test
  information leaks.
- The Mendeley raw data (`campath.simulect_anonymised_data.csv`, CC BY 4.0) is not redistributed
  here; obtain it from the cited Mendeley Data record and pass its path to `survival_real.py`.
- `factorial_2x2.py` reports both the per-cell reductions and the direct between-cell contrasts;
  the contrasts' 95% CIs all include zero, so the factorial is consistent with the gain being
  confined to the unconstrained regime but does not establish that any single constraint reduces it.
- `mlp_capacity.py` result is learner- and training-protocol-dependent: at the released budget SPO+
  lowers MLP regret in both settings; the convergence trajectory (unconstrained, 3 seeds) is reported
  as a hypothesis for an optimization-budget explanation, not as an established null for the MLP.
- `loop_closure.py` is a reduced-form **survival-label proxy** (ridge on a binary 10-year label, 35%
  missing-completely-at-random), NOT the interval-hazard estimator; the end-to-end arm is given the
  clean oracle (privileged information), and the integrated interval spans zero.

## Exact commands (run from `code/`; each script creates its output dir relative to `code/`)
```
pip install -r requirements.txt
python decision_focused.py                 # -> results_dfl/metrics_dfl.json, out/
python integration_experiment.py           # -> results_integration/metrics_integration.json
python integration_lambda_sweep.py         # -> results_integration/metrics_lambda_sweep.json
python integration_regime.py               # -> results_integration/metrics_regime.json
python factorial_2x2.py                     # -> results_integration/metrics_factorial.json
python mlp_capacity.py                      # -> results_integration/metrics_mlp.json
python loop_closure.py                      # -> results_integration/metrics_loop.json
python star_pipeline.py jakarta_clean.csv --time PT_DC_y --event DC_Graft \
    --num rec_age60 rec_overweight dial_gt24mo smoking diabetes cardiovascular \
          don_age40 related_donor don_overweight don_gfr100 \
    --cat dialysis_type crossmatch --grid 0 1 5 10   # -> results_star/metrics_star.json
python survival_real.py <campath.csv> --endpoint dcgl  # -> results_survival/metrics_survival.json
python arch.py                              # -> out/fig_architecture.png
python generate_fig_integration.py         # -> out/fig_dfl_integration.png  (reads the JSONs above)
python build_fig_survival2.py              # -> out/fig_survival.png          (reads results_survival + results_star)
```
