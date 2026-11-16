# A Decision-Focused, Fairness-Constrained Framework for Multi-Horizon Graft-Rejection Prediction and Equitable Kidney Allocation

Reproducibility repository for the manuscript submitted to *IEEE Transactions on Artificial Intelligence*.

This repository contains the seed-fixed synthetic generator, all experiment scripts, and the real-data
analysis code used in the paper. The synthetic results are fully reproducible from source; the two
real-data arms use openly licensed public datasets (see [`data/README.md`](data/README.md)).

> **Note on scope.** The survival predictor, conformal uncertainty, and NOTTO-inspired fairness
> benchmark are evaluated as honestly-scoped modules. The decision-focused loop is additionally
> tested **end-to-end** — the fitted benefit feeds a fair, feasibility-constrained optimizer with the
> equity priority and ABO/crossmatch feasible-edge set inside **both** training and evaluation. That
> integration test finds the SPO+ advantage does **not** survive the constraints (a simple two-stage
> predictor matches it across equity weights and data regimes); see `results_integration/` and the
> paper's Results/Discussion. Substituting fitted survival probabilities and conformal abstention into
> the same trained pipeline remains future work.

## Repository layout

```
code/                     all experiment scripts (Python)
  sim.py                  synthetic generator + multi-horizon models, calibration, conformal, ablation
  synthetic_cox.py        Cox baseline, paired bootstrap CIs, event-count crossover, full-label sensitivity
  decision_focused.py     SPO+ vs two-stage, seed-clustered bootstrap (utility-only module)
  integration_experiment.py   end-to-end fair+feasible integration: two-stage vs SPO+ (importable run())
  integration_lambda_sweep.py equity-weight (lambda) sweep of the integration regret
  integration_regime.py   integration across data/pool-size regimes (scarce labels, small pools)
  generate_fig_integration.py composes the 4-panel decision-focused figure (Fig. 8)
  fairness_multiscenario.py  NOTTO-inspired additive priority, scarcity regimes, ABO/crossmatch filter
  figs2.py                cohort, cumulative-incidence, conformal, confusion figures
  arch.py                 architecture schematic
  ckd_detection.py        real-data CKD detection (UCI dataset)
  rejection_real.py       real-data acute-rejection prediction (Mendeley cohort)
  survival_real.py        real-data discrete-time survival vs Cox (Mendeley cohort)
  star_pipeline.py        schema-driven large-registry survival pipeline (paired C-index win test)
  make_synthetic_star.py  synthetic STAR-shaped validation file for star_pipeline.py
  prep_jakarta.py         prepare the public Jakarta cohort (months->years, drop leakage cols)
  build_fig_survival2.py  two-cohort real-data survival figure (Mendeley + Jakarta)
  make_tables.py          emits LaTeX-ready numbers
  build_survival_fig.py   composes the real-data survival figure
figures/                  the 12 figures used in the manuscript
results_integration/      saved metrics for the end-to-end integration experiment (JSON)
data/                     instructions for obtaining the two public datasets
paper/                    compiled manuscript (PDF, DOCX, LaTeX source)
requirements.txt          Python dependencies
```

## Reproduce the synthetic results

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python code/sim.py                 # core synthetic arm -> out/summary.json + calibration
python code/synthetic_cox.py       # Cox comparison + crossover + full-label sensitivity
python code/decision_focused.py    # SPO+ utility-only module experiment
python code/integration_experiment.py    # end-to-end fair+feasible integration (baseline lambda)
python code/integration_lambda_sweep.py  # integration across equity weights
python code/integration_regime.py        # integration across data/pool-size regimes
python code/fairness_multiscenario.py   # NOTTO fairness + compatibility-filtered sensitivity
python code/figs2.py && python code/generate_fig_integration.py  # figures (Fig. 8 = decision-focused)
```

All synthetic scripts are seed-fixed (seed `20261116`). Independent re-runs under different library
versions vary by about ±0.001 in AUROC and a few hundredths of a point in regret, as noted in the paper.

## Reproduce the real-data arms

See [`data/README.md`](data/README.md) to obtain the UCI Chronic Kidney Disease dataset and the Mendeley
kidney-allograft cohort, then:

```bash
python code/ckd_detection.py                                   # UCI loads via ucimlrepo
python code/rejection_real.py  /path/to/mendeley.csv           # positional path to the CSV
python code/survival_real.py   /path/to/mendeley.csv --endpoint dcgl
```

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)). A Zenodo DOI will be
minted on release and added here.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The public datasets retain their own
licenses (UCI CKD: open; Mendeley cohort: CC BY 4.0).
