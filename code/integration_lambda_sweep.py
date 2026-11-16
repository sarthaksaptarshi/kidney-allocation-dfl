"""
Why does the decision-focused (SPO+) advantage vanish under the integrated objective?
Sweep the fairness weight lambda from 0 (utility-only) upward. If SPO+ wins at lambda=0 and the
gain shrinks to null as lambda grows, the mechanism is clear: the fairness priority and the
feasible-edge restriction make the realized allocation robust to HOW benefit is predicted, so the
simpler two-stage predictor suffices once equity/compatibility constraints are active.
"""
import json, os
import integration_experiment as ix

OUT = ix.OUT
rows = []
for lam in [0.0, 0.03, 0.06, 0.12, 0.25, 0.50]:
    print(f"\nlambda = {lam}")
    r = ix.run(lam, verbose=False)
    print(f"  two-stage {r['regret_two_stage']:.3f}%  SPO+ {r['regret_spo_plus']:.3f}%  "
          f"reduction {r['paired_reduction']:+.3f} CI {r['seed_clustered_95ci']}  seeds+ {r['n_seeds_positive']}/10")
    rows.append(r)

json.dump(rows, open(f"{OUT}/metrics_lambda_sweep.json", "w"), indent=2)
print("\nSaved", f"{OUT}/metrics_lambda_sweep.json")
