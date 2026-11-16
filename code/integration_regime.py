"""
Does decision-focused learning help in the OPERATIONALLY REALISTIC regime?
Real offer batches at a center are small (a handful of candidates per organ) and labeled graft
outcomes are scarce. SPO+ theory says decision-focused training helps most when data is limited and
each decision is pivotal. Test under the fair+feasible objective (lam=0.12), varying training size
and pool size. Compare to the data-rich regime already run.
"""
import json
import integration_experiment as ix

configs = [
    ("data-rich / large pool (baseline)", dict(N_TRAIN=24, nD=40, nR=55)),
    ("scarce labels / large pool",        dict(N_TRAIN=5,  nD=40, nR=55)),
    ("data-rich / small pool",            dict(N_TRAIN=24, nD=10, nR=14)),
    ("scarce labels / small pool",        dict(N_TRAIN=5,  nD=10, nR=14)),
    ("very scarce / small pool",          dict(N_TRAIN=3,  nD=8,  nR=11)),
]
rows = []
for name, cfg in configs:
    r = ix.run(0.12, N_TEST=24, verbose=False, **cfg)
    r["config"] = name
    print(f"{name:38s}  two-stage {r['regret_two_stage']:.3f}%  SPO+ {r['regret_spo_plus']:.3f}%  "
          f"reduction {r['paired_reduction']:+.3f} CI {r['seed_clustered_95ci']}  seeds+ {r['n_seeds_positive']}/10")
    rows.append(r)

json.dump(rows, open(f"{ix.OUT}/metrics_regime.json", "w"), indent=2)
print("\nSaved", f"{ix.OUT}/metrics_regime.json")
