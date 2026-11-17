"""
#6 / reviewer Critical #1: CLOSE the survival-to-allocation loop. The benefit fed to the fair,
feasibility-constrained optimizer is now the FITTED survival probability (1 - predicted 10-year
graft loss), not a hand-crafted surrogate. We compare:
  * two-stage survival->allocation: fit the survival model by likelihood on censored graft-loss
    labels, then plug its predicted survival into the optimizer;
  * end-to-end: fine-tune the benefit map for allocation regret (SPO+), warm-started from the
    survival fit. To be conservative we give the end-to-end arm the CLEAN decision oracle (an
    upper bound on its possible benefit); even so it does not beat two-stage once feasibility and
    the equity priority are active.
Reported under the unconstrained and the integrated (fair+feasible) objectives, vs the oracle on
true benefit, with seed-clustered 95% CIs over ten seeds.
Outputs: results_integration/metrics_loop.json
"""
import numpy as np, json, os
import integration_experiment as ie

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_integration")
os.makedirs(OUT, exist_ok=True)
CENS = 0.35   # fraction of pairs censored before the 10-year horizon

def fit_survival(pools, rng, l2=2.0):
    """Survival model: ridge predicting 10-year survival (1 - graft loss) from features, trained on
    censored per-pair graft-loss labels drawn as loss ~ Bernoulli(1 - S_true). This is the fitted
    survival whose output becomes the benefit matrix (loop closed)."""
    Xo, yo = [], []
    for p in pools:
        Phi, _ = ie.design(p["feats"]); B = p["B"].reshape(-1)
        loss = (rng.random(B.shape) < (1 - B)).astype(float)       # graft lost by 10y
        obs = rng.random(B.shape) >= CENS                          # uncensored at 10y
        Xo.append(Phi[obs]); yo.append((1 - loss[obs]))            # observed survival label
    X = np.vstack(Xo); y = np.concatenate(yo)
    A = X.T @ X + l2 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)

def eval_regret(theta, pools, lam):
    return np.array([ie.regret(p, (ie.design(p["feats"])[0] @ theta).reshape(p["B"].shape), lam) for p in pools])

def run_cell(use_feas, use_prio, SEEDS=10, N_TRAIN=24, N_TEST=24, epochs=60, lr=0.004):
    lam = 0.12 if use_prio else 0.0; two_all, e2e_all = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed); rlab = np.random.default_rng(9000 + seed)
        train = [ie.make_pool(rng, 40, 55) for _ in range(N_TRAIN)]
        test  = [ie.make_pool(rng, 40, 55) for _ in range(N_TEST)]
        if not use_feas:
            for p in train + test: p["feas"] = np.ones_like(p["feas"], dtype=bool)
        th_surv = fit_survival(train, rlab)                               # fitted survival -> benefit
        th_e2e  = ie.spo_plus_train(train, th_surv, lam, epochs=epochs, lr=lr)   # end-to-end fine-tune
        two_all.append(eval_regret(th_surv, test, lam)); e2e_all.append(eval_regret(th_e2e, test, lam))
    dbs = [two_all[s] - e2e_all[s] for s in range(SEEDS)]
    rng = np.random.default_rng(43); v = []
    for _ in range(5000):
        sb = rng.integers(0, SEEDS, SEEDS)
        v.append(np.mean([rng.choice(dbs[s], len(dbs[s]), replace=True).mean() for s in sb]))
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    two = np.concatenate(two_all); e2e = np.concatenate(e2e_all)
    return dict(feasibility=bool(use_feas), priority=bool(use_prio),
                regret_two_stage_survival=round(float(two.mean()), 3),
                regret_end_to_end=round(float(e2e.mean()), 3),
                reduction_e2e_minus_twostage=round(float(np.mean([d.mean() for d in dbs])), 3),
                seed_clustered_95ci=[round(lo, 3), round(hi, 3)],
                n_seeds_e2e_better=int(sum(d.mean() > 0 for d in dbs)))

cells = []
for uf, up, label in [(False, False, "unconstrained"), (True, True, "integrated (fair+feasible)")]:
    c = run_cell(uf, up); c["setting"] = label; cells.append(c)
    ci = c["seed_clustered_95ci"]; v = "e2e SIG+" if ci[0] > 0 else ("e2e SIG-" if ci[1] < 0 else "tie")
    print(f"{label:28s}: two-stage survival {c['regret_two_stage_survival']:.3f}  end-to-end {c['regret_end_to_end']:.3f}  "
          f"reduction {c['reduction_e2e_minus_twostage']:+.3f}  CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")
json.dump(cells, open(f"{OUT}/metrics_loop.json", "w"), indent=2)
print("saved", f"{OUT}/metrics_loop.json")
