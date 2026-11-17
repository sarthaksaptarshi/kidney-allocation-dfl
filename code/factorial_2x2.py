"""
2x2 feasibility x priority factorial (review Comment 3) + direct between-cell contrasts
(second-review Comment 1). Same deliberately misspecified linear predictor throughout (base
feature map); we toggle feasibility (ABO/crossmatch edge pruning) and the NOTTO equity priority.
Each cell = paired regret reduction (two-stage - SPO+). Contrasts test whether the cells DIFFER
(a positive cell and a null cell do not by themselves establish a difference), using a paired,
seed-clustered two-level bootstrap. Outputs: results_integration/metrics_factorial.json
Run: python factorial_2x2.py
"""
import numpy as np, json, os
import integration_experiment as ie

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_integration")
os.makedirs(OUT, exist_ok=True)
LAM_ON = 0.12

def run_cell(use_feas, use_prio, SEEDS=10, N_TRAIN=24, N_TEST=24, nD=40, nR=55, epochs=60, lr=0.004):
    lam = LAM_ON if use_prio else 0.0
    two_all, spo_all = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed); rng_obs = np.random.default_rng(2000 + seed)
        train = [ie.make_pool(rng, nD, nR) for _ in range(N_TRAIN)]
        test  = [ie.make_pool(rng, nD, nR) for _ in range(N_TEST)]
        if not use_feas:
            for p in train + test: p["feas"] = np.ones_like(p["feas"], dtype=bool)
        th_ts  = ie.fit_two_stage(train, rng_obs)
        th_spo = ie.spo_plus_train(train, th_ts, lam, epochs=epochs, lr=lr)
        r_ts = np.array([ie.regret(p, (ie.design(p["feats"])[0] @ th_ts ).reshape(p["B"].shape), lam) for p in test])
        r_sp = np.array([ie.regret(p, (ie.design(p["feats"])[0] @ th_spo).reshape(p["B"].shape), lam) for p in test])
        two_all.append(r_ts); spo_all.append(r_sp)
    dbs = [two_all[s] - spo_all[s] for s in range(SEEDS)]            # per-seed arrays of per-batch diffs
    return dict(feasibility=bool(use_feas), priority=bool(use_prio), lam=lam), dbs

def cell_ci(dbs, rng, B=5000):
    S = len(dbs); v = []
    for _ in range(B):
        sb = rng.integers(0, S, S)
        v.append(np.mean([rng.choice(dbs[s], len(dbs[s]), replace=True).mean() for s in sb]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def contrast_ci(dbsA, dbsB, rng, B=5000):
    """A - B, paired by seed AND by batch (same pools, different objective)."""
    S = len(dbsA); v = []
    for _ in range(B):
        sb = rng.integers(0, S, S); aa, bb = [], []
        for s in sb:
            nb = len(dbsA[s]); idx = rng.integers(0, nb, nb)
            aa.append(dbsA[s][idx].mean()); bb.append(dbsB[s][idx].mean())
        v.append(np.mean(aa) - np.mean(bb))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

# ---- run the four cells ----
cells, DBS = [], {}
for uf in [False, True]:
    for up in [False, True]:
        meta, dbs = run_cell(uf, up); DBS[(uf, up)] = dbs
        red = float(np.mean([d.mean() for d in dbs]))
        lo, hi = cell_ci(dbs, np.random.default_rng(43))
        meta.update(reduction=round(red, 3), seed_clustered_95ci=[round(lo, 3), round(hi, 3)],
                    n_seeds_positive=int(sum(d.mean() > 0 for d in dbs)))
        cells.append(meta)
        v = "SIG+" if lo > 0 else ("SIG-" if hi < 0 else "tie")
        print(f"feas={uf!s:<5} prio={up!s:<5}: reduction {red:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {v}")

# ---- direct contrasts vs the unconstrained (neither) cell, + the interaction ----
neither = DBS[(False, False)]
contrasts = []
for name, A, B in [
    ("priority only - neither",    DBS[(False, True)],  neither),
    ("feasibility only - neither",  DBS[(True, False)],  neither),
    ("both - neither",              DBS[(True, True)],   neither)]:
    estimate = float(np.mean([a.mean() for a in A]) - np.mean([b.mean() for b in B]))
    lo, hi = contrast_ci(A, B, np.random.default_rng(71))
    contrasts.append(dict(contrast=name, estimate=round(estimate, 3), ci95=[round(lo, 3), round(hi, 3)]))
# interaction: (both - feas_only) - (prio_only - neither)
def smean(dbs): return np.array([d.mean() for d in dbs])
inter_per_seed = (smean(DBS[(True, True)]) - smean(DBS[(True, False)])) - (smean(DBS[(False, True)]) - smean(DBS[(False, False)]))
rng_i = np.random.default_rng(72); iv = [np.mean(inter_per_seed[rng_i.integers(0, 10, 10)]) for _ in range(5000)]
contrasts.append(dict(contrast="feasibility x priority interaction", estimate=round(float(inter_per_seed.mean()), 3),
                      ci95=[round(float(np.percentile(iv, 2.5)), 3), round(float(np.percentile(iv, 97.5)), 3)]))

print("\nDirect contrasts (change in SPO+ gain):")
for c in contrasts:
    v = "SIG" if (c["ci95"][0] > 0 or c["ci95"][1] < 0) else "n.s."
    print(f"  {c['contrast']:<38} {c['estimate']:+.3f}  CI[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]  {v}")

json.dump(dict(cells=cells, contrasts=contrasts), open(f"{OUT}/metrics_factorial.json", "w"), indent=2)
print("\nsaved", f"{OUT}/metrics_factorial.json")
