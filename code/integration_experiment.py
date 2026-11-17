"""
END-TO-END INTEGRATION EXPERIMENT (addresses the standing "modular, not integrated" gap).

One benefit definition throughout. A survival/benefit predictor's fitted output is fed into a
FAIR, FEASIBILITY-CONSTRAINED assignment optimizer, and we compare two-stage vs decision-focused
(SPO+) training under the SAME objective -- the fairness priority s_i and the ABO/crossmatch
feasible-edge set are inside BOTH training and evaluation. Regret is measured against the fair,
feasible oracle. This is the joint test the utility-only benchmark could not provide.

Outputs: results_integration/metrics_integration.json
"""
import numpy as np, json, os
from scipy.optimize import linear_sum_assignment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_integration")
os.makedirs(OUT, exist_ok=True)
ABO = ["O", "B", "A", "AB"]; ABO_P = [0.37, 0.33, 0.23, 0.07]
BIG = 1e6

def make_pool(rng, nD=40, nR=55):
    # recipient / donor attributes
    r_age = rng.uniform(18, 75, nR); r_hla = rng.integers(0, 7, nR).astype(float)
    r_cpra = np.where(rng.random(nR) < 0.18, rng.uniform(55, 99, nR), rng.beta(1.4, 6, nR) * 100)
    r_dial = np.clip(rng.gamma(2, 1.6, nR), 0, 18)
    female = rng.binomial(1, 0.40, nR).astype(bool); ped = rng.binomial(1, 0.08, nR).astype(bool)
    r_abo = rng.choice(ABO, nR, p=ABO_P)
    d_age = rng.uniform(5, 78, nD); d_hla = rng.integers(0, 7, nD).astype(float); d_abo = rng.choice(ABO, nD, p=ABO_P)
    hla = np.abs(d_hla[:, None] - r_hla[None, :])
    cpra = np.tile(r_cpra[None, :], (nD, 1)); da = np.tile(d_age[:, None], (1, nR)); ra = np.tile(r_age[None, :], (nD, 1))
    dial = np.tile(r_dial[None, :], (nD, 1))
    # base per-pair features (what the predictor sees)
    feats = np.stack([np.ones_like(hla), hla / 6, (cpra - 30) / 30, (da - 45) / 20, (ra - 45) / 18, (dial - 3) / 4], axis=-1)
    # TRUE benefit = a 5-year graft-survival proxy with structure a linear learner cannot fully fit
    beta = np.array([0.80, -0.22, -0.10, -0.06, -0.04, 0.05]); lin = feats @ beta
    inter = (-0.35 * (hla / 6) * (cpra > 80) - 0.25 * (hla / 6) * ((cpra - 30) / 30) - 0.22 * (hla / 6) ** 2 + 0.15 * np.exp(-dial / 4))
    B_true = np.clip(lin + inter, 0.05, 0.98)
    # NOTTO-inspired priority (normalized to [0,1]); groups are per-recipient
    sens = r_cpra > 80
    pts = 0.3 * r_dial + 2.0 * female + 5.0 * ped + 4.0 * sens; s = pts / pts.max()
    # feasibility: ABO-compatible AND crossmatch-negative (XM+ prob rises with cPRA)
    xm_pos = rng.random((nD, nR)) < (0.02 + 0.9 * r_cpra[None, :] / 100.0)
    abo_ok = np.array([[(d_abo[j] == "O") | (d_abo[j] == r_abo[i]) | (r_abo[i] == "AB") for i in range(nR)] for j in range(nD)])
    feas = abo_ok & (~xm_pos)
    return dict(feats=feats, B=B_true, s=s, feas=feas, groups=dict(female=female, sensitized=sens, pediatric=ped))

def fair_feasible_assign(Bpred, s, feas, lam):
    """Hungarian on the FAIR, FEASIBLE objective: maximize sum a_ji (B_ji + lam s_i), feasible edges only."""
    nD, nR = Bpred.shape; cost = -(Bpred + lam * s[None, :]); cost = np.where(feas, cost, BIG)
    n = max(nD, nR); C = np.full((n, n), BIG); C[:nD, :nR] = cost
    r, c = linear_sum_assignment(C)
    return [(ri, ci) for ri, ci in zip(r, c) if ri < nD and ci < nR and feas[ri, ci]]

def fair_obj(B, s, m, lam): return sum(B[j, i] + lam * s[i] for j, i in m)
def regret(pool, Bpred, lam):
    B, s, feas = pool["B"], pool["s"], pool["feas"]
    U_or = fair_obj(B, s, fair_feasible_assign(B, s, feas, lam), lam)          # fair+feasible oracle on TRUE benefit
    U_hat = fair_obj(B, s, fair_feasible_assign(Bpred, s, feas, lam), lam)     # act on predicted benefit
    return (U_or - U_hat) / U_or * 100

def design(feats): nD, nR, p = feats.shape; return feats.reshape(-1, p), (nD, nR)

def fit_two_stage(pools, rng_obs, lam=1.0):
    X = np.vstack([design(p["feats"])[0] for p in pools])
    y = np.concatenate([p["B"].reshape(-1) for p in pools])   # MATCHED LABELS (review C2): same clean benefit for both methods
    A = X.T @ X + lam * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ y)

def spo_plus_train(pools, theta0, LAM, epochs=60, lr=0.004):
    """SPO+ with the FAIR, FEASIBLE optimizer inside training (priority + feasibility in the loss)."""
    theta = theta0.copy(); pre = []
    for p in pools:
        Phi, (nD, nR) = design(p["feats"]); s, feas = p["s"], p["feas"]
        w_true = np.zeros((nD, nR))
        for j, i in fair_feasible_assign(p["B"], s, feas, LAM):   # fair+feasible oracle decision w*(c)
            w_true[j, i] = 1.0
        pre.append((Phi, (nD, nR), p["B"], s, feas, w_true))
    rng_spo = np.random.default_rng(777)
    for ep in range(epochs):
        g = np.zeros_like(theta)
        for idx in rng_spo.permutation(len(pre)):
            Phi, (nD, nR), B, s, feas, w_true = pre[idx]
            chat = (Phi @ theta).reshape(nD, nR)
            cost_hat = -(chat + LAM * s[None, :]); c_true = -(B + LAM * s[None, :])
            spo_cost = _pad(np.where(feas, 2 * cost_hat - c_true, BIG), nD, nR)
            rr, cc = linear_sum_assignment(spo_cost); w_spo = np.zeros((nD, nR))
            for ri, ci in zip(rr, cc):
                if ri < nD and ci < nR and feas[ri, ci]: w_spo[ri, ci] = 1.0
            sub = 2 * (w_true - w_spo); g += -(Phi.T @ sub.reshape(-1))
        theta -= lr * g / len(pre)
    return theta

def _pad(cost, nD, nR):
    n = max(nD, nR); C = np.full((n, n), BIG); C[:nD, :nR] = cost; return C

# ---------------- experiment ----------------
def run(LAM, SEEDS=10, N_TRAIN=24, N_TEST=24, nD=40, nR=55, epochs=60, lr=0.004, verbose=True):
    two_all, spo_all = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed); rng_obs = np.random.default_rng(2000 + seed)
        train = [make_pool(rng, nD, nR) for _ in range(N_TRAIN)]; test = [make_pool(rng, nD, nR) for _ in range(N_TEST)]
        th_ts = fit_two_stage(train, rng_obs)
        th_spo = spo_plus_train(train, th_ts, LAM, epochs=epochs, lr=lr)
        r_ts = np.array([regret(p, (design(p["feats"])[0] @ th_ts).reshape(p["B"].shape), LAM) for p in test])
        r_sp = np.array([regret(p, (design(p["feats"])[0] @ th_spo).reshape(p["B"].shape), LAM) for p in test])
        two_all.append(r_ts); spo_all.append(r_sp)
        if verbose: print(f"  seed {seed}: two-stage {r_ts.mean():.3f}%  SPO+ {r_sp.mean():.3f}%")
    two = np.concatenate(two_all); spo = np.concatenate(spo_all); diff_by_seed = [two_all[s] - spo_all[s] for s in range(SEEDS)]
    rc = np.random.default_rng(43); cboot = []
    for _ in range(5000):
        sb = rc.integers(0, SEEDS, SEEDS)
        cboot.append(np.mean([rc.choice(diff_by_seed[s], len(diff_by_seed[s]), replace=True).mean() for s in sb]))
    cci = (float(np.percentile(cboot, 2.5)), float(np.percentile(cboot, 97.5)))
    return dict(lam=LAM, seeds=SEEDS, test_batches_per_seed=N_TEST,
               regret_two_stage=round(float(two.mean()), 3), regret_spo_plus=round(float(spo.mean()), 3),
               paired_reduction=round(float((two - spo).mean()), 3), seed_clustered_95ci=[round(cci[0], 3), round(cci[1], 3)],
               n_seeds_positive=int(sum(d.mean() > 0 for d in diff_by_seed)))

if __name__ == "__main__":
    res = run(0.12); res["objective"] = "fair + feasibility-constrained (priority s_i and ABO/crossmatch in BOTH training and evaluation)"
    json.dump(res, open(f"{OUT}/metrics_integration.json", "w"), indent=2)
    print("\n=== INTEGRATION (fair + feasible, lam=0.12) ===\n", json.dumps(res, indent=2))
