"""
STEP 1 (reviewer comment 1): a DIRECT test of decision-focused training.

Head-to-head, with MATCHED features and capacity, on HELD-OUT donor batches:
  (A) Two-stage  (predict-then-optimize): predictor trained to minimize MSE, then
      its benefit estimates are fed to the assignment optimizer.
  (B) Decision-focused (SPO+): the SAME linear predictor trained with the Smart
      Predict-then-Optimize loss (Elmachtoub & Grigas, 2022), whose subgradient
      2[w*(c) - w*(2c_hat - c)] puts the assignment optimizer inside training.

Why SPO+ can win at equal capacity: the TRUE benefit is mildly nonlinear (pairwise
interactions), while BOTH predictors are linear in the same base features. Under this
(realistic) capacity limit, minimizing prediction error is not the same as minimizing
allocation regret -- SPO+ spends its limited capacity on the decisions.

Reports, across SEEDS x held-out BATCHES:
  * allocation regret (% vs oracle) for each method, with a PAIRED bootstrap 95% CI
    of the difference (exactly what the reviewer asked for);
  * prediction MSE for each method (SPO+ may be worse here yet better on regret).
Outputs: results_dfl/metrics_dfl.json and fig_dfl_regret.png

Dependencies: numpy, scipy, matplotlib only (no deep-learning framework needed).
Run:  python decision_focused.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_dfl"); os.makedirs(OUT, exist_ok=True)

# ----- problem generator: one donor batch + waitlist, with per-pair features -----
def make_pool(rng, nD=40, nR=55):
    # recipient / donor attributes
    r_age  = rng.uniform(18, 75, nR); r_hla = rng.integers(0, 7, nR).astype(float)
    r_cpra = np.where(rng.random(nR) < 0.18, rng.uniform(55, 99, nR), rng.beta(1.4, 6, nR)*100)
    r_dial = np.clip(rng.gamma(2, 1.6, nR), 0, 18)
    d_age  = rng.uniform(5, 78, nD); d_hla = rng.integers(0, 7, nD).astype(float)
    # per-pair base features (what BOTH predictors see): standardized-ish
    hla = np.abs(d_hla[:, None] - r_hla[None, :])          # (nD,nR)
    cpra = np.tile(r_cpra[None, :], (nD, 1))
    da = np.tile(d_age[:, None], (1, nR)); ra = np.tile(r_age[None, :], (nD, 1))
    dial = np.tile(r_dial[None, :], (nD, 1))
    feats = np.stack([np.ones_like(hla), hla/6, (cpra-30)/30,
                      (da-45)/20, (ra-45)/18, (dial-3)/4], axis=-1)   # (nD,nR,6)
    # TRUE benefit: linear part the predictor CAN fit + interactions it CANNOT
    beta = np.array([0.80, -0.22, -0.10, -0.06, -0.04, 0.05])
    lin = feats @ beta
    inter = (-0.35*(hla/6)*(cpra>80)                       # high-mismatch x sensitized
             - 0.25*(hla/6)*((cpra-30)/30)                 # mismatch x sensitization
             - 0.22*(hla/6)**2                             # curvature a linear model can't fit
             + 0.15*np.exp(-dial/4))                       # diminishing vintage effect
    B = np.clip(lin + inter, 0.05, 0.98)
    return feats, B

def oracle_benefit(B):                                     # best achievable (maximize)
    r, c = linear_sum_assignment(-B); return B[r, c].sum()
def realized(B, Bpred):                                    # true benefit of acting on Bpred
    r, c = linear_sum_assignment(-Bpred); return B[r, c].sum()
def regret(B, Bpred):
    Uo = oracle_benefit(B); return (Uo - realized(B, Bpred)) / Uo * 100

def design(feats):                                         # (nD,nR,p) -> (nD*nR,p)
    nD, nR, p = feats.shape; return feats.reshape(-1, p), (nD, nR)

def rich_feats(feats):
    # R5-C2 misspecification check: append the TARGET's own nonlinear transforms so the
    # linear learner becomes (near-)well-specified. feats cols: 1=hla/6, 2=(cpra-30)/30, 5=(dial-3)/4.
    h = feats[..., 1]; cs = feats[..., 2]; dl = feats[..., 5]
    cpra = cs * 30 + 30; dial = dl * 4 + 3
    extra = np.stack([h**2, h * (cpra > 80), h * cs, np.exp(-dial / 4)], axis=-1)   # the inter terms
    return np.concatenate([feats, extra], axis=-1)         # (nD,nR,10)

# ----- two-stage: ridge regression of observed benefit on base features -----
def fit_two_stage(pools, lam=1.0):
    X = np.vstack([design(f)[0] for f, _ in pools])
    y = np.concatenate([ (B + rng_obs.normal(0, 0.08, B.shape)).reshape(-1) for _, B in pools])
    A = X.T @ X + lam*np.eye(X.shape[1]); theta = np.linalg.solve(A, X.T @ y)
    return theta

# ----- SPO+: subgradient training with the optimizer in the loop -----
def spo_plus_train(pools, theta0, epochs=60, lr=0.004):
    theta = theta0.copy()
    # precompute oracle assignments (w*(c)) on TRUE cost c=-B for each pool
    oracles = []
    for f, B in pools:
        Phi, (nD, nR) = design(f)
        r, c = linear_sum_assignment(-B)                   # w*(c): minimize cost = -B
        w_true = np.zeros((nD, nR)); w_true[r, c] = 1.0
        oracles.append((Phi, (nD, nR), B, w_true))
    for ep in range(epochs):
        order = rng_spo.permutation(len(oracles)); g = np.zeros_like(theta)
        for idx in order:
            Phi, (nD, nR), B, w_true = oracles[idx]
            chat = (Phi @ theta).reshape(nD, nR)            # predicted benefit
            cost_hat = -chat                                # predicted cost
            c_true = -B
            # SPO+ inner solve at (2*chat_cost - c_true)
            spo_cost = 2*cost_hat - c_true
            r, cc = linear_sum_assignment(spo_cost)
            w_spo = np.zeros((nD, nR)); w_spo[r, cc] = 1.0
            # subgradient wrt predicted COST: 2(w*(c) - w*(2chat-c))
            sub_cost = 2*(w_true - w_spo)                   # (nD,nR)
            # chain to theta: cost_hat = -Phi theta  => d cost/d theta = -Phi
            g += -(Phi.T @ sub_cost.reshape(-1))
        theta -= lr * g / len(oracles)
    return theta

def eval_method(theta, pools):
    regs, mses = [], []
    for f, B in pools:
        Phi, (nD, nR) = design(f); pred = (Phi @ theta).reshape(nD, nR)
        regs.append(regret(B, pred)); mses.append(float(np.mean((pred - B)**2)))
    return np.array(regs), np.array(mses)

# ------------------------- experiment -------------------------
SEEDS = 10; N_TRAIN = 24; N_TEST = 24
all_two, all_spo, mse_two, mse_spo = [], [], [], []
all_two_r, all_spo_r = [], []                              # R5-C2: well-specified (rich) learner
for seed in range(SEEDS):
    rng = np.random.default_rng(1000+seed)
    global rng_obs, rng_spo; rng_obs = np.random.default_rng(2000+seed); rng_spo = np.random.default_rng(3000+seed)
    train = [make_pool(rng) for _ in range(N_TRAIN)]
    test  = [make_pool(rng) for _ in range(N_TEST)]
    th_ts = fit_two_stage(train)
    th_spo = spo_plus_train(train, theta0=th_ts)            # warm-start from two-stage
    r_ts, m_ts = eval_method(th_ts, test)
    r_sp, m_sp = eval_method(th_spo, test)
    all_two.append(r_ts); all_spo.append(r_sp); mse_two.append(m_ts); mse_spo.append(m_sp)
    # --- misspecification sensitivity: give the learner the target's nonlinear transforms ---
    rng_obs = np.random.default_rng(2000+seed); rng_spo = np.random.default_rng(3000+seed)  # same noise/order as base
    train_r = [(rich_feats(f), B) for f, B in train]; test_r = [(rich_feats(f), B) for f, B in test]
    th_ts_r = fit_two_stage(train_r); th_spo_r = spo_plus_train(train_r, theta0=th_ts_r)
    r_ts_r, _ = eval_method(th_ts_r, test_r); r_sp_r, _ = eval_method(th_spo_r, test_r)
    all_two_r.append(r_ts_r); all_spo_r.append(r_sp_r)
    print(f"seed {seed}: [base] two {r_ts.mean():.2f} spo {r_sp.mean():.2f} | [rich] two {r_ts_r.mean():.2f} spo {r_sp_r.mean():.2f}")

two = np.concatenate(all_two); spo = np.concatenate(all_spo)
diff = two - spo                                           # paired per held-out batch (unit = batch)
diff_by_seed = [all_two[s] - all_spo[s] for s in range(SEEDS)]   # nesting: batches within seed
# (i) naive per-batch bootstrap: treats all 240 batches as independent (ignores nesting)
rb = np.random.default_rng(42); boot = np.array([rb.choice(diff, len(diff), replace=True).mean() for _ in range(5000)])
ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
# (ii) seed-CLUSTERED two-level bootstrap: resample seeds (clusters), then batches within each.
# Correct unit of replication is the seed (one trained predictor pair), so batches are nested.
rc = np.random.default_rng(43); cboot = []
for _ in range(5000):
    seeds_b = rc.integers(0, SEEDS, SEEDS)
    vals = [rc.choice(diff_by_seed[s], len(diff_by_seed[s]), replace=True).mean() for s in seeds_b]
    cboot.append(np.mean(vals))
cci = (np.percentile(cboot, 2.5), np.percentile(cboot, 97.5))
# seed-level summary (mean reduction per seed) for a transparent between-seed view
seed_means = np.array([d.mean() for d in diff_by_seed])
# --- R5-C2 misspecification sensitivity: the well-specified (rich) learner ---
two_r = np.concatenate(all_two_r); spo_r = np.concatenate(all_spo_r)
diff_r_by_seed = [all_two_r[s] - all_spo_r[s] for s in range(SEEDS)]
rc2 = np.random.default_rng(44); cboot_r = []
for _ in range(5000):
    sb = rc2.integers(0, SEEDS, SEEDS)
    cboot_r.append(np.mean([rc2.choice(diff_r_by_seed[s], len(diff_r_by_seed[s]), replace=True).mean() for s in sb]))
cci_r = (np.percentile(cboot_r, 2.5), np.percentile(cboot_r, 97.5))
misspec = dict(
    base_feature_map="constant, |HLA mismatch|/6, cPRA, donor age, recipient age, dialysis vintage (6 terms)",
    rich_feature_map="base + (HLA/6)^2, (HLA/6)*1[cPRA>80], (HLA/6)*cPRA, exp(-vintage/4) (10 terms; the target's own nonlinear structure)",
    base_regret_two=round(float(two.mean()),3), base_regret_spo=round(float(spo.mean()),3), base_reduction=round(float(diff.mean()),3),
    rich_regret_two=round(float(two_r.mean()),3), rich_regret_spo=round(float(spo_r.mean()),3),
    rich_reduction=round(float((two_r-spo_r).mean()),3), rich_seed_clustered_95ci=[round(cci_r[0],3), round(cci_r[1],3)])
print("\nMisspecification check:", json.dumps(misspec, indent=2))
summary = dict(seeds=SEEDS, test_batches_per_seed=N_TEST, bootstrap_unit="held-out donor batch; seeds as clusters",
    regret_two_stage_mean=round(float(two.mean()),3), regret_two_stage_sd=round(float(two.std()),3),
    regret_spo_plus_mean=round(float(spo.mean()),3), regret_spo_plus_sd=round(float(spo.std()),3),
    paired_regret_reduction_mean=round(float(diff.mean()),3),
    naive_perbatch_95ci=[round(ci[0],3), round(ci[1],3)],
    seed_clustered_95ci=[round(cci[0],3), round(cci[1],3)],
    seed_mean_reduction_mean=round(float(seed_means.mean()),3),
    seed_mean_reduction_sd=round(float(seed_means.std(ddof=1)),3),
    n_seeds_positive=int((seed_means>0).sum()),
    mse_two_stage=round(float(np.concatenate(mse_two).mean()),4),
    mse_spo_plus=round(float(np.concatenate(mse_spo).mean()),4),
    misspecification_check=misspec)
json.dump(summary, open(f"{OUT}/metrics_dfl.json","w"), indent=2)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print(f"\nSPO+ reduces regret by {diff.mean():.3f} pts. Per-batch CI [{ci[0]:.3f},{ci[1]:.3f}]; "
      f"seed-clustered CI [{cci[0]:.3f},{cci[1]:.3f}] ({'sig' if cci[0]>0 else 'NOT sig'}); "
      f"{(seed_means>0).sum()}/{SEEDS} seeds positive.")

# figure: paired regret with CI
fig, ax = plt.subplots(figsize=(5.0,3.6))
ax.bar(["Two-stage\n(predict-then-optimize)","Decision-focused\n(SPO+)"],
       [two.mean(), spo.mean()], yerr=[two.std()/np.sqrt(len(two)), spo.std()/np.sqrt(len(spo))],
       capsize=4, color=["#d6604d","#1b7837"])
for i,v in enumerate([two.mean(), spo.mean()]): ax.text(i, v+0.05, f"{v:.2f}%", ha="center", fontsize=10)
ax.set_ylabel("Allocation regret (% vs oracle)")
ax.set_title(f"Decision-focused vs two-stage (matched capacity)\npaired reduction {diff.mean():.2f} pts, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dfl_regret.png", dpi=150)
print(f"\nSaved {OUT}/metrics_dfl.json and fig_dfl_regret.png")
