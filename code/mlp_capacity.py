"""
#4 (reviewer: extend DFL capacity beyond linear ridge). Repeat the two extreme factorial cells
--- unconstrained (feasibility off, priority off) and fully integrated (feasibility on, priority on)
--- with a compact 1-hidden-layer MLP (16 tanh units) trained two ways:
  * two-stage: minimize MSE to the benefit target;
  * decision-focused (SPO+): optimizer-in-the-loop subgradient 2(w_spo - w_true) on the predicted
    benefit, backpropagated through the MLP by hand (no autograd framework needed).
If the constraint-absorption pattern (SPO+ helps unconstrained, ties when integrated) holds for the
MLP as it does for linear ridge, the central finding is not an artifact of restricted capacity.
Outputs: results_integration/metrics_mlp.json
"""
import numpy as np, json, os, copy
from scipy.optimize import linear_sum_assignment
import integration_experiment as ie

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_integration")
os.makedirs(OUT, exist_ok=True)
BIG = 1e6; H = 16

def init_mlp(p, rng):
    return dict(W1=rng.normal(0, 0.3, (p, H)), b1=np.zeros(H),
                W2=rng.normal(0, 0.3, (H, 1)), b2=np.zeros(1))

def fwd(mlp, Phi):
    z1 = Phi @ mlp["W1"] + mlp["b1"]; a1 = np.tanh(z1)
    pred = (a1 @ mlp["W2"] + mlp["b2"]).ravel()
    return pred, (a1,)

def backward(mlp, Phi, cache, dpred):
    (a1,) = cache; dp = dpred.reshape(-1, 1)
    dW2 = a1.T @ dp; db2 = dp.sum(0)
    dz1 = (dp @ mlp["W2"].T) * (1 - a1**2)
    return dict(W1=Phi.T @ dz1, b1=dz1.sum(0), W2=dW2, b2=db2)

def step(mlp, g, lr):
    for k in mlp: mlp[k] -= lr * g[k]

def train_two_stage(pools, rng, epochs, lr):
    p = pools[0]["feats"].shape[-1]; mlp = init_mlp(p, rng)
    data = [(ie.design(pp["feats"])[0], pp["B"].reshape(-1)) for pp in pools]
    for _ in range(epochs):
        for Phi, y in data:
            pred, cache = fwd(mlp, Phi)
            step(mlp, backward(mlp, Phi, cache, 2*(pred - y)/len(y)), lr)
    return mlp

def train_spo(pools, mlp0, lam, epochs, lr):
    mlp = copy.deepcopy(mlp0); pre = []
    for pp in pools:
        Phi, (nD, nR) = ie.design(pp["feats"]); s, feas = pp["s"], pp["feas"]
        w_true = np.zeros((nD, nR))
        for j, i in ie.fair_feasible_assign(pp["B"], s, feas, lam): w_true[j, i] = 1.0
        pre.append((Phi, (nD, nR), pp["B"], s, feas, w_true))
    for _ in range(epochs):
        for Phi, (nD, nR), B, s, feas, w_true in pre:
            pred, cache = fwd(mlp, Phi); chat = pred.reshape(nD, nR)
            cost_hat = -(chat + lam*s[None, :]); c_true = -(B + lam*s[None, :])
            spo_cost = ie._pad(np.where(feas, 2*cost_hat - c_true, BIG), nD, nR)
            rr, cc = linear_sum_assignment(spo_cost); w_spo = np.zeros((nD, nR))
            for ri, ci in zip(rr, cc):
                if ri < nD and ci < nR and feas[ri, ci]: w_spo[ri, ci] = 1.0
            dpred = (2*(w_spo - w_true)).reshape(-1)
            step(mlp, backward(mlp, Phi, cache, dpred), lr)
    return mlp

def eval_regret(mlp, pools, lam):
    return np.array([ie.regret(pp, fwd(mlp, ie.design(pp["feats"])[0])[0].reshape(pp["B"].shape), lam) for pp in pools])

def run_cell(use_feas, use_prio, SEEDS=10, N_TRAIN=24, N_TEST=24,
             ts_epochs=700, ts_lr=0.03, spo_epochs=150, spo_lr=0.02):
    # Fair protocol: train BOTH learners to convergence. Two-stage MSE has no optimizer in its
    # loop, so it is cheap and gets a large budget; SPO+ (Hungarian each step) gets enough to
    # converge in decision space. A well-trained MLP can represent the benefit, so the honest
    # comparison is between two converged learners, not an underfit baseline vs SPO+.
    lam = 0.12 if use_prio else 0.0; two_all, spo_all = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        train = [ie.make_pool(rng, 40, 55) for _ in range(N_TRAIN)]
        test  = [ie.make_pool(rng, 40, 55) for _ in range(N_TEST)]
        if not use_feas:
            for p in train + test: p["feas"] = np.ones_like(p["feas"], dtype=bool)
        ts = train_two_stage(train, np.random.default_rng(5000 + seed), ts_epochs, ts_lr)
        sp = train_spo(train, ts, lam, spo_epochs, spo_lr)
        two_all.append(eval_regret(ts, test, lam)); spo_all.append(eval_regret(sp, test, lam))
    dbs = [two_all[s] - spo_all[s] for s in range(SEEDS)]
    rng = np.random.default_rng(43); v = []
    for _ in range(4000):
        sb = rng.integers(0, SEEDS, SEEDS)
        v.append(np.mean([rng.choice(dbs[s], len(dbs[s]), replace=True).mean() for s in sb]))
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    two = np.concatenate(two_all); spo = np.concatenate(spo_all)
    return dict(learner="MLP (16 tanh)", feasibility=bool(use_feas), priority=bool(use_prio),
                regret_two_stage=round(float(two.mean()), 3), regret_spo_plus=round(float(spo.mean()), 3),
                reduction=round(float(np.mean([d.mean() for d in dbs])), 3),
                seed_clustered_95ci=[round(lo, 3), round(hi, 3)],
                n_seeds_positive=int(sum(d.mean() > 0 for d in dbs)))

cells = []
for uf, up, label in [(False, False, "unconstrained"), (True, True, "integrated (fair+feasible)")]:
    c = run_cell(uf, up); c["setting"] = label; cells.append(c)
    ci = c["seed_clustered_95ci"]; v = "SIG+" if ci[0] > 0 else ("SIG-" if ci[1] < 0 else "tie")
    print(f"MLP {label:28s}: two {c['regret_two_stage']:.3f}  spo {c['regret_spo_plus']:.3f}  "
          f"reduction {c['reduction']:+.3f}  CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {v}")

# Convergence trajectory (unconstrained, 3 seeds): the two-stage MLP's regret falls monotonically
# with training budget toward SPO+'s level, so the gap is an optimization-speed effect, not a
# fundamental decision-focused advantage of SPO+ for the flexible learner.
traj = []
for ep in [80, 700, 2000, 4000]:
    regs = []
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        train = [ie.make_pool(rng, 40, 55) for _ in range(24)]; test = [ie.make_pool(rng, 40, 55) for _ in range(24)]
        for p in train + test: p["feas"] = np.ones_like(p["feas"], dtype=bool)
        ts = train_two_stage(train, np.random.default_rng(5000 + seed), ep, 0.03)
        regs.append(float(eval_regret(ts, test, 0.0).mean()))
    traj.append(dict(epochs=ep, two_stage_regret=round(float(np.mean(regs)), 3)))
    print(f"  two-stage MLP epochs={ep}: regret {np.mean(regs):.3f}")

out = dict(cells=cells, two_stage_convergence=traj,
           spo_plus_unconstrained_regret=cells[0]["regret_spo_plus"])
json.dump(out, open(f"{OUT}/metrics_mlp.json", "w"), indent=2)
print("saved", f"{OUT}/metrics_mlp.json")
