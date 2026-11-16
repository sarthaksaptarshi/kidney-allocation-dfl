"""
Peer-review R2, Comment 3: censoring-aware synthetic comparison against Cox PH,
with a paired bootstrap CI for the 10-year difference, and an experimental
demonstration of the event-count crossover (proposed overtakes Cox as events grow).
Reproduces sim.py's synthetic cohort exactly (seed 20261116) so Table V stays consistent.
Rebuilds out/fig_auroc.png as a 2-panel figure (multi-horizon AUROC incl. Cox; crossover).
"""
import numpy as np, pandas as pd, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RNG = np.random.default_rng(20261116)
N = 18000

# ---- generative model (verbatim from sim.py) ----
def gen_population(n):
    rec_age = np.clip(RNG.normal(45, 13, n), 18, 75); don_age = np.clip(RNG.normal(42, 15, n), 5, 78)
    hla_mm = RNG.integers(0, 7, n).astype(float); base_cpra = RNG.beta(1.4, 6, n) * 100
    sens = RNG.binomial(1, 0.18, n).astype(bool); cpra = np.where(sens, RNG.uniform(55, 99, n), base_cpra)
    cit = np.clip(RNG.gamma(3.0, 3.2, n), 1, 40); dial_yr = np.clip(RNG.gamma(2.0, 1.6, n), 0, 18)
    bmi = np.clip(RNG.normal(24, 4.5, n), 15, 42); cyp3a5 = RNG.binomial(1, 0.35, n)
    ddtx = RNG.binomial(1, 0.62, n); diabetes = RNG.binomial(1, 0.28, n); female = RNG.binomial(1, 0.40, n)
    dsa = RNG.binomial(1, np.clip(0.03 + cpra / 300, 0, 0.7)).astype(float); tac_tr = np.clip(RNG.normal(7.5, 2.4, n), 2, 16)
    return pd.DataFrame(dict(rec_age=rec_age, don_age=don_age, hla_mm=hla_mm, cpra=cpra, cit=cit, dial_yr=dial_yr,
                             bmi=bmi, cyp3a5=cyp3a5, ddtx=ddtx, diabetes=diabetes, female=female, dsa=dsa, tac_trough=tac_tr))
df = gen_population(N)
feat_cols = ["rec_age", "don_age", "hla_mm", "cpra", "cit", "dial_yr", "bmi", "cyp3a5", "ddtx", "diabetes", "female", "dsa", "tac_trough"]
# R4-C2: offer-time feature set (measured tacrolimus trough excluded; CYP3A5 genotype kept).
# tac_trough stays in df for data generation (hazards) but is NOT a model input here.
feat_cols = [c for c in feat_cols if c != "tac_trough"]
u = RNG.normal(0, 1, N)
def z(col, c): return (df[col] - c)
def ctr(a): return a - a.mean()
tacU = (np.minimum(df.tac_trough - 5, 0) ** 2 + np.maximum(df.tac_trough - 11, 0) ** 2).to_numpy()
imm = ctr((0.42 * df.hla_mm + 0.030 * z("cpra", 18) + 1.4 * df.dsa + 0.045 * z("cit", 10) + 0.018 * z("don_age", 40)
           + 0.40 * df.ddtx + 0.55 * df.hla_mm * df.dsa + 0.6 * (df.cpra > 80)).to_numpy())
chr_ = ctr((0.10 * df.hla_mm + 0.012 * z("cpra", 18) + 0.045 * z("don_age", 40) + 0.12 * df.dial_yr + 0.25 * df.diabetes
            + 0.040 * z("bmi", 24) + 0.030 * z("cit", 10) + 0.10 * tacU + 0.20 * (df.dial_yr > 5) * df.diabetes).to_numpy())
def sig(x): return 1 / (1 + np.exp(-x))
h1 = sig(-2.95 + 0.9 * imm + 0.8 * u); h2 = sig(-1.70 + 0.8 * imm + 0.3 * chr_ + 0.9 * u); h3 = sig(-0.35 + 0.3 * imm + 0.9 * chr_ + 1.0 * u)
rej1 = RNG.binomial(1, h1).astype(bool); rej2 = (~rej1) & (RNG.binomial(1, h2).astype(bool)); rej3 = (~rej1) & (~rej2) & (RNG.binomial(1, h3).astype(bool))
event_time = np.full(N, 1e9)
event_time[rej1] = RNG.uniform(0.1, 1, rej1.sum()); event_time[rej2] = RNG.uniform(1, 12, rej2.sum()); event_time[rej3] = RNG.uniform(12, 120, rej3.sum())
chr_std = (chr_ - chr_.mean()) / chr_.std(); fu_scale = np.clip(54 * np.exp(-0.35 * chr_std), 8, 120)
fu = np.minimum(1 + RNG.exponential(fu_scale), 120)
y_1y_true = (event_time <= 12).astype(int)
def horizon_data(H):
    naive = ((event_time <= H) & (event_time <= fu)).astype(int)
    known = ((event_time <= np.minimum(H, fu)) | (fu >= H)); true = (event_time <= H).astype(int)
    return naive, known, true
H_map = {"1mo": 1, "1yr": 12, "10yr": 120}
X = df[feat_cols].to_numpy()
Xtr, Xte, itr, ite = train_test_split(X, np.arange(N), test_size=0.3, random_state=7, stratify=y_1y_true)

def gbm(): return GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.06, random_state=1)
def rf(): return RandomForestClassifier(n_estimators=160, max_depth=9, n_jobs=-1, random_state=1)
def lr(): return LogisticRegression(max_iter=500)

# ---- Breslow Cox (linear predictor is a monotone transform of F(H|x), so it suffices for AUROC/C) ----
def cox_fit(Xa, dur, ev, l2=1e-3):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Xs = (Xa - mu) / sd
    order = np.argsort(dur); Xo = Xs[order]; evo = ev[order].astype(bool)
    def obj(beta):
        eta = Xo @ beta; w = np.exp(eta - eta.max())
        rc = np.cumsum(w[::-1])[::-1]
        rcx = np.cumsum((w[:, None] * Xo)[::-1], axis=0)[::-1]
        ll = np.sum(eta[evo] - (np.log(rc[evo] + 1e-12) + eta.max())) - 0.5 * l2 * beta @ beta
        grad = np.sum(Xo[evo] - rcx[evo] / (rc[evo, None] + 1e-12), axis=0) - l2 * beta
        return -ll, -grad
    res = minimize(obj, np.zeros(Xa.shape[1]), jac=True, method="L-BFGS-B")
    return res.x, mu, sd
def cox_score(Xa, beta, mu, sd): return ((Xa - mu) / sd) @ beta

# ---- proposed discrete-time survival (verbatim grid from sim.py) ----
grid = [0, 1, 3, 6, 12, 24, 48, 84, 120]; edges = list(zip(grid[:-1], grid[1:]))
def fit_proposed(tr_idx, eval_X, n_est=180):
    hp = np.zeros((len(eval_X), len(edges)))
    for k, (s, e) in enumerate(edges):
        at = (event_time[tr_idx] > s) & (fu[tr_idx] > s)
        yk = ((event_time[tr_idx] > s) & (event_time[tr_idx] <= e) & (event_time[tr_idx] <= fu[tr_idx])).astype(int)
        if at.sum() > 20 and len(np.unique(yk[at])) > 1:
            m = GradientBoostingClassifier(n_estimators=n_est, max_depth=3, learning_rate=0.06, random_state=1).fit(X[tr_idx][at], yk[at])
            hp[:, k] = m.predict_proba(eval_X)[:, 1]
    return 1 - np.cumprod(1 - hp, axis=1)
idx_of = {1: grid.index(1) - 1, 12: grid.index(12) - 1, 120: grid.index(120) - 1}

# ============ (1) multi-horizon AUROC incl. Cox (known-status) ============
res = {}
for name, ctor in [("LogReg", lr), ("RandForest", rf), ("GBM_single", gbm)]:
    for h, H in H_map.items():
        naive, known, true = horizon_data(H); m = ctor().fit(Xtr, naive[itr]); p = m.predict_proba(Xte)[:, 1]; kn = known[ite]
        res[(name, h)] = roc_auc_score(true[ite][kn], p[kn])
# Cox on observed (duration, event) with event=rejection seen before censoring
dur_tr = np.minimum(event_time[itr], fu[itr]); ev_tr = (event_time[itr] <= fu[itr]).astype(int)
beta, mu, sd = cox_fit(Xtr, dur_tr, ev_tr)
cox_eta_te = cox_score(Xte, beta, mu, sd)
for h, H in H_map.items():
    naive, known, true = horizon_data(H); kn = known[ite]
    res[("Cox", h)] = roc_auc_score(true[ite][kn], cox_eta_te[kn])
cum_te = fit_proposed(itr, Xte)
prop_te = {h: cum_te[:, idx_of[H]] for h, H in H_map.items()}
for h, H in H_map.items():
    naive, known, true = horizon_data(H); kn = known[ite]
    res[("Proposed_MH", h)] = roc_auc_score(true[ite][kn], prop_te[h][kn])

tab = pd.DataFrame(index=["LogReg", "RandForest", "GBM_single", "Cox", "Proposed_MH"], columns=["1mo", "1yr", "10yr"], dtype=float)
for (m, h), v in res.items(): tab.loc[m, h] = v
print("AUROC by horizon (known-status), incl. Cox:\n", tab.round(3))

# ---- paired bootstrap CI at 10yr: Proposed - GBM_single, Proposed - Cox ----
naive, known10, true10 = horizon_data(120); kn10 = known10[ite]
gbm10 = gbm().fit(Xtr, naive[itr]); gbm10_te = gbm10.predict_proba(Xte)[:, 1]
yk = true10[ite][kn10]; pr = prop_te["10yr"][kn10]; gb = gbm10_te[kn10]; cx = cox_eta_te[kn10]
rngb = np.random.default_rng(11); dP_gbm, dP_cox = [], []
nkn = kn10.sum()
for _ in range(2000):
    b = rngb.integers(0, nkn, nkn)
    if len(np.unique(yk[b])) < 2: continue
    aP = roc_auc_score(yk[b], pr[b]); aG = roc_auc_score(yk[b], gb[b]); aC = roc_auc_score(yk[b], cx[b])
    dP_gbm.append(aP - aG); dP_cox.append(aP - aC)
ci_gbm = (float(np.percentile(dP_gbm, 2.5)), float(np.percentile(dP_gbm, 97.5)))
ci_cox = (float(np.percentile(dP_cox, 2.5)), float(np.percentile(dP_cox, 97.5)))
print(f"10y Proposed-GBM diff {np.mean(dP_gbm):.4f} CI {ci_gbm}")
print(f"10y Proposed-Cox diff {np.mean(dP_cox):.4f} CI {ci_cox}")

# ---- C3 full-label SENSITIVITY: evaluate on ALL test patients with counterfactual true labels ----
# (checks whether the known-status restriction biases the 10-y comparison)
yfull = (event_time[ite] <= 120).astype(int); nF = len(yfull)
prF, gbF, cxF = prop_te["10yr"], gbm10_te, cox_eta_te
auroc_full = {"Proposed": round(float(roc_auc_score(yfull, prF)), 3), "GBM": round(float(roc_auc_score(yfull, gbF)), 3), "Cox": round(float(roc_auc_score(yfull, cxF)), 3)}
rngf = np.random.default_rng(13); dF = []
for _ in range(2000):
    b = rngf.integers(0, nF, nF)
    if len(np.unique(yfull[b])) < 2: continue
    dF.append(roc_auc_score(yfull[b], prF[b]) - roc_auc_score(yfull[b], cxF[b]))
ciF = (round(float(np.percentile(dF, 2.5)), 4), round(float(np.percentile(dF, 97.5)), 4))
print(f"FULL-LABEL 10y AUROC {auroc_full}; Proposed-Cox {np.mean(dF):.4f} CI {ciF}  (known-status gave {np.mean(dP_cox):.4f} {ci_cox})")

# ============ (2) event-count crossover (same 10y known-status AUROC metric as Table V) ============
# Cox and the proposed model are retrained on training subsamples of growing event count,
# evaluated on the SAME fixed known-status test set, so the axis isolates events, not test noise.
yk10 = true10[ite][kn10]                      # fixed test, 10y known-status labels
pool = itr.copy(); erate = (event_time[pool] <= fu[pool]).mean()
targets = [30, 60, 120, 250, 500, 1000, 2000, 4000]
cross = {"events": [], "prop_mean": [], "prop_sd": [], "cox_mean": [], "cox_sd": []}
for tev in targets:
    n_sub = int(min(len(pool), tev / max(erate, 1e-6)))
    pm, cm, evs = [], [], []
    for s in range(3):
        sub = np.random.default_rng(100 + s).choice(pool, n_sub, replace=False)
        evs.append(int((event_time[sub] <= fu[sub]).sum()))
        cum = fit_proposed(sub, Xte, n_est=140); risk_p = cum[:, idx_of[120]]
        pm.append(roc_auc_score(yk10, risk_p[kn10]))
        b, m_, s_ = cox_fit(X[sub], np.minimum(event_time[sub], fu[sub]), (event_time[sub] <= fu[sub]).astype(int))
        cm.append(roc_auc_score(yk10, cox_score(Xte, b, m_, s_)[kn10]))
    cross["events"].append(int(np.mean(evs))); cross["prop_mean"].append(float(np.mean(pm))); cross["prop_sd"].append(float(np.std(pm)))
    cross["cox_mean"].append(float(np.mean(cm))); cross["cox_sd"].append(float(np.std(cm)))
    print(f"events~{int(np.mean(evs))}: proposed AUROC={np.mean(pm):.3f}  cox AUROC={np.mean(cm):.3f}")

# ============ figure: (a) multi-horizon AUROC incl Cox ; (b) crossover ============
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 3.8))
mods = ["GBM_single", "Cox", "Proposed_MH"]; labs = ["GBM (fixed-horizon)", "Cox PH", "Proposed (MH survival)"]
cols = ["#d6604d", "#4393c3", "#1b7837"]; hs = ["1mo", "1yr", "10yr"]; xh = np.arange(3); bw = 0.26
for k, (mm, cc) in enumerate(zip(mods, cols)):
    axa.bar(xh + (k - 1) * bw, [tab.loc[mm, h] for h in hs], bw, color=cc, label=labs[k])
axa.set_xticks(xh); axa.set_xticklabels(["1 month", "1 year", "10 year"]); axa.set_ylim(0.78, 0.90)
axa.set_ylabel("AUROC (known-status)"); axa.set_title("(a) Multi-horizon discrimination vs Cox")
axa.legend(fontsize=7, loc="upper left")
axa.annotate(f"10y: Proposed$-$Cox\n$\\Delta$={np.mean(dP_cox):.3f} [{ci_cox[0]:.3f}, {ci_cox[1]:.3f}]",
             xy=(2, 0.852), fontsize=6.8, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f7f7f7", ec="gray", lw=0.5))
axb.errorbar(cross["events"], cross["prop_mean"], yerr=cross["prop_sd"], fmt="o-", color="#1b7837", capsize=3, label="Proposed (MH survival)")
axb.errorbar(cross["events"], cross["cox_mean"], yerr=cross["cox_sd"], fmt="s-", color="#4393c3", capsize=3, label="Cox PH")
axb.set_xscale("log"); axb.set_xlabel("Training events (log scale)"); axb.set_ylabel("10-year AUROC (fixed test)")
axb.set_title("(b) Event-count crossover"); axb.legend(fontsize=7, loc="lower right"); axb.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_auroc.png", dpi=170); plt.close(fig)
print("wrote out/fig_auroc.png")

out = dict(auroc_table=tab.round(3).to_dict(), prop_minus_gbm_10y=dict(diff=round(float(np.mean(dP_gbm)), 4), ci=[round(ci_gbm[0], 4), round(ci_gbm[1], 4)]),
           prop_minus_cox_10y=dict(diff=round(float(np.mean(dP_cox)), 4), ci=[round(ci_cox[0], 4), round(ci_cox[1], 4)]),
           full_label_10y=dict(auroc=auroc_full, prop_minus_cox_diff=round(float(np.mean(dF)), 4), ci=list(ciF)),
           crossover=cross)
json.dump(out, open(f"{OUT}/synthetic_cox_metrics.json", "w"), indent=2)
print("saved synthetic_cox_metrics.json")
