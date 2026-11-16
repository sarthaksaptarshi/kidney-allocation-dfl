"""
Peer-review R2, Comment 2: implement and benchmark the NOTTO additive priority
explicitly (dialysis-month credit + group-specific bonuses), not a single shared
union bonus. Resolves the dialysis-sign conflict: the medical benefit proxy favors
SHORT vintage (better graft survival), while the NOTTO priority credit rewards LONG
waits -- the two oppose, and that opposition is the equity-utility tension.
Reports group denominators and variability across multiple random pools, across
three donor-scarcity regimes. Rebuilds out/fig_fairness.png (2 panels) + CSV.
"""
import numpy as np, pandas as pd, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
nR = 300

def survival_benefit(d_age, d_hla, r_age, r_hla, r_cpra, r_dial):
    hla = abs(d_hla - r_hla)
    base = 0.90 - 0.045 * hla - 0.0018 * r_cpra - 0.003 * max(0, d_age - 40) - 0.002 * max(0, r_age - 55) + 0.01 * np.exp(-r_dial / 6)
    return float(np.clip(base, 0.05, 0.98))

ABO = ["O", "B", "A", "AB"]; ABO_P = [0.37, 0.33, 0.23, 0.07]   # approx Indian distribution
def make_pool(seed):
    rng = np.random.default_rng(seed)
    R = pd.DataFrame(dict(
        age=np.clip(rng.normal(48, 12, nR), 18, 75), hla=rng.integers(0, 7, nR),
        cpra=np.where(rng.binomial(1, 0.18, nR).astype(bool), rng.uniform(55, 99, nR), rng.beta(1.4, 6, nR) * 100),
        dial=np.clip(rng.gamma(2, 1.6, nR), 0, 18), female=rng.binomial(1, 0.40, nR), pediatric=rng.binomial(1, 0.08, nR),
        abo=rng.choice(ABO, nR, p=ABO_P)))
    return R

def donors(nD, seed):
    rng = np.random.default_rng(seed + 9991)
    return pd.DataFrame(dict(age=np.clip(rng.normal(44, 15, nD), 5, 78), hla=rng.integers(0, 7, nD),
                             abo=rng.choice(ABO, nD, p=ABO_P)))

def abo_ok(d, r):  # donor d -> recipient r organ-ABO compatibility
    return (d == "O") | (d == r) | (r == "AB")

def feasible_mask(D, R, seed):
    """Edge feasibility: ABO-compatible AND crossmatch-negative. Crossmatch-positive
    probability rises with recipient cPRA, so highly sensitized recipients lose many edges."""
    rng = np.random.default_rng(seed + 555); nD = len(D)
    M = np.zeros((nD, nR), dtype=bool)
    xm_pos = rng.random((nD, nR)) < (0.02 + 0.9 * (R.cpra.to_numpy()[None, :] / 100.0))  # per-pair XM+
    for j in range(nD):
        M[j, :] = abo_ok(D.abo[j], R.abo.to_numpy()) & (~xm_pos[j, :])
    return M

def benefit_matrix(D, R):
    nD = len(D); B = np.zeros((nD, nR))
    for j in range(nD):
        for i in range(nR):
            B[j, i] = survival_benefit(D.age[j], D.hla[j], R.age[i], R.hla[i], R.cpra[i], R.dial[i])
    return B

def assign(B, bonus=None):
    nD = B.shape[0]; cost = -B.copy()
    if bonus is not None: cost = cost - bonus[None, :]
    n = max(nD, nR); C = np.zeros((n, n)); C[:nD, :nR] = cost
    r, c = linear_sum_assignment(C)
    return [(ri, ci) for ri, ci in zip(r, c) if ri < nD and ci < nR]

def assign_feas(B, feas, bonus=None):
    """Assignment restricted to the feasible (ABO-compatible, crossmatch-negative) edge set."""
    nD = B.shape[0]; BIG = 1e6; cost = -B.copy()
    if bonus is not None: cost = cost - bonus[None, :]
    cost[~feas] = BIG
    n = max(nD, nR); C = np.full((n, n), BIG); C[:nD, :nR] = cost
    r, c = linear_sum_assignment(C)
    return [(ri, ci) for ri, ci in zip(r, c) if ri < nD and ci < nR and feas[ri, ci]]

def util(B, m): return sum(B[j, i] for j, i in m)
def access(m, mask):
    ch = [i for _, i in m]; return float(mask[ch].sum() / mask.sum()) if mask.sum() > 0 else np.nan

def notto_priority(R):
    """NOTTO additive priority POINTS (policy structure as cited): a dialysis-time credit
    that rewards LONGER waits, plus group bonuses with automatic pediatric priority largest.
    Weights are balanced so the vintage credit and the group bonuses are of comparable scale
    (the raw per-month figure otherwise swamps the bonuses); exact points are illustrative and
    tunable. Normalized to [0,1] so a single weight lambda scales the whole policy."""
    fem = R.female.to_numpy().astype(bool); sens = (R.cpra.to_numpy() > 80); ped = R.pediatric.to_numpy().astype(bool)
    pts = 0.3 * R.dial.to_numpy() + 2.0 * fem + 5.0 * ped + 4.0 * sens   # vintage credit (yrs) + group bonuses
    pts = pts / pts.max()
    return pts, fem, sens, ped

scenarios = [("Scarce (0.50)", 150), ("Baseline (0.67)", 200), ("Abundant (0.87)", 260)]
lam_grid = np.linspace(0, 0.40, 16); lam_star = 0.24
POOLS = 8
rows, frontiers, acc_stats, pof_stats, denoms, dial_effect = [], {}, {}, {}, {}, {}

for name, nD in scenarios:
    front_pacc = np.zeros((POOLS, len(lam_grid))); front_ret = np.zeros((POOLS, len(lam_grid)))
    accU = {"female": [], "sensitized": [], "pediatric": []}; accF = {"female": [], "sensitized": [], "pediatric": []}
    pofs, dial_u, dial_f, dens = [], [], [], {"female": [], "sensitized": [], "pediatric": []}
    for p in range(POOLS):
        R = make_pool(1000 + p); D = donors(nD, 1000 + p); B = benefit_matrix(D, R)
        pts, fem, sens, ped = notto_priority(R)
        for g, mk in [("female", fem), ("sensitized", sens), ("pediatric", ped)]: dens[g].append(int(mk.sum()))
        base = assign(B); U0 = util(B, base)
        for lk, lam in enumerate(lam_grid):
            m = assign(B, lam * pts)
            front_pacc[p, lk] = np.nanmean([access(m, fem), access(m, sens), access(m, ped)])
            front_ret[p, lk] = util(B, m) / U0 * 100
        mstar = assign(B, lam_star * pts)
        for g, mk in [("female", fem), ("sensitized", sens), ("pediatric", ped)]:
            accU[g].append(access(base, mk)); accF[g].append(access(mstar, mk))
        pofs.append((U0 - util(B, mstar)) / U0 * 100)
        dial_u.append(np.mean([R.dial.to_numpy()[i] for _, i in base])); dial_f.append(np.mean([R.dial.to_numpy()[i] for _, i in mstar]))
    frontiers[name] = (front_pacc.mean(0), front_ret.mean(0))
    acc_stats[name] = {g: (float(np.nanmean(accU[g])), float(np.nanstd(accU[g])), float(np.nanmean(accF[g])), float(np.nanstd(accF[g]))) for g in accU}
    pof_stats[name] = (float(np.mean(pofs)), float(np.std(pofs)))
    denoms[name] = {g: (float(np.mean(v)), float(np.std(v))) for g, v in dens.items()}
    dial_effect[name] = (float(np.mean(dial_u)), float(np.mean(dial_f)))
    rows.append(dict(scenario=name, nD=nD, pof_mean=round(np.mean(pofs), 2), pof_sd=round(np.std(pofs), 2),
                     dial_util=round(np.mean(dial_u), 2), dial_notto=round(np.mean(dial_f), 2),
                     **{f"{g}_denom": round(np.mean(v), 1) for g, v in dens.items()}))

# ---- C5/C4 compatibility-filtered sensitivity (baseline ratio, same 8 pools) ----
fk = {"female": ([], []), "sensitized": ([], []), "pediatric": ([], [])}; fpof = []; feas_frac = []
n_match_unf, n_match_fil = [], []
for p in range(POOLS):
    R = make_pool(1000 + p); D = donors(200, 1000 + p); B = benefit_matrix(D, R)
    feas = feasible_mask(D, R, 1000 + p); feas_frac.append(float(feas.mean()))
    pts, fem, sens, ped = notto_priority(R)
    base = assign_feas(B, feas); U0 = util(B, base); mstar = assign_feas(B, feas, lam_star * pts)
    n_match_unf.append(len(assign(B, lam_star * pts)))   # matches made, fully connected
    n_match_fil.append(len(mstar))                        # matches made, feasibility-filtered
    for g, mk in [("female", fem), ("sensitized", sens), ("pediatric", ped)]:
        fk[g][0].append(access(base, mk)); fk[g][1].append(access(mstar, mk))
    fpof.append((U0 - util(B, mstar)) / U0 * 100 if U0 > 0 else np.nan)
filtered = {g: (round(float(np.nanmean(v[0])), 3), round(float(np.nanstd(v[0])), 3), round(float(np.nanmean(v[1])), 3), round(float(np.nanstd(v[1])), 3)) for g, v in fk.items()}
filtered_pof = (round(float(np.nanmean(fpof)), 2), round(float(np.nanstd(fpof)), 2)); feas_frac = round(float(np.mean(feas_frac)), 3)
matches = dict(unfiltered=round(float(np.mean(n_match_unf)), 1), filtered=round(float(np.mean(n_match_fil)), 1), donors=200)
print("FILTERED baseline access u->notto:", filtered, "PoF", filtered_pof, "feas-frac", feas_frac, "matches", matches)

summary = dict(lam_star=lam_star, pools=POOLS, price_of_fairness=pof_stats, access=acc_stats, group_denominators=denoms,
               dialysis_vintage_util_vs_notto=dial_effect, variation="mean +/- sd across 8 independent pools",
               compatibility_filtered_baseline=dict(access=filtered, price_of_fairness=filtered_pof, feasible_edge_fraction=feas_frac, matches=matches))
pd.DataFrame(rows).to_csv(f"{OUT}/fairness_multiscenario.csv", index=False)
json.dump(summary, open(f"{OUT}/fairness_summary.json", "w"), indent=2, default=float)
print(json.dumps(summary["access"]["Baseline (0.67)"], indent=2, default=float))
print("PoF:", {k: v for k, v in pof_stats.items()})

# ---- figure: (a) frontier vs scarcity ; (b) access with variability at baseline ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.7))
colors = {"Scarce (0.50)": "#b2182b", "Baseline (0.67)": "#2166ac", "Abundant (0.87)": "#1b7837"}
mark = {"Scarce (0.50)": "o", "Baseline (0.67)": "s", "Abundant (0.87)": "^"}
for name in [s[0] for s in scenarios]:
    pacc, ret = frontiers[name]; o = np.argsort(pacc)
    axa.plot(pacc[o] * 100, ret[o], mark[name] + "-", color=colors[name], ms=4, lw=1.6, label=name)
axa.set_xlabel("Mean protected-group access (%)"); axa.set_ylabel("Utility retained (%)")
axa.set_title("(a) NOTTO priority: equity–utility vs scarcity"); axa.legend(fontsize=7, title="Donor:recipient ratio", title_fontsize=7); axa.grid(alpha=0.25)

bl = acc_stats["Baseline (0.67)"]; den = denoms["Baseline (0.67)"]
grps = ["Female", "Sensitized", "Pediatric"]; keys = ["female", "sensitized", "pediatric"]
u = [bl[k][0] for k in keys]; us = [bl[k][1] for k in keys]; f = [bl[k][2] for k in keys]; fs = [bl[k][3] for k in keys]
x = np.arange(3); bw = 0.38
axb.bar(x - bw / 2, u, bw, yerr=us, capsize=3, label="Utility-only", color="#bdbdbd")
axb.bar(x + bw / 2, f, bw, yerr=fs, capsize=3, label="NOTTO priority", color="#2166ac")
axb.set_xticks(x); axb.set_xticklabels([f"{g}\n(n≈{int(den[k][0])})" for g, k in zip(grps, keys)], fontsize=8)
axb.set_ylabel("Access rate"); axb.set_ylim(0, 1.18)
axb.set_title(f"(b) Access at baseline (PoF {pof_stats['Baseline (0.67)'][0]:.1f}%)"); axb.legend(fontsize=7, loc="upper center", framealpha=0.95)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_fairness.png", dpi=170); plt.close(fig)
print("wrote out/fig_fairness.png")
