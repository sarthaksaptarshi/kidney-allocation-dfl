"""
Rebuild out/fig_survival.png as a TWO-COHORT real-data survival figure:
(a) Harrell C-index, proposed vs Cox, on the Mendeley cohort (31 events) and the independent public
    Jakarta cohort (86 events) for death-censored graft loss --- Cox leads at 31 events, the two tie
    at 86, confirming the gap is driven by event count, not method.
(b) Censoring-adjusted 5-year calibration on the larger (Jakarta) cohort.
Panel (a) C-indices are read from the leak-free metrics JSONs (results_survival for Mendeley,
results_star for Jakarta); panel (b) is computed from jakarta_clean.csv via the star_pipeline path.
"""
import numpy as np, pandas as pd, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from star_pipeline import discrete_survival, km_at

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
GREEN, BLUE = "#1b7837", "#2166ac"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9})

# ---- panel (b) data: Jakarta DCGL 5-year calibration (same pipeline code path) ----
df = pd.read_csv(os.path.join(HERE, "jakarta_clean.csv"))
t = df["PT_DC_y"].to_numpy(float); e = df["DC_Graft"].to_numpy(int)
num = ["rec_age60", "rec_overweight", "dial_gt24mo", "smoking", "diabetes", "cardiovascular",
       "don_age40", "related_donor", "don_overweight", "don_gfr100"]
cat = ["dialysis_type", "crossmatch"]
parts = [df[c].to_numpy(float) for c in num]
for c in cat:
    d = pd.get_dummies(df[c].astype(str).replace({"nan": np.nan}), prefix=c, dummy_na=False)
    for cc in d.columns: parts.append(d[cc].to_numpy(float))
X = np.column_stack(parts); keep = np.isfinite(t) & (t > 0); X, t, e = X[keep], t[keep], e[keep]
grid = [0, 1, 5, 10]; n = len(t); cum = np.zeros((n, len(grid)-1))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, e):   # review C5: fit preprocessing inside each fold
    imp = SimpleImputer(strategy="median").fit(X[tr]); sc = StandardScaler().fit(imp.transform(X[tr]))
    Xtr = sc.transform(imp.transform(X[tr])); Xte = sc.transform(imp.transform(X[te]))
    cum[te] = discrete_survival(Xtr, t[tr], e[tr], Xte, grid)
risk5 = cum[:, 1]                                  # cumulative incidence at the 5-year cut
tert = np.quantile(risk5, [1/3, 2/3]); grp = np.digitize(risk5, tert)
obs, pred = [], []
for g in range(3):
    m = grp == g
    if m.sum() > 5: obs.append(1 - km_at(t[m], e[m], 5.0)); pred.append(float(risk5[m].mean()))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.6))

# ---- panel (a): two-cohort C-index, proposed vs Cox (read from metrics JSONs; no hard-coding) ----
_mend = json.load(open(os.path.join(HERE, "results_survival", "metrics_survival.json")))   # Mendeley, leak-free
_jak  = json.load(open(os.path.join(HERE, "results_star", "metrics_star.json")))            # Jakarta,  leak-free
def _cc(d, key): return d[key]["c_index"], tuple(d[key]["c_ci"])
mp, mpc = _cc(_mend, "Proposed (discrete-time survival)"); mc, mcc = _cc(_mend, "Cox PH")
jp, jpc = _cc(_jak,  "Proposed (discrete-time survival)"); jc, jcc = _cc(_jak,  "Cox PH")
cohorts = ["Mendeley\n(31 events)", "Jakarta\n(86 events)"]
prop = [mp, jp]; prop_ci = [mpc, jpc]
cox  = [mc, jc]; cox_ci  = [mcc, jcc]
x = np.arange(2); w = 0.36
eP = [[prop[i]-prop_ci[i][0] for i in range(2)], [prop_ci[i][1]-prop[i] for i in range(2)]]
eC = [[cox[i]-cox_ci[i][0] for i in range(2)], [cox_ci[i][1]-cox[i] for i in range(2)]]
axA.bar(x - w/2, prop, w, yerr=eP, capsize=4, color=GREEN, label="Proposed survival")
axA.bar(x + w/2, cox, w, yerr=eC, capsize=4, color=BLUE, label="Cox PH")
axA.axhline(0.5, ls="--", color="gray", lw=1)
axA.set_xticks(x); axA.set_xticklabels(cohorts); axA.set_ylabel("Harrell C-index")
axA.set_ylim(0.45, 0.88); axA.set_title("(a) Proposed vs Cox across two real cohorts")
for xi in range(2):
    axA.text(xi - w/2, prop[xi]+ (prop_ci[xi][1]-prop[xi]) + 0.012, f"{prop[xi]:.3f}", ha="center", fontsize=7.5, color=GREEN)
    axA.text(xi + w/2, cox[xi] + (cox_ci[xi][1]-cox[xi]) + 0.012, f"{cox[xi]:.3f}", ha="center", fontsize=7.5, color=BLUE)
axA.legend(loc="upper right", fontsize=7.6, framealpha=0.9)

# ---- panel (b): Jakarta 5-year calibration ----
hi = max(0.5, (max(pred+obs)+0.05) if (pred and obs) else 0.5)
axB.plot([0, hi], [0, hi], "k--", lw=1, label="ideal")
axB.plot(pred, obs, "o-", color=GREEN, ms=7, label="risk tertiles")
axB.set_xlabel("Predicted 5-y graft failure"); axB.set_ylabel("KM-observed 5-y graft failure")
axB.set_title("(b) Calibration, Jakarta cohort (5-y)"); axB.legend(fontsize=8)

fig.tight_layout(w_pad=1.6)
fig.savefig(f"{OUT}/fig_survival.png", dpi=200)
print("saved", f"{OUT}/fig_survival.png  tertiles pred=", [round(p,3) for p in pred], "obs=", [round(o,3) for o in obs])
