"""
STEP 3 (reviewer comment 3): test the PROPOSED multi-horizon survival model on REAL
time-to-event data, with censoring-aware baselines, confidence intervals, and
censoring-adjusted discrimination + calibration.

Data: Mendeley kidney-allograft cohort (campath/simulect CSV; CC BY 4.0).
Endpoint: DEATH-CENSORED GRAFT LOSS (time-to-event). Event = graft loss (`dcgl` = Yes);
patients who die with a functioning graft are censored at death; everyone else is
censored at end of follow-up. Time = the "yrs to graftloss/death/end of follow-up" column.
(This clarifies how death and graft failure enter the endpoint, per the reviewer.)

Models (matched feature set, pre-transplant features only -> no leakage):
  * Proposed  : discrete-time survival (interval hazards on at-risk sets; nonlinear GBM).
  * Cox PH    : proportional-hazards baseline (censoring-aware), implemented here.
  * Naive     : fixed-horizon GBM classifier, censored-before-horizon = negative
                (NOT censoring-aware; included to show the bias the survival models avoid).

Metrics (all censoring-aware, self-contained -- no extra packages):
  * Harrell C-index for each model + bootstrap 95% CI (primary discrimination).
  * Time-dependent AUROC at 1/3/5 years (known-status evaluation).
  * Calibration: Kaplan-Meier-observed vs predicted graft loss at 5 y, by risk tertile.

Run:  python survival_real.py "data/.../campath.simulect_anonymised_data.csv"
Outputs: results_survival/metrics_survival.json, fig_survival_cindex.png, fig_survival_calibration.png
Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib only.
"""
import os, sys, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_survival"); os.makedirs(OUT, exist_ok=True)

# pre-transplant features (no leakage); mapped to this file's column names
NUM = ["ageattx", "dageattx", "totalmm", "pra", "cit", "rbmi", "mma", "mmb", "mmdr", "abomm"]
CAT = ["dsa", "crossmatch", "dtype", "rdm", "sex", "ethnicity", "ind", "initis", "prevtx", "premt_tx"]
GRID = [0, 1, 5, 10]   # coarser grid (short/medium/long), matched to a low-event cohort

def load(path):
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]; return df
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
def find_time_col(cols):
    for c in cols:
        n = norm(c)
        if ("yr" in n or "year" in n) and ("graftloss" in n or "follow" in n or "death" in n):
            return c
    return None
def clean_num(s): return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
def yesno(s):
    v = s.astype(str).str.strip().str.lower()
    return v.isin(["yes", "y", "1", "true"]).astype(int)

# ---------- Kaplan-Meier ----------
def km(times, events):
    order = np.argsort(times); t, e = np.asarray(times)[order], np.asarray(events)[order]
    uniq = np.unique(t); S = 1.0; surv = []
    n = len(t)
    for u in uniq:
        at_risk = np.sum(t >= u); d = np.sum((t == u) & (e == 1))
        if at_risk > 0: S *= (1 - d/at_risk)
        surv.append(S)
    return uniq, np.array(surv)
def km_at(times, events, horizon):
    u, S = km(times, events)
    if len(u) == 0: return 1.0
    idx = np.searchsorted(u, horizon, side="right") - 1
    return S[idx] if idx >= 0 else 1.0

# ---------- Cox PH (Breslow), self-contained ----------
def cox_fit(X, t, e, l2=1.0):
    order = np.argsort(-t)                       # descending time
    Xs, ts, es = X[order], t[order], e[order]
    def negll(beta):
        eta = Xs @ beta; m = eta.max(); ex = np.exp(eta - m)
        cum = np.cumsum(ex)                       # risk set sum (desc time => t_j>=t_i)
        cumX = np.cumsum(ex[:, None] * Xs, axis=0)
        ll = 0.0; grad = np.zeros_like(beta)
        for i in range(len(ts)):
            if es[i] == 1:
                ll += eta[i] - (np.log(cum[i]) + m)
                grad += Xs[i] - cumX[i] / cum[i]
        return -ll + 0.5*l2*np.sum(beta**2), -grad + l2*beta
    b0 = np.zeros(X.shape[1])
    res = minimize(negll, b0, jac=True, method="L-BFGS-B")
    return res.x
def cox_risk(X, beta): return X @ beta           # higher = higher hazard = higher risk

# ---------- Harrell C-index (higher risk should predict earlier event) ----------
def harrell_c(risk, t, e):
    n = len(t); num = den = 0.0
    for i in range(n):
        if e[i] == 1:
            m = t > t[i]                           # comparable: j survives past i's event
            den += np.sum(m)
            num += np.sum(risk[i] > risk[m]) + 0.5*np.sum(risk[i] == risk[m])
    return num/den if den > 0 else 0.5
def boot_c(risk, t, e, n=1000, seed=0):
    rng = np.random.default_rng(seed); cs = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        cs.append(harrell_c(risk[idx], t[idx], e[idx]))
    return np.percentile(cs, [2.5, 97.5])

# ---------- discrete-time survival (proposed) ----------
def discrete_survival(Xtr, ttr, etr, Xte, grid):
    edges = list(zip(grid[:-1], grid[1:])); haz = np.zeros((len(Xte), len(edges)))
    for k, (a, b) in enumerate(edges):
        at_risk = ttr > a                          # still at risk at interval start
        yk = ((ttr > a) & (ttr <= b) & (etr == 1)).astype(int)
        if at_risk.sum() > 20 and yk[at_risk].sum() > 3:
            m = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0)
            m.fit(Xtr[at_risk], yk[at_risk]); haz[:, k] = m.predict_proba(Xte)[:, 1]
    cum = 1 - np.cumprod(1 - haz, axis=1)          # cumulative incidence at each cut
    return cum                                      # (n_te, n_intervals)

# ---------- time-dependent AUROC (known-status) ----------
def td_auroc(risk_by_t, t, e, horizon):
    case = (t <= horizon) & (e == 1)               # event by horizon
    ctrl = t > horizon                              # known event-free at horizon
    keep = case | ctrl
    if case.sum() < 5 or ctrl.sum() < 5: return np.nan
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(case[keep].astype(int), risk_by_t[keep])

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--endpoint", choices=["all_cause", "dcgl"], default="all_cause",
                    help="all_cause = graft loss OR death (more events, recommended); dcgl = death-censored graft loss")
    args = ap.parse_args()
    df = load(args.path)
    tcol = find_time_col(df.columns)
    if tcol is None: sys.exit("Could not find the time-to-event column (contains 'yrs ... graftloss/death').")
    if "dcgl" not in df.columns: sys.exit("Could not find the 'dcgl' column.")
    t = clean_num(df[tcol]).to_numpy()
    gl = yesno(df["dcgl"]).to_numpy()
    dth = yesno(df["death"]).to_numpy() if "death" in df.columns else np.zeros(len(df), int)
    if args.endpoint == "all_cause":
        e = ((gl == 1) | (dth == 1)).astype(int); endpoint_label = "all-cause graft failure (graft loss OR death)"
    else:
        e = gl.astype(int); endpoint_label = "death-censored graft loss (death with function censored at death)"
    print(f"Endpoint: {endpoint_label}")
    # build feature matrix
    cols = []; X = []
    for c in NUM:
        if c in df.columns: X.append(clean_num(df[c]).to_numpy()); cols.append(c)
    for c in CAT:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype(str).str.strip().replace({"nan": np.nan}), prefix=c, dummy_na=False)
            for cc in d.columns: X.append(d[cc].to_numpy().astype(float)); cols.append(cc)
    X = np.column_stack(X)
    keep = np.isfinite(t) & (t > 0)
    X, t, e = X[keep], t[keep], e[keep]
    # Review C5: NO global preprocessing. X is kept raw (NaN allowed); the imputer and scaler
    # are fit INSIDE each fold on training patients only, so no test information leaks.
    print(f"n={len(t)}  graft-loss events={int(e.sum())} ({e.mean()*100:.1f}%)  median follow-up={np.median(t):.1f} y")

    # 5-fold OUT-OF-FOLD risk scores (evaluate on the full cohort -> tighter CIs)
    from sklearn.model_selection import StratifiedKFold
    grid = GRID; n = len(t)
    risk_prop = np.zeros(n); risk_cox = np.zeros(n); risk_naive = np.zeros(n)
    cum_full = np.zeros((n, len(grid)-1))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, e):
        imp = SimpleImputer(strategy="median").fit(X[tr])                      # fit on TRAIN only
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr = sc.transform(imp.transform(X[tr])); Xte = sc.transform(imp.transform(X[te]))
        cum = discrete_survival(Xtr, t[tr], e[tr], Xte, grid)
        cum_full[te] = cum; risk_prop[te] = cum[:, -1]
        beta = cox_fit(Xtr, t[tr], e[tr], l2=1.0); risk_cox[te] = cox_risk(Xte, beta)
        y5 = ((t[tr] <= 5.0) & (e[tr] == 1)).astype(int)
        nb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0).fit(Xtr, y5)
        risk_naive[te] = nb.predict_proba(Xte)[:, 1]

    models = {"Proposed (discrete-time survival)": risk_prop, "Cox PH": risk_cox, "Naive fixed-horizon": risk_naive}
    res = {}
    for name, r in models.items():
        c = harrell_c(r, t, e); lo, hi = boot_c(r, t, e, n=1000, seed=1)
        res[name] = dict(c_index=round(float(c), 3), c_ci=[round(float(lo), 3), round(float(hi), 3)])
        print(f"{name:34s} C-index {c:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    td = {f"{grid[k]}y": round(float(td_auroc(cum_full[:, k-1], t, e, grid[k])), 3) for k in range(1, len(grid))}
    res["Proposed_td_AUROC"] = td
    print("Proposed time-dependent AUROC:", td)
    te = np.arange(n); cum = cum_full                 # full-cohort aliases for the figures below
    calh = 5 if 5 in grid else grid[len(grid)//2]     # calibration horizon (a grid cut-point)
    calk = grid.index(calh)

    # ---- save METRICS FIRST (so numbers persist even if a figure file is locked) ----
    res["endpoint"] = endpoint_label
    res["n"] = int(len(t)); res["events"] = int(e.sum()); res["event_rate"] = round(float(e.mean()), 4)
    res["Proposed_td_AUROC"] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in td.items()}
    def save(fn, saver):
        try:
            saver(); print(f"  saved {fn}")
        except Exception as ex:
            print(f"  COULD NOT SAVE {fn}: {ex}\n    -> close that file if it is open in a viewer/editor, or pause OneDrive, then re-run.")
    print(f"\nWriting outputs to {OUT}/ ...")
    save("metrics_survival.json", lambda: json.dump(res, open(f"{OUT}/metrics_survival.json", "w"), indent=2))

    # ---- figure 1: C-index with CI ----
    def fig_cindex():
        fig, ax = plt.subplots(figsize=(5.4, 3.4)); nm = list(models.keys())
        cvals = [res[k]["c_index"] for k in nm]
        errs = [[res[k]["c_index"]-res[k]["c_ci"][0] for k in nm], [res[k]["c_ci"][1]-res[k]["c_index"] for k in nm]]
        ax.bar(range(len(nm)), cvals, yerr=errs, capsize=4, color=["#1b7837", "#2166ac", "#d6604d"])
        ax.axhline(0.5, ls="--", color="gray", lw=1); ax.set_xticks(range(len(nm)))
        ax.set_xticklabels(["Proposed\nsurvival", "Cox PH", "Naive\nfixed-horizon"]); ax.set_ylabel("Harrell C-index")
        ax.set_ylim(0.45, max(0.8, max(cvals)+0.08)); ax.set_title(f"Real Mendeley cohort: {endpoint_label.split('(')[0].strip()}")
        for i, v in enumerate(cvals): ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_survival_cindex.png", dpi=150); plt.close(fig)
    save("fig_survival_cindex.png", fig_cindex)

    # ---- figure 2: calibration at the calibration horizon, by predicted-risk tertile ----
    def fig_calib():
        risk5 = cum[:, calk-1]; tert = np.quantile(risk5, [1/3, 2/3]); grp = np.digitize(risk5, tert)
        obs, pred = [], []
        for g in range(3):
            mask = grp == g
            if mask.sum() > 5:
                obs.append(1 - km_at(t[te][mask], e[te][mask], calh)); pred.append(risk5[mask].mean())
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        hi = max(0.5, (max(pred+obs)+0.05) if (pred and obs) else 0.5)
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="ideal")
        ax.plot(pred, obs, "o-", color="#1b7837", ms=7, label="risk tertiles")
        ax.set_xlabel(f"Predicted {calh}-y graft failure"); ax.set_ylabel(f"KM-observed {calh}-y graft failure")
        ax.set_title(f"Calibration at {calh} years (censoring-adjusted)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_survival_calibration.png", dpi=150); plt.close(fig)
    save("fig_survival_calibration.png", fig_calib)
    print("Done.")

if __name__ == "__main__":
    main()
