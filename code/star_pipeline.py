"""
REAL-DATA PIPELINE for a large transplant registry (OPTN/UNOS STAR or SRTR SAF).

Schema-driven: you pass the column names for time-to-event, the event indicator, and the
pre-transplant predictors; the pipeline fits the paper's models and reports the discrimination
comparison that decides the "real-data win" --- a PAIRED bootstrap CI on the C-index difference
(proposed discrete-time survival MINUS Cox), plus per-model C-index, time-dependent AUROC, and
censoring-adjusted calibration. Pre-transplant features only (no leakage).

Scales to registry size: Harrell's C is O(n^2), so C-index and its bootstrap are computed on a
random subsample (--c-sample, default 20000); model fitting uses all rows.

Run (defaults match the synthetic STAR-shaped validation file):
  python star_pipeline.py star.csv \
      --time GTIME_Y --event GSTATUS \
      --num rec_age don_age hla_mm cpra cold_isch dial_vintage don_bmi rec_bmi \
      --cat diabetes abo_mm dcd regraft \
      --grid 0 1 5 10
Outputs: results_star/metrics_star.json, fig_star_cindex.png, fig_star_calibration.png
Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_star"); os.makedirs(OUT, exist_ok=True)

# ---------- Kaplan-Meier (for censoring-adjusted calibration) ----------
def km(times, events):
    order = np.argsort(times); t, e = np.asarray(times)[order], np.asarray(events)[order]
    uniq = np.unique(t); S = 1.0; surv = []
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

# ---------- Cox PH (Breslow), self-contained, O(n) per eval ----------
def cox_fit(X, t, e, l2=1.0):
    order = np.argsort(-t); Xs, ts, es = X[order], t[order], e[order]
    def negll(beta):
        eta = Xs @ beta; m = eta.max(); ex = np.exp(eta - m)
        cum = np.cumsum(ex); cumX = np.cumsum(ex[:, None] * Xs, axis=0)
        ev = es == 1
        ll = np.sum(eta[ev] - (np.log(cum[ev]) + m))
        grad = np.sum(Xs[ev] - cumX[ev] / cum[ev, None], axis=0)
        return -ll + 0.5*l2*np.sum(beta**2), -grad + l2*beta
    res = minimize(negll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return res.x
def cox_risk(X, beta): return X @ beta

# ---------- Harrell C-index via fast vectorized comparable-pair sampling ----------
# Exact Harrell C is O(n^2); on a registry that is infeasible. We estimate it by sampling random
# pairs and scoring the comparable ones (one subject has a definite earlier event). With ~1e6 pairs
# the estimate is tight; CIs come from subject-level bootstrap, reusing the same sampled pairs for
# paired model comparisons so C(A)-C(B) is evaluated on identical pairs (the correct paired test).
def _c_on_pairs(risk, t, e, i, j):
    ti, tj, ei, ej, ri, rj = t[i], t[j], e[i], e[j], risk[i], risk[j]
    compA = (ti < tj) & (ei == 1)      # i has the earlier, definite event
    compB = (tj < ti) & (ej == 1)      # j has the earlier, definite event
    comp = compA | compB; den = int(comp.sum())
    if den == 0: return 0.5
    conc = (np.sum(compA & (ri > rj)) + np.sum(compB & (rj > ri)) + 0.5*np.sum(comp & (ri == rj)))
    return float(conc/den)
def c_with_ci(risk, t, e, pairs, nboot=300, seed=1):
    rng = np.random.default_rng(seed); n = len(t)
    point = _c_on_pairs(risk, t, e, rng.integers(0, n, min(2_000_000, 40*n)), rng.integers(0, n, min(2_000_000, 40*n)))
    boots = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)                                   # subject-level bootstrap
        boots.append(_c_on_pairs(risk[idx], t[idx], e[idx], rng.integers(0, n, pairs), rng.integers(0, n, pairs)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), [float(lo), float(hi)]
def paired_c_diff(rA, rB, t, e, pairs, nboot=300, seed=7):
    """Paired bootstrap CI for C(A) - C(B): same resampled subjects and same pairs for both (the win/lose test)."""
    rng = np.random.default_rng(seed); n = len(t); diffs = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n); i = rng.integers(0, n, pairs); j = rng.integers(0, n, pairs)
        ta, ea = t[idx], e[idx]
        diffs.append(_c_on_pairs(rA[idx], ta, ea, i, j) - _c_on_pairs(rB[idx], ta, ea, i, j))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(mean=round(float(np.mean(diffs)), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                prob_A_better=round(float(np.mean(np.array(diffs) > 0)), 3))

# ---------- discrete-time survival (proposed) ----------
def discrete_survival(Xtr, ttr, etr, Xte, grid):
    edges = list(zip(grid[:-1], grid[1:])); haz = np.zeros((len(Xte), len(edges)))
    for k, (a, b) in enumerate(edges):
        at_risk = ttr > a; yk = ((ttr > a) & (ttr <= b) & (etr == 1)).astype(int)
        if at_risk.sum() > 20 and yk[at_risk].sum() > 3:
            m = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0)
            m.fit(Xtr[at_risk], yk[at_risk]); haz[:, k] = m.predict_proba(Xte)[:, 1]
    return 1 - np.cumprod(1 - haz, axis=1)

def td_auroc(risk_by_t, t, e, horizon):
    case = (t <= horizon) & (e == 1); ctrl = t > horizon; keep = case | ctrl
    if case.sum() < 5 or ctrl.sum() < 5: return np.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(case[keep].astype(int), risk_by_t[keep]))

def main():
    ap = argparse.ArgumentParser(description="Large-registry transplant survival pipeline (proposed vs Cox).")
    ap.add_argument("path")
    ap.add_argument("--time", required=True, help="column: time-to-event (years; see --time-unit)")
    ap.add_argument("--event", required=True, help="column: event indicator (1=graft failure, 0=censored)")
    ap.add_argument("--num", nargs="+", default=[], help="numeric pre-transplant predictor columns")
    ap.add_argument("--cat", nargs="+", default=[], help="categorical pre-transplant predictor columns")
    ap.add_argument("--grid", nargs="+", type=float, default=[0, 1, 5, 10], help="horizon cut-points (years)")
    ap.add_argument("--time-unit", choices=["years", "days"], default="years")
    ap.add_argument("--event-positive", default=None, help="if the event column is a label, its positive value (e.g. 'Y')")
    ap.add_argument("--pairs", type=int, default=500000, help="comparable pairs sampled per bootstrap for the C-index")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.path) if args.path.lower().endswith(".csv") else pd.read_excel(args.path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in [args.time, args.event] + args.num + args.cat:
        if c not in df.columns: sys.exit(f"column not found: {c!r}. Available: {list(df.columns)[:40]}")

    t = pd.to_numeric(df[args.time], errors="coerce").to_numpy(float)
    if args.time_unit == "days": t = t / 365.25
    if args.event_positive is not None:
        e = (df[args.event].astype(str).str.strip() == args.event_positive).astype(int).to_numpy()
    else:
        e = pd.to_numeric(df[args.event], errors="coerce").fillna(0).astype(int).to_numpy()

    cols, X = [], []
    for c in args.num:
        X.append(pd.to_numeric(df[c], errors="coerce").to_numpy(float)); cols.append(c)
    for c in args.cat:
        d = pd.get_dummies(df[c].astype(str).str.strip().replace({"nan": np.nan}), prefix=c, dummy_na=False)
        for cc in d.columns: X.append(d[cc].to_numpy(float)); cols.append(cc)
    if not X: sys.exit("no predictors given (use --num / --cat).")
    X = np.column_stack(X)
    keep = np.isfinite(t) & (t > 0)
    X, t, e = X[keep], t[keep], e[keep]
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X)
    grid = list(args.grid)
    print(f"n={len(t)}  events={int(e.sum())} ({e.mean()*100:.1f}%)  median follow-up={np.median(t):.1f} y  "
          f"predictors={X.shape[1]}  C-index pairs/boot={args.pairs}")

    n = len(t); risk_prop = np.zeros(n); risk_cox = np.zeros(n); risk_naive = np.zeros(n)
    cum_full = np.zeros((n, len(grid)-1))
    ch = grid[len(grid)//2] if len(grid) > 2 else grid[-1]      # a middle horizon for the naive classifier
    for tr, te in StratifiedKFold(args.folds, shuffle=True, random_state=0).split(Xs, e):
        cum_full[te] = discrete_survival(Xs[tr], t[tr], e[tr], Xs[te], grid)
        risk_prop[te] = cum_full[te][:, -1]
        risk_cox[te] = cox_risk(Xs[te], cox_fit(Xs[tr], t[tr], e[tr], l2=1.0))
        yk = ((t[tr] <= ch) & (e[tr] == 1)).astype(int)
        nb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0).fit(Xs[tr], yk)
        risk_naive[te] = nb.predict_proba(Xs[te])[:, 1]

    res = {"endpoint_time": args.time, "endpoint_event": args.event, "n": int(n),
           "events": int(e.sum()), "event_rate": round(float(e.mean()), 4),
           "c_index_pairs_per_boot": int(args.pairs), "grid": grid}
    models = {"Proposed (discrete-time survival)": risk_prop, "Cox PH": risk_cox, "Naive fixed-horizon": risk_naive}
    for name, r in models.items():
        c, ci = c_with_ci(r, t, e, args.pairs)
        res[name] = dict(c_index=round(c, 3), c_ci=[round(ci[0], 3), round(ci[1], 3)])
        print(f"{name:34s} C-index {c:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    # THE WIN TEST: paired C-index difference, proposed - Cox
    res["proposed_minus_cox_paired"] = paired_c_diff(risk_prop, risk_cox, t, e, args.pairs)
    d = res["proposed_minus_cox_paired"]
    verdict = ("proposed WINS (CI excludes 0)" if d["ci"][0] > 0 else
               "Cox WINS (CI excludes 0)" if d["ci"][1] < 0 else "tie (CI includes 0)")
    print(f"PAIRED proposed - Cox: {d['mean']:+.4f}  95% CI {d['ci']}  P(proposed better)={d['prob_A_better']}  -> {verdict}")
    res["win_verdict"] = verdict
    res["Proposed_td_AUROC"] = {f"{grid[k]}y": (lambda v: None if np.isnan(v) else round(v, 3))(td_auroc(cum_full[:, k-1], t, e, grid[k]))
                                for k in range(1, len(grid))}
    print("Proposed time-dependent AUROC:", res["Proposed_td_AUROC"])

    json.dump(res, open(f"{OUT}/metrics_star.json", "w"), indent=2)
    print(f"saved {OUT}/metrics_star.json")

    # figure 1: C-index with CI + paired verdict
    nm = list(models.keys()); cvals = [res[k]["c_index"] for k in nm]
    errs = [[res[k]["c_index"]-res[k]["c_ci"][0] for k in nm], [res[k]["c_ci"][1]-res[k]["c_index"] for k in nm]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(range(len(nm)), cvals, yerr=errs, capsize=4, color=["#1b7837", "#2166ac", "#d6604d"])
    ax.axhline(0.5, ls="--", color="gray", lw=1); ax.set_xticks(range(len(nm)))
    ax.set_xticklabels(["Proposed\nsurvival", "Cox PH", "Naive\nfixed-horizon"]); ax.set_ylabel("Harrell C-index")
    ax.set_ylim(0.45, max(0.8, max(cvals)+0.08))
    ax.set_title(f"Registry graft-failure discrimination (n={n:,}, {int(e.sum()):,} events)\nproposed$-$Cox {d['mean']:+.3f}, 95% CI [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]", fontsize=9)
    for i, v in enumerate(cvals): ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_star_cindex.png", dpi=150); plt.close(fig)

    # figure 2: censoring-adjusted calibration at a middle horizon, by predicted-risk tertile
    calk = len(grid)//2 if len(grid) > 2 else len(grid)-1; calh = grid[calk]
    risk5 = cum_full[:, calk-1]; tert = np.quantile(risk5, [1/3, 2/3]); grp = np.digitize(risk5, tert)
    obs, pred = [], []
    for g in range(3):
        mask = grp == g
        if mask.sum() > 5: obs.append(1 - km_at(t[mask], e[mask], calh)); pred.append(risk5[mask].mean())
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    hi = max(0.5, (max(pred+obs)+0.05) if (pred and obs) else 0.5)
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="ideal")
    ax.plot(pred, obs, "o-", color="#1b7837", ms=7, label="risk tertiles")
    ax.set_xlabel(f"Predicted {calh:g}-y graft failure"); ax.set_ylabel(f"KM-observed {calh:g}-y graft failure")
    ax.set_title(f"Calibration at {calh:g} years (censoring-adjusted)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_star_calibration.png", dpi=150); plt.close(fig)
    print(f"saved {OUT}/fig_star_cindex.png and {OUT}/fig_star_calibration.png")

if __name__ == "__main__":
    main()
