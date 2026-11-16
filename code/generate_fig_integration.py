"""
Four-panel decision-focused figure (replaces the old 2-panel out/fig_allocation.png; saved as
out/fig_dfl_integration.png). Tells the complete, honest story in one place:
(a) noise sweep: allocation regret rises with benefit-estimation error (sensitivity, not a law);
(b) utility-only / unconstrained: SPO+ beats two-stage at matched capacity (CI excludes zero);
(c) integrated fair+feasible, lambda sweep: the paired reduction's 95% CI crosses zero everywhere;
(d) integrated, operational regimes: same null, and SPO+ overfits when data is very scarce.
Colors match the paper (two-stage red #d6604d, SPO+ green #1b7837). No text overlaps the marks.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(HERE, "results_integration")
RED, GREEN, GREY = "#d6604d", "#1b7837", "#555555"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9})

lam = json.load(open(f"{RES}/metrics_lambda_sweep.json"))
reg = json.load(open(f"{RES}/metrics_regime.json"))
rc = pd.read_csv(f"{OUT}/regret_curve.csv")

fig, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(15.6, 3.5))

# ---- (a) noise sweep ----
axA.errorbar(rc.sigma, rc.regret, yerr=rc.sd, fmt="o-", color="#333", capsize=2)
axA.set_xlabel("Benefit-estimation error ($\\sigma$)")
axA.set_ylabel("Allocation regret (% vs oracle)")
axA.set_title("(a) Regret rises with estimation error")
axA.grid(alpha=0.25)

# ---- (b) utility-only, unconstrained (SPO+ wins) ----
groups = ["Base\nlearner", "Enriched\nlearner"]
two = [1.901, 1.378]; spo = [1.815, 1.317]; red = [0.086, 0.061]; ci = [(0.015, 0.160), (0.011, 0.111)]
x = np.arange(2); w = 0.36
axB.bar(x - w/2, two, w, color=RED, label="Two-stage")
axB.bar(x + w/2, spo, w, color=GREEN, label="Decision-focused (SPO+)")
axB.set_xticks(x); axB.set_xticklabels(groups)
axB.set_ylabel("Allocation regret (% vs oracle)")
axB.set_ylim(0, 2.5)
axB.set_title("(b) Utility-only, unconstrained")
for xi, (t, s, r, c) in enumerate(zip(two, spo, red, ci)):
    axB.text(xi - w/2, t + 0.04, f"{t:.2f}", ha="center", va="bottom", fontsize=8, color=RED)
    axB.text(xi + w/2, s + 0.04, f"{s:.2f}", ha="center", va="bottom", fontsize=8, color=GREEN)
    axB.text(xi, 2.28, f"reduction {r:.3f}*\nCI[{c[0]:.3f},{c[1]:.3f}]", ha="center", va="center", fontsize=7.0)
axB.legend(loc="lower left", fontsize=7.2, framealpha=0.9)

# ---- (c) integrated, lambda sweep (null everywhere) ----
L = [r["lam"] for r in lam]; pr = [r["paired_reduction"] for r in lam]
lo = [r["seed_clustered_95ci"][0] for r in lam]; hi = [r["seed_clustered_95ci"][1] for r in lam]
xc = np.arange(len(L)); yerr = np.array([[p - l for p, l in zip(pr, lo)], [h - p for p, h in zip(pr, hi)]])
axC.axhline(0, color=GREY, lw=1, ls="--", zorder=1)
axC.errorbar(xc, pr, yerr=yerr, fmt="o", color=GREEN, ecolor=GREY, capsize=3, ms=5, zorder=3)
axC.set_xticks(xc); axC.set_xticklabels([f"{v:g}" for v in L])
axC.set_xlabel("Equity weight $\\lambda$")
axC.set_ylabel("Reduction, two-stage $-$ SPO+ (pts)")
axC.set_title("(c) Integrated (fair+feasible): $\\lambda$ sweep")
axC.set_ylim(-0.22, 0.24)
axC.text(0.4, 0.2, "95% CI crosses 0 at every $\\lambda$", fontsize=7.4)

# ---- (d) integrated, operational regimes ----
names = ["rich\nlarge", "scarce\nlarge", "rich\nsmall", "scarce\nsmall", "v.scarce\nsmall"]
prc = [r["paired_reduction"] for r in reg]
loc = [r["seed_clustered_95ci"][0] for r in reg]; hic = [r["seed_clustered_95ci"][1] for r in reg]
xd = np.arange(len(names)); yerrd = np.array([[p - l for p, l in zip(prc, loc)], [h - p for p, h in zip(prc, hic)]])
axD.axhline(0, color=GREY, lw=1, ls="--", zorder=1)
axD.errorbar(xd, prc, yerr=yerrd, fmt="s", color=GREEN, ecolor=GREY, capsize=3, ms=5, zorder=3)
axD.set_xticks(xd); axD.set_xticklabels(names, fontsize=8)
axD.set_xlabel("Training labels / pool size")
axD.set_ylabel("Reduction (pts)")
axD.set_title("(d) Integrated: operational regimes")
axD.set_ylim(-0.52, 0.34)
axD.text(2.0, 0.26, "no regime reaches significance", fontsize=7.4, ha="center")

fig.tight_layout(w_pad=1.4)
fig.savefig(f"{OUT}/fig_dfl_integration.png", dpi=200)
print("saved", f"{OUT}/fig_dfl_integration.png")
