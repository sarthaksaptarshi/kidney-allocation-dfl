"""
Four-panel decision-focused figure (out/fig_dfl_integration.png). Every number is read from
the metrics JSONs produced by the experiment scripts -- nothing is hard-coded (review C1/C8).
(a) noise sweep: allocation regret rises with benefit-estimation error (sensitivity, not a law);
(b) utility-only / unconstrained: SPO+ beats two-stage only for the BASE (misspecified) learner
    (CI excludes zero); with the ENRICHED (well-specified) learner the gain is a tie;
(c) integrated fair+feasible, lambda sweep: tie at operating weights, significant only at the
    extreme lambda=0.5;
(d) integrated, operational regimes: null at operating points; SPO+ overfits and loses when data
    are very scarce.
Colors match the paper (two-stage red #d6604d, SPO+ green #1b7837). No text overlaps the marks.
Sources: results_dfl/metrics_dfl.json, results_integration/metrics_lambda_sweep.json,
         results_integration/metrics_regime.json, out/regret_curve.csv
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

# ---- all numbers from JSON (no hard-coding) ----
dfl_all = json.load(open(f"{HERE}/results_dfl/metrics_dfl.json"))
dfl = dfl_all["misspecification_check"]; base_ci = dfl_all["seed_clustered_95ci"]
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

# ---- (b) utility-only, unconstrained: base (misspecified) wins, enriched (well-specified) ties ----
groups = ["Base\n(misspecified)", "Enriched\n(well-specified)"]
two = [dfl["base_regret_two"], dfl["rich_regret_two"]]
spo = [dfl["base_regret_spo"], dfl["rich_regret_spo"]]
red = [dfl["base_reduction"], dfl["rich_reduction"]]
ci  = [tuple(base_ci), tuple(dfl["rich_seed_clustered_95ci"])]
sig = [ci[0][0] > 0, ci[1][0] > 0]                     # significant iff CI excludes zero
x = np.arange(2); w = 0.36
axB.bar(x - w/2, two, w, color=RED, label="Two-stage")
axB.bar(x + w/2, spo, w, color=GREEN, label="Decision-focused (SPO+)")
axB.set_xticks(x); axB.set_xticklabels(groups)
axB.set_ylabel("Allocation regret (% vs oracle)")
axB.set_ylim(0, 2.6)
axB.set_title("(b) Utility-only, unconstrained")
for xi, (t, s, r, c, sg) in enumerate(zip(two, spo, red, ci, sig)):
    axB.text(xi - w/2, t + 0.04, f"{t:.2f}", ha="center", va="bottom", fontsize=8, color=RED)
    axB.text(xi + w/2, s + 0.04, f"{s:.2f}", ha="center", va="bottom", fontsize=8, color=GREEN)
    tag = f"reduction {r:.3f}{'*' if sg else ' n.s.'}\nCI[{c[0]:.3f},{c[1]:.3f}]"
    axB.text(xi, 2.46, tag, ha="center", va="center", fontsize=7.0)
axB.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=7.0, framealpha=0.9)

# ---- (c) integrated, lambda sweep ----
L = [r["lam"] for r in lam]; pr = [r["paired_reduction"] for r in lam]
lo = [r["seed_clustered_95ci"][0] for r in lam]; hi = [r["seed_clustered_95ci"][1] for r in lam]
xc = np.arange(len(L)); yerr = np.array([[p - l for p, l in zip(pr, lo)], [h - p for p, h in zip(pr, hi)]])
axC.axhline(0, color=GREY, lw=1, ls="--", zorder=1)
axC.errorbar(xc, pr, yerr=yerr, fmt="o", color=GREEN, ecolor=GREY, capsize=3, ms=5, zorder=3)
for xi_, (p_, l_, h_) in enumerate(zip(pr, lo, hi)):
    if l_ > 0: axC.text(xi_, h_ + 0.012, "*", ha="center", va="bottom", fontsize=12, color=GREEN)
axC.set_xticks(xc); axC.set_xticklabels([f"{v:g}" for v in L])
axC.set_xlabel("Equity weight $\\lambda$")
axC.set_ylabel("Reduction, two-stage $-$ SPO+ (pts)")
axC.set_title("(c) Integrated (fair+feasible): $\\lambda$ sweep")
axC.set_ylim(-0.22, 0.26)
axC.text(0.3, -0.19, "tie for $\\lambda\\leq0.25$; * = CI excludes 0", fontsize=7.0, ha="left")

# ---- (d) integrated, operational regimes ----
names = ["rich\nlarge", "scarce\nlarge", "rich\nsmall", "scarce\nsmall", "v.scarce\nsmall"]
prc = [r["paired_reduction"] for r in reg]
loc = [r["seed_clustered_95ci"][0] for r in reg]; hic = [r["seed_clustered_95ci"][1] for r in reg]
xd = np.arange(len(names)); yerrd = np.array([[p - l for p, l in zip(prc, loc)], [h - p for p, h in zip(prc, hic)]])
axD.axhline(0, color=GREY, lw=1, ls="--", zorder=1)
axD.errorbar(xd, prc, yerr=yerrd, fmt="s", color=GREEN, ecolor=GREY, capsize=3, ms=5, zorder=3)
for xi_, (p_, l_, h_) in enumerate(zip(prc, loc, hic)):
    if h_ < 0: axD.text(xi_, l_ - 0.03, "*", ha="center", va="top", fontsize=12, color=GREEN)
axD.set_xticks(xd); axD.set_xticklabels(names, fontsize=8)
axD.set_xlabel("Training labels / pool size")
axD.set_ylabel("Reduction (pts)")
axD.set_title("(d) Integrated: operational regimes")
axD.set_ylim(-0.74, 0.34)
axD.text(2.0, 0.27, "null at operating points; SPO+ overfits when scarce", fontsize=7.0, ha="center")

fig.tight_layout(w_pad=1.4)
fig.savefig(f"{OUT}/fig_dfl_integration.png", dpi=200)
print("saved", f"{OUT}/fig_dfl_integration.png")
print("panel b:", "base", red[0], "sig" if sig[0] else "n.s.", "| enriched", red[1], "sig" if sig[1] else "n.s.")
