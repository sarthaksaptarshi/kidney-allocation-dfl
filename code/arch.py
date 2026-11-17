import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size": 9})
# short, wide band: aspect ~3.5:1 so it reserves little vertical space and the page fills with text
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 120); ax.set_ylim(0, 36)

def box(x, y, w, h, txt, fc, ec="#333", fs=8.7, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.3",
                 fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x1, y1, x2, y2, c="#444", ls="-", lw=1.6, cs=None):
    kw = dict(arrowstyle="-|>", mutation_scale=13, color=c, ls=ls, lw=lw)
    if cs: kw["connectionstyle"] = cs
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))

blue="#d6e4f0"; green="#d9ecdb"; purple="#e8def2"; orange="#f7e3cf"; grey="#ececec"; yellow="#fbf4d0"

# inputs (merged data + feature layer -> fewer boxes, fewer words)
box(1, 9.5, 19, 15, "Data & features\npublic + synthetic pool\nHLA, cPRA, DSA,\nvintage, tacrolimus", grey, fs=8.4)
# (A) multi-horizon predictor
box(24, 13, 20, 10, "(A) Multi-horizon\nsurvival predictor\n$F$(1mo,1yr,10yr), monotone", green)
# benefit matrix (surrogate in the main experiments; fitted survival in the loop-closure study)
box(48, 13.5, 11, 9, "Benefit $B$\n(surrogate /\nproxy)", orange, fs=7.6)
# (C) fair feasible optimizer
box(63, 13, 21, 10, "(C) Fair, feasible\noptimizer\nNOTTO priority + ABO/XM", purple)
# allocation output
box(88, 13.5, 20, 9, "Allocation\n+ price of fairness\n+ equity audit", blue)
# (B) uncertainty + differential (branch below A)
box(24, 0.5, 20, 9.5, "(B) Conformal uncertainty\n+ AMR/toxicity gate\n$\\rightarrow$ abstain / biopsy", yellow)

# main left-to-right flow
arrow(20, 17, 24, 18)
arrow(44, 18, 48, 18)
arrow(59, 18, 63, 18)
arrow(84, 18, 88, 18)
arrow(34, 13, 34, 10, c="#777")              # predictor -> uncertainty

# (D) decision-focused feedback: regret from the optimizer trains the BENEFIT-matrix learner
# (the surrogate / fitted-survival benefit), NOT the survival predictor (A).
arrow(72, 23.5, 54, 23.5, c="#b2182b", ls="--", lw=1.7, cs="arc3,rad=0.42")
ax.text(63, 31.5, "(D) decision-focused feedback (regret-aware):\ntrains the benefit learner", color="#b2182b", fontsize=7.8, ha="center")

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(_OUT, exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig_architecture.png"), dpi=200, bbox_inches="tight")
print("saved architecture")
