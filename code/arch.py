import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size":9.5})
fig,ax=plt.subplots(figsize=(10.5,6.2)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,62)

def box(x,y,w,h,txt,fc,ec="#333",fs=9.5,tc="#111"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.4",
        fc=fc,ec=ec,lw=1.3)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,wrap=True)
def arrow(x1,y1,x2,y2,c="#444",ls="-",lw=1.6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=c,ls=ls,lw=lw))

# palette
blue="#d6e4f0"; green="#d9ecdb"; purple="#e7ddف0".replace("ف","f"); purple="#e8def2"; orange="#f7e3cf"; grey="#ececec"; yellow="#fbf4d0"

# Stage 1: Data
box(2,46,22,12,"Public & synthetic data\n• SRTR / OPTN STAR\n• UCI-CKD (Apollo, India)\n• NHANES • synthetic\n   NOTTO-parameterised pool",blue,fs=8.6)
# Stage 2: Feature / frailty layer
box(2,26,22,13,"Feature & risk layer\n• HLA mm, cPRA, DSA\n• cold ischemia, vintage\n• tacrolimus, CYP3A5\n• competing-risks structure",grey,fs=8.6)
arrow(13,46,13,39)

# Stage 3: Multi-horizon survival predictor
box(30,40,26,16,"(A) Multi-horizon survival predictor\nDiscrete-time interval hazards\non at-risk sets → cumulative\nincidence F(1mo),F(1yr),F(10yr)\nisotonic temporal consistency",green,fs=8.7)
arrow(24,33,30,46)
# Stage 3b conformal + differential
box(30,20,26,15,"(B) Uncertainty & differential\n• Conformal prediction sets\n   → abstain / refer to biopsy\n• AMR vs tacrolimus-toxicity\n   gated classifier",yellow,fs=8.7)
arrow(24,30,30,27)
arrow(43,40,43,35,c="#777",ls="--")

# Stage 4: benefit / utility estimation
box(62,40,16,16,"Expected graft-\nsurvival benefit\nB(donor,recip)\n= utility matrix",orange,fs=8.7)
arrow(56,48,62,48)

# Stage 5: fairness-constrained optimizer
box(82,34,16,22,"(C) Fairness-\nconstrained\nallocation\noptimizer\n(assignment +\nNOTTO bonus\nconstraints)",purple,fs=8.7)
arrow(78,48,82,46)

# Output
box(82,12,16,12,"Allocation\n+ price-of-fairness\n+ equity audit",blue,fs=8.7)
arrow(90,34,90,24)

# Decision-focused feedback loop (B -> predictor training)
arrow(82,40,60,20,c="#b2182b",ls="--",lw=1.8)
ax.text(66,25,"(D) decision-focused\nfeedback (regret-aware)",color="#b2182b",fontsize=8.3,ha="center")

# Continuous / federated updating loop
arrow(90,12,43,19,c="#2166ac",ls=":",lw=1.7)
ax.text(55,9,"continuous / federated model updating (DPDP-compliant)",color="#2166ac",fontsize=8.3,ha="center")

ax.text(50,60.4,"Decision-focused, fairness-constrained kidney graft-rejection prediction & allocation",
        ha="center",fontsize=11.5,fontweight="bold")
_OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out"); os.makedirs(_OUT,exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(_OUT,"fig_architecture.png"),dpi=135,bbox_inches="tight")
print("saved architecture")
