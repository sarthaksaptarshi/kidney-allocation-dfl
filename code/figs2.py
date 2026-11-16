"""Additional publication figures (6-12), reproducing the same synthetic cohort
and models as sim.py (seed 20261116) so every number stays synchronized."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
RNG=np.random.default_rng(20261116); OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"figure.dpi":150,"font.size":10,"axes.grid":True,"grid.alpha":0.25,
                     "axes.axisbelow":True,"savefig.bbox":"tight"})
C1,C2,C3,C4="#2166ac","#d6604d","#1b7837","#762a83"

# ---- regenerate cohort (identical to sim.py) ----
N=18000
def gen(n):
    rec_age=np.clip(RNG.normal(45,13,n),18,75); don_age=np.clip(RNG.normal(42,15,n),5,78)
    hla_mm=RNG.integers(0,7,n).astype(float); base=RNG.beta(1.4,6,n)*100
    sens=RNG.binomial(1,0.18,n).astype(bool); cpra=np.where(sens,RNG.uniform(55,99,n),base)
    cit=np.clip(RNG.gamma(3.0,3.2,n),1,40); dial=np.clip(RNG.gamma(2.0,1.6,n),0,18)
    bmi=np.clip(RNG.normal(24,4.5,n),15,42); cyp=RNG.binomial(1,0.35,n); ddtx=RNG.binomial(1,0.62,n)
    dia=RNG.binomial(1,0.28,n); fem=RNG.binomial(1,0.40,n)
    dsa=RNG.binomial(1,np.clip(0.03+cpra/300,0,0.7)).astype(float); tac=np.clip(RNG.normal(7.5,2.4,n),2,16)
    return pd.DataFrame(dict(rec_age=rec_age,don_age=don_age,hla_mm=hla_mm,cpra=cpra,cit=cit,
        dial_yr=dial,bmi=bmi,cyp3a5=cyp,ddtx=ddtx,diabetes=dia,female=fem,dsa=dsa,tac_trough=tac))
df=gen(N); feat=["rec_age","don_age","hla_mm","cpra","cit","dial_yr","bmi","cyp3a5","ddtx","diabetes","female","dsa","tac_trough"]
u=RNG.normal(0,1,N)
def zc(c,k): return df[c]-k
def ctr(a): return a-a.mean()
tacU=(np.minimum(df.tac_trough-5,0)**2+np.maximum(df.tac_trough-11,0)**2).to_numpy()
imm=ctr((0.42*df.hla_mm+0.030*zc("cpra",18)+1.4*df.dsa+0.045*zc("cit",10)+0.018*zc("don_age",40)+0.40*df.ddtx+0.55*df.hla_mm*df.dsa+0.6*(df.cpra>80)).to_numpy())
chr_=ctr((0.10*df.hla_mm+0.012*zc("cpra",18)+0.045*zc("don_age",40)+0.12*df.dial_yr+0.25*df.diabetes+0.040*zc("bmi",24)+0.030*zc("cit",10)+0.10*tacU+0.20*(df.dial_yr>5)*df.diabetes).to_numpy())
tox=ctr((0.70*zc("tac_trough",7.5)+1.8*(1-df.cyp3a5)*(df.tac_trough>9)+0.02*zc("rec_age",45)+0.10*tacU).to_numpy())
def sig(x): return 1/(1+np.exp(-x))
h1=sig(-2.95+0.9*imm+0.8*u); h2=sig(-1.70+0.8*imm+0.3*chr_+0.9*u); h3=sig(-0.35+0.3*imm+0.9*chr_+1.0*u)
r1=RNG.binomial(1,h1).astype(bool); r2=(~r1)&(RNG.binomial(1,h2).astype(bool)); r3=(~r1)&(~r2)&(RNG.binomial(1,h3).astype(bool))
event_time=np.full(N,1e9); event_time[r1]=RNG.uniform(0.1,1,r1.sum()); event_time[r2]=RNG.uniform(1,12,r2.sum()); event_time[r3]=RNG.uniform(12,120,r3.sum())
chr_std=(chr_-chr_.mean())/chr_.std(); fu=np.minimum(1+RNG.exponential(np.clip(54*np.exp(-0.35*chr_std),8,120)),120)
X=df[feat].to_numpy(); H={"1mo":1,"1yr":12,"10yr":120}
feat_offer=[c for c in feat if c!="tac_trough"]; X_offer=df[feat_offer].to_numpy()  # R4-C2: offer-time set for the MH model (fig7, fig10)
def hd(Hm):
    naive=((event_time<=Hm)&(event_time<=fu)).astype(int)
    known=((event_time<=np.minimum(Hm,fu))|(fu>=Hm)); true=(event_time<=Hm).astype(int)
    return naive,known,true
Xtr,Xte,itr,ite=train_test_split(X_offer,np.arange(N),test_size=0.3,random_state=7,stratify=hd(12)[2])  # offer-time MH model
def gbm(): return GradientBoostingClassifier(n_estimators=180,max_depth=3,learning_rate=0.06,random_state=1)

# survival hazards (fine grid)
grid=[0,1,3,6,12,24,48,84,120]; edges=list(zip(grid[:-1],grid[1:]))
haz=np.zeros((len(ite),len(edges)))
for k,(s,e) in enumerate(edges):
    ar=((event_time>s)&(fu>s))[itr]; yk=((event_time>s)&(event_time<=e)&(event_time<=fu)).astype(int)
    haz[:,k]=gbm().fit(Xtr[ar],yk[itr][ar]).predict_proba(Xte)[:,1]
cum=1-np.cumprod(1-haz,axis=1); idx={1:grid.index(1)-1,12:grid.index(12)-1,120:grid.index(120)-1}
r1m,r2m,r3m=cum[:,idx[1]],cum[:,idx[12]],cum[:,idx[120]]

# ---------- Fig 6: cohort prevalence + censoring ----------
fig,ax=plt.subplots(figsize=(5.2,3.4))
hz=["1 month","1 year","10 year"]; prev=[hd(1)[2].mean(),hd(12)[2].mean(),hd(120)[2].mean()]
cens=[1-hd(1)[1].mean(),1-hd(12)[1].mean(),1-hd(120)[1].mean()]
x=np.arange(3); w=0.38
ax.bar(x-w/2,prev,w,label="Rejection prevalence (true)",color=C1)
ax.bar(x+w/2,cens,w,label="Censored before horizon",color=C2)
ax.set_xticks(x); ax.set_xticklabels(hz); ax.set_ylabel("Fraction of cohort"); ax.set_ylim(0,0.75)
for xi,(p,c) in enumerate(zip(prev,cens)):
    ax.text(xi-w/2,p+0.015,f"{p:.2f}",ha="center",fontsize=8.5); ax.text(xi+w/2,c+0.015,f"{c:.2f}",ha="center",fontsize=8.5)
ax.set_title("Synthetic cohort: rejection prevalence and censoring"); ax.legend(fontsize=8.5,loc="upper left")
fig.savefig(f"{OUT}/fig6_cohort.png"); plt.close(fig)

# ---------- Fig 7: cumulative incidence by predicted-risk tertile ----------
fig,ax=plt.subplots(figsize=(5.4,3.6))
tert=np.quantile(r3m,[1/3,2/3]); grp=np.digitize(r3m,tert)
tgrid=grid[1:]; labels=["Low risk","Medium risk","High risk"]; cols=[C3,"#f6a800",C2]
for g,(lab,col) in enumerate(zip(labels,[C1,"#f1a340",C2])):
    ax.plot(tgrid,cum[grp==g].mean(0),"o-",color=col,label=f"{lab} (n={np.sum(grp==g)})",ms=4)
ax.set_xlabel("Months since transplant"); ax.set_ylabel("Predicted cumulative incidence of rejection")
ax.set_title("Risk stratification by predicted 10-year risk tertile"); ax.legend(fontsize=8.5); ax.set_xlim(0,120)
fig.savefig(f"{OUT}/fig7_cuminc.png"); plt.close(fig)

# ---------- Fig 8: ROC curves (proposed, three horizons) ----------
fig,ax=plt.subplots(figsize=(4.6,4.4))
for col_,(h,p) in zip([C1,C3,C4],[("1-month",r1m),("1-year",r2m),("10-year",r3m)]):
    naive,known,true=hd(H[h.split("-")[0].replace("1","1")+("mo" if "month" in h else "yr")]) if False else hd({"1-month":1,"1-year":12,"10-year":120}[h])
    kn=known[ite]; fpr,tpr,_=roc_curve(true[ite][kn],p[kn]); a=roc_auc_score(true[ite][kn],p[kn])
    ax.plot(fpr,tpr,color=col_,label=f"{h} (AUC {a:.3f})",lw=1.8)
ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC: proposed multi-horizon predictor"); ax.legend(fontsize=8.5,loc="lower right")
fig.savefig(f"{OUT}/fig8_roc.png"); plt.close(fig)

# ---------- Fig 9: differential confusion matrix ----------
p_tox=sig(-0.6+0.9*tox+0.3*u-1.1*df.dsa.to_numpy())
dysf=(event_time<=12)&(event_time<=fu); ylab=(RNG.binomial(1,p_tox)==1).astype(int)
yd=np.where(dysf,ylab,-1); dtr=itr[yd[itr]>=0]; dte=ite[yd[ite]>=0]
gd=GradientBoostingClassifier(n_estimators=240,max_depth=3,learning_rate=0.06,random_state=6).fit(X[dtr],yd[dtr])
pdp=gd.predict_proba(X[dte])[:,1]; pdpred=(pdp>=0.5).astype(int); cm=confusion_matrix(yd[dte],pdpred)
fig,ax=plt.subplots(figsize=(4.0,3.6)); im=ax.imshow(cm,cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["AMR","Toxicity"]); ax.set_yticklabels(["AMR","Toxicity"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("AMR vs tacrolimus toxicity")
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=12)
fig.colorbar(im,fraction=0.046,pad=0.04); fig.savefig(f"{OUT}/fig9_confusion.png"); plt.close(fig)

# ---------- Fig 10: conformal coverage & abstention vs alpha ----------
from sklearn.isotonic import IsotonicRegression
naive1,known1,true1=hd(12); kn=known1[ite]; yk=true1[ite][kn]; rk=r2m[kn]
ci,ti=train_test_split(np.arange(kn.sum()),test_size=0.5,random_state=21)
iso=IsotonicRegression(out_of_bounds="clip").fit(rk[ci],yk[ci]); pcal=iso.predict(rk)
P=np.column_stack([1-pcal,pcal]); alphas=np.linspace(0.02,0.30,15); covs=[];absr=[]
for al in alphas:
    nc=1-P[ci,yk[ci]]; q=np.quantile(nc,min(1,np.ceil((len(ci)+1)*(1-al))/len(ci)))
    in0=(1-P[ti,0])<=q; in1=(1-P[ti,1])<=q; ss=in0.astype(int)+in1.astype(int)
    cov=np.where(yk[ti]==0,in0,in1).mean(); covs.append(cov); absr.append((ss!=1).mean())
fig,ax=plt.subplots(figsize=(5.2,3.5))
ax.plot(1-alphas,covs,"o-",color=C1,label="Empirical coverage"); ax.plot(1-alphas,1-alphas,"k--",lw=1,label="Target coverage")
ax2=ax.twinx(); ax2.plot(1-alphas,absr,"s-",color=C2,label="Abstention rate")
ax.set_xlabel("Target coverage (1 − α)"); ax.set_ylabel("Empirical coverage"); ax2.set_ylabel("Abstention rate")
ax.set_title("Conformal prediction: coverage and abstention")
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,fontsize=8,loc="center left")
fig.savefig(f"{OUT}/fig10_conformal.png"); plt.close(fig)

# ---------- Fig 11: permutation importance (1yr) ----------
m1=gbm().fit(Xtr,naive1[itr])
pi=permutation_importance(m1,Xte[kn],true1[ite][kn],n_repeats=8,random_state=0,scoring="roc_auc")
order=np.argsort(pi.importances_mean)[::-1][:10][::-1]
fig,ax=plt.subplots(figsize=(5.2,3.8))
ax.barh([feat[i] for i in order],pi.importances_mean[order],xerr=pi.importances_std[order],color=C1)
ax.set_xlabel("Drop in AUROC when permuted"); ax.set_title("Permutation importance (1-year rejection)")
fig.savefig(f"{OUT}/fig11_importance.png"); plt.close(fig)

# ---------- Fig 12: fairness access bars (util vs fair) ----------
S=json.load(open(f"{OUT}/summary.json"))["allocation"]
fig,ax=plt.subplots(figsize=(5.2,3.5)); groups=["Female","Highly\nsensitized","Pediatric"]
util=[S["female_access_util"],S["sensitized_access_util"],S["pediatric_access_util"]]
fair=[S["female_access_fair"],S["sensitized_access_fair"],S["pediatric_access_fair"]]
x=np.arange(3); w=0.38
ax.bar(x-w/2,util,w,label="Utility-only",color=C2); ax.bar(x+w/2,fair,w,label="Fairness-constrained",color=C3)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Access rate"); ax.set_ylim(0,1.08)
for xi,(a,b) in enumerate(zip(util,fair)):
    ax.text(xi-w/2,a+0.02,f"{a:.2f}",ha="center",fontsize=8.5); ax.text(xi+w/2,b+0.02,f"{b:.2f}",ha="center",fontsize=8.5)
ax.set_title(f"Equity gain (price of fairness {S['price_of_fairness_pct']}%)"); ax.legend(fontsize=8.5,loc="upper right")
fig.savefig(f"{OUT}/fig12_access.png"); plt.close(fig)
print("saved figs 6-12")
