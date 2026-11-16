"""
REAL-DATA GRAFT-REJECTION ARM on the open Mendeley kidney-allograft dataset
(CC BY 4.0, https://data.mendeley.com/datasets/3vdszzmz2n/1). No license wall.

It is written to work WITHOUT you knowing the exact column names: it auto-detects the
usual transplant fields (HLA mismatch, PRA, cold ischemia, donor/recipient age, dialysis
vintage, sex, diabetes) and the rejection outcome, then prints what it matched so you can
check it. If auto-detect misses something, fill in the small MAP dicts at the top.

USAGE
  Step 1 (look at the file):   python rejection_real.py --inspect "data/yourfile.xlsx"
  Step 2 (run the models):     python rejection_real.py "data/yourfile.xlsx"
  (Excel needs one extra package:  pip install openpyxl )

Outputs (in results_transplant/): metrics_transplant.json, ROC + importance figures.
"""
import argparse, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Configured for the Mendeley campath/simulect UK transplant dataset ----
# (set LABEL_COL=None and FEATURE_MAP={} to go back to auto-detect for other files)
LABEL_COL = "hxar"         # history of acute rejection (Yes/No)
FEATURE_MAP = {            # standard model inputs -> this file's column names
    "rec_age": "ageattx", "don_age": "dageattx", "hla_mm": "totalmm", "cpra": "pra",
    "cit": "cit", "bmi": "rbmi", "diabetes": "rdm", "sex": "sex", "donor_type": "dtype",
}
# extra baseline (pre-transplant) predictors to include, by this file's exact column names.
# NOTE: we deliberately EXCLUDE post-transplant fields (egfr3/egfr12, infections, dcgl,
# failcause, death) to avoid leakage - they are consequences, not predictors, of rejection.
EXTRA_FEATURES = ["dsa", "crossmatch", "abomm", "mma", "mmb", "mmdr",
                  "prevtx", "premt_tx", "ethnicity", "ind", "initis"]

# keyword patterns used for auto-detection (lower-cased substring match)
KEYS = {
 "hla_mm":  ["hla mism","hla_mism","hlamm","mismatch","hla mm"],
 "cpra":    ["cpra","pra","panel reactive","panel-reactive","sensiti"],
 "cit":     ["cold isch","cold_isch","ischemia","ischaemia","cit"],
 "don_age": ["donor age","donor_age","d_age","age donor"],
 "rec_age": ["recipient age","recipient_age","r_age","age recipient","age at transplant"],
 "dial_yr": ["dialysis","vintage","time on dialysis"],
 "bmi":     ["bmi","body mass"],
 "diabetes":["diabet"],
 "sex":     ["sex","gender"],
 "donor_type":["deceased","living","donor type","donor_type"],
}
LABEL_KEYS = ["reject", "acute rejection", "ar ", "rejection"]
POS_TOKENS = {"1","yes","y","true","t","positive","pos","rejection","present","r"}

def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)          # needs openpyxl
    return pd.read_csv(path)

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())

def find_col(cols, patterns, exclude=()):
    for c in cols:
        n = norm(c)
        if any(x in exclude for x in [c]): continue
        if any(p in n for p in patterns):
            return c
    return None

def to_binary(series):
    v = series.astype(str).str.strip().str.lower()
    # already numeric 0/1?
    num = pd.to_numeric(series, errors="coerce")
    if num.dropna().isin([0,1]).all() and num.notna().mean() > 0.8:
        return num
    return v.map(lambda x: 1 if x in POS_TOKENS else (np.nan if x in ("nan","","?") else 0))

def inspect(df):
    print(f"\nFile loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    print(f"{'column':32s} {'dtype':10s} {'%missing':8s} example values")
    print("-"*80)
    for c in df.columns:
        miss = df[c].isna().mean()*100
        uniq = df[c].dropna().unique()[:4]
        ex = ", ".join(map(lambda x: str(x)[:14], uniq))
        print(f"{str(c)[:32]:32s} {str(df[c].dtype)[:10]:10s} {miss:6.1f}%  {ex}")
    print("\nTip: find the acute-rejection column above; if it isn't auto-detected,")
    print("set LABEL_COL at the top of this script to its exact name.\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to the Mendeley CSV or Excel file")
    ap.add_argument("--inspect", action="store_true", help="just print the columns and exit")
    ap.add_argument("--label", default=None, help="name of the rejection outcome column")
    ap.add_argument("--out", default=os.path.join(HERE, "results_transplant"))
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    df = load(args.path)
    df.columns = [str(c).strip() for c in df.columns]
    if args.inspect:
        inspect(df); return

    cols = list(df.columns)
    # ---- label ----
    label = args.label or LABEL_COL or find_col(cols, LABEL_KEYS)
    if label is None:
        inspect(df); raise SystemExit("Could not find a rejection column. Re-run with --label \"<name>\".")
    y = to_binary(df[label])
    print(f"Label column: '{label}'  ->  rejection rate {np.nanmean(y):.3f} (n={int(y.notna().sum())})")

    # ---- features ----
    chosen = {}
    for k, pats in KEYS.items():
        c = FEATURE_MAP.get(k) or find_col(cols, pats, exclude=(label,))
        if c and c in df.columns and c != label: chosen[k] = c
    for c in EXTRA_FEATURES:
        if c in df.columns and c != label and c not in chosen.values(): chosen[c] = c
    print("Features used:")
    for k, c in chosen.items(): print(f"   {k:12s} <- '{c}'")
    if len(chosen) < 3:
        inspect(df); raise SystemExit("Too few features matched; edit FEATURE_MAP/EXTRA_FEATURES at the top.")

    X = df[list(dict.fromkeys(chosen.values()))].copy()
    num_cols = [c for c in X.columns if pd.to_numeric(X[c], errors="coerce").notna().mean() > 0.6]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in num_cols: X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_cols: X[c] = X[c].astype(str).str.strip().replace({"nan":np.nan,"":np.nan})
    keep = y.notna()
    X, y = X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                          ("oh",OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])
    def model(name):
        clf = {"LogReg":LogisticRegression(max_iter=1000,class_weight="balanced"),
               "RandomForest":RandomForestClassifier(n_estimators=300,random_state=0,class_weight="balanced"),
               "GradientBoosting":GradientBoostingClassifier(random_state=0)}[name]
        return Pipeline([("pre",pre),("clf",clf)])

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    metrics = {}; plt.figure(figsize=(5,4.6))
    for name in ["LogReg","RandomForest","GradientBoosting"]:
        p = cross_val_predict(model(name), X, y, cv=cv, method="predict_proba")[:,1]
        auc=roc_auc_score(y,p); acc=accuracy_score(y,p>=0.5); f1=f1_score(y,p>=0.5)
        metrics[name]=dict(auroc=round(auc,4),accuracy=round(acc,4),f1=round(f1,4))
        fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr,label=f"{name} (AUC {auc:.3f})",lw=1.8)
        print(f"{name:18s} AUROC {auc:.3f}  ACC {acc:.3f}  F1 {f1:.3f}")
    plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Graft-rejection prediction (real Mendeley data)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{args.out}/fig_transplant_roc.png", dpi=150)

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    m = model("GradientBoosting").fit(Xtr,ytr)
    pi = permutation_importance(m,Xte,yte,n_repeats=10,random_state=0,scoring="roc_auc")
    order = np.argsort(pi.importances_mean)[::-1][:12][::-1]
    plt.figure(figsize=(5.4,4)); plt.barh([list(X.columns)[i] for i in order], pi.importances_mean[order])
    plt.xlabel("Drop in AUROC when permuted"); plt.title("Rejection: feature importance"); plt.tight_layout()
    plt.savefig(f"{args.out}/fig_transplant_importance.png", dpi=150)

    meta=dict(n=int(len(y)), rejection_rate=round(float(y.mean()),4), label_column=label,
              features={k:v for k,v in chosen.items()}, metrics=metrics)
    json.dump(meta, open(f"{args.out}/metrics_transplant.json","w"), indent=2)
    print(f"\nSaved metrics + figures to {args.out}/  (n={len(y)}, rejection rate {y.mean():.3f})")

if __name__ == "__main__":
    main()
