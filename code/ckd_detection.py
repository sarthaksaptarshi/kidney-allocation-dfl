"""
CKD/ESRD DETECTION ARM on fully-open, no-license data (UCI Chronic Kidney Disease).

Data: UCI CKD dataset #336 (400 patients, Tamil Nadu, India). Free, openly licensed.
No application or data-use agreement is required.

Two ways to supply the data (the script tries them in order):
  1.  pip install ucimlrepo   -> the script auto-downloads it, OR
  2.  --csv path/to/ckd.csv   -> point it at a CSV you downloaded yourself.

What it does: cleans the data, trains three models (logistic regression, random
forest, gradient boosting) with 5-fold cross-validation, prints accuracy / AUROC /
F1, and saves a ROC figure, a feature-importance figure, and a metrics JSON.

Run:   python ckd_detection.py
   or: python ckd_detection.py --csv ckd.csv --out results_ckd
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.inspection import permutation_importance

NUMERIC = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
BINARY  = {"rbc":("normal","abnormal"),"pc":("normal","abnormal"),
           "pcc":("present","notpresent"),"ba":("present","notpresent"),
           "htn":("yes","no"),"dm":("yes","no"),"cad":("yes","no"),
           "appet":("good","poor"),"pe":("yes","no"),"ane":("yes","no")}

def clean_text(s):
    return (s.astype(str).str.strip().str.replace("\t","",regex=False)
             .str.replace(" ","",regex=False).str.lower().replace({"nan":np.nan,"?":np.nan,"":np.nan}))

def load_data(csv):
    if csv:
        df = pd.read_csv(csv)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            sys.exit("Install the loader first:  pip install ucimlrepo   (or pass --csv)")
        ds = fetch_ucirepo(id=336)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [c.strip().lower() for c in df.columns]
    # the target column is 'class' (values ckd / notckd)
    tgt = "class" if "class" in df.columns else df.columns[-1]
    return df, tgt

def prepare(df, tgt):
    y = clean_text(df[tgt]).map(lambda v: 1 if isinstance(v,str) and v.startswith("ckd") else (0 if isinstance(v,str) else np.nan))
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        if c in df.columns:
            X[c] = pd.to_numeric(clean_text(df[c]), errors="coerce")
    for c,(pos,neg) in BINARY.items():
        if c in df.columns:
            v = clean_text(df[c]); X[c] = v.map({pos:1, neg:0})
    keep = y.notna()
    return X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)

def model(name):
    if name=="LogReg":
        return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                         ("clf",LogisticRegression(max_iter=1000))])
    if name=="RandomForest":
        return Pipeline([("imp",SimpleImputer(strategy="median")),
                         ("clf",RandomForestClassifier(n_estimators=300,random_state=0))])
    return Pipeline([("imp",SimpleImputer(strategy="median")),
                     ("clf",GradientBoostingClassifier(random_state=0))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path to a local CKD csv (optional)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"results_ckd"), help="output folder")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    df, tgt = load_data(args.csv)
    X, y = prepare(df, tgt)
    print(f"Loaded {len(X)} patients, {X.shape[1]} features, CKD prevalence {y.mean():.2f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    metrics = {}
    plt.figure(figsize=(5,4.6))
    for name in ["LogReg","RandomForest","GradientBoosting"]:
        p = cross_val_predict(model(name), X, y, cv=cv, method="predict_proba")[:,1]
        auc = roc_auc_score(y,p); acc = accuracy_score(y,(p>=0.5)); f1 = f1_score(y,(p>=0.5))
        metrics[name] = dict(auroc=round(auc,4), accuracy=round(acc,4), f1=round(f1,4))
        fpr,tpr,_ = roc_curve(y,p); plt.plot(fpr,tpr,label=f"{name} (AUC {auc:.3f})",lw=1.8)
        print(f"{name:18s} AUROC {auc:.3f}  ACC {acc:.3f}  F1 {f1:.3f}")
    plt.plot([0,1],[0,1],"k--",lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("CKD detection (UCI dataset)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{args.out}/fig_ckd_roc.png", dpi=150)

    # feature importance (gradient boosting, held-out split)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    m = model("GradientBoosting").fit(Xtr,ytr)
    pi = permutation_importance(m,Xte,yte,n_repeats=10,random_state=0,scoring="roc_auc")
    order = np.argsort(pi.importances_mean)[::-1][:12][::-1]
    plt.figure(figsize=(5.2,4)); plt.barh([X.columns[i] for i in order], pi.importances_mean[order])
    plt.xlabel("Drop in AUROC when permuted"); plt.title("CKD feature importance"); plt.tight_layout()
    plt.savefig(f"{args.out}/fig_ckd_importance.png", dpi=150)

    json.dump(metrics, open(f"{args.out}/metrics_ckd.json","w"), indent=2)
    print(f"\nSaved figures + metrics to {args.out}/")

if __name__ == "__main__":
    main()
