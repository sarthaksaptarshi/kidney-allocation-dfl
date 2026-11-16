"""
Turn experiment output (out/summary.json) into paper-ready LaTeX, so you never
hand-copy a number into the manuscript.

Run AFTER sim.py:   python make_tables.py
It prints:
  1. the AUROC table rows  (paste into Table II of paper.tex)
  2. the ablation table rows (paste into Table IV)
  3. every inline number used in the text, with the value to check/replace.
Optionally reads results_ckd/metrics_ckd.json if you ran the CKD arm.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

s = json.load(open(os.path.join(HERE, "out", "summary.json")))
a = s["auroc"]; al = s["allocation"]; d = s["differential"]; ab = s["ablation"]; mono = s["monotonicity"]; cal = s["calibration"]

def row(name, key):
    return f"{name} & {a['1mo'][key]:.3f} & {a['1yr'][key]:.3f} & {a['10yr'][key]:.3f} \\\\"

print("="*64, "\n1) AUROC TABLE ROWS  ->  Table II (tab:auroc) in paper.tex\n"+"="*64)
print(row("Logistic regression", "LogReg"))
print(row("Random forest", "RandForest"))
print(row("Gradient boosting (single)", "GBM_single").replace(f"{a['1yr']['GBM_single']:.3f}", f"\\textbf{{{a['1yr']['GBM_single']:.3f}}}"))
print(row("\\textbf{Proposed (survival, MH)}", "Proposed_MH").replace(f"{a['10yr']['Proposed_MH']:.3f}", f"\\textbf{{{a['10yr']['Proposed_MH']:.3f}}}"))
print(f"% monotonicity: proposed {mono['violation_rate_proposed']*100:.1f}%  vs independent {mono['violation_rate_independent']*100:.1f}%")

print("\n"+"="*64, "\n2) ABLATION TABLE ROWS  ->  Table IV (tab:abl)\n"+"="*64)
full = ab["full_proposed"]
for label,key in [("Full model","full_proposed"),("$-$ HLA, cPRA, DSA","no_HLA_cPRA_DSA"),
                  ("$-$ tacrolimus, CYP3A5","no_tac_cyp3a5"),("$-$ cold ischemia","no_cold_ischemia"),
                  ("$-$ dialysis vintage","no_dialysis_vintage")]:
    v=ab[key]; delta="--" if key=="full_proposed" else f"{v-full:+.3f}"
    print(f"{label} & {v:.3f} & {delta} \\\\")

print("\n"+"="*64, "\n3) INLINE NUMBERS  ->  check/replace in abstract + Section V text\n"+"="*64)
pairs = [
 ("10-yr AUROC, proposed", f"{a['10yr']['Proposed_MH']:.3f}"),
 ("10-yr AUROC, best baseline", f"{a['10yr']['GBM_single']:.3f}"),
 ("monotonicity violations, independent", f"{mono['violation_rate_independent']*100:.1f}%"),
 ("calibration ECE raw -> cal", f"{cal['ece_raw']:.3f} -> {cal['ece_cal']:.3f}"),
 ("Brier raw -> cal", f"{cal['brier_raw']:.3f} -> {cal['brier_cal']:.3f}"),
 ("conformal coverage", f"{cal['conformal_coverage']*100:.1f}%"),
 ("abstention / confident-AUROC", f"{cal['abstain_rate']*100:.0f}% / {cal['auc_on_confident']:.3f}"),
 ("AMR/tox AUROC", f"{d['auroc']:.3f}"),
 ("AMR/tox accuracy raw -> gated", f"{d['accuracy']:.3f} -> {d['accuracy_gated']:.3f}"),
 ("AMR/tox macro-F1 raw -> gated", f"{d['macro_f1']:.3f} -> {d['macro_f1_gated']:.3f}"),
 ("biopsy referral rate", f"{d['referral_rate']*100:.1f}%"),
 ("regret two-stage -> decision-focused", f"{al['regret_two_stage_pct']}% -> {al['regret_decision_focused_pct']}%"),
 ("price of fairness", f"{al['price_of_fairness_pct']}%"),
 ("female access util -> fair", f"{al['female_access_util']} -> {al['female_access_fair']}"),
 ("sensitized access util -> fair", f"{al['sensitized_access_util']} -> {al['sensitized_access_fair']}"),
 ("pediatric access util -> fair", f"{al['pediatric_access_util']} -> {al['pediatric_access_fair']}"),
]
for k,v in pairs: print(f"  {k:42s}: {v}")

_ckd = os.path.join(HERE, "results_ckd", "metrics_ckd.json")
if os.path.exists(_ckd):
    print("\n"+"="*64, "\n4) CKD-DETECTION ARM (real UCI data)\n"+"="*64)
    for m,vals in json.load(open(_ckd)).items():
        print(f"  {m:18s} AUROC {vals['auroc']}  ACC {vals['accuracy']}  F1 {vals['f1']}")
print()
