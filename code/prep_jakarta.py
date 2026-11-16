"""
Prepare the public Jakarta kidney-transplant cohort (Dryad doi:10.5061/dryad.41ns1rnfr) for
star_pipeline.py. Converts the month-scale survival times to years, keeps PRE-transplant predictors
only (drops the post-transplant leakage variables Rejection / Infections / Delayed_Graft), and writes
a clean CSV with both the death-censored and all-cause graft-loss endpoints.

Usage: python prep_jakarta.py [survival_data.xlsx]   ->  jakarta_clean.csv
"""
import sys, os
import pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "survival_data.xlsx"
df = pd.read_excel(src)

out = pd.DataFrame({
    "PT_DC_y": df["PT_DC"] / 12.0, "DC_Graft": df["DC_Graft"].astype(int),   # death-censored graft loss (paper's endpoint)
    "PT_AC_y": df["PT_AC"] / 12.0, "AC_Graft": df["AC_Graft"].astype(int),   # all-cause graft failure (more events)
    # pre-transplant predictors only (NO Rejection / Infections / Delayed_Graft = post-transplant leakage)
    "rec_age60": df["Age60"], "rec_overweight": df["Overweight"], "dial_gt24mo": df["Duration24"],
    "smoking": df["Smoking"], "diabetes": df["Diabetes"], "cardiovascular": df["Cardiovascular"],
    "dialysis_type": df["Dialysis_type"], "crossmatch": df["Crossmatch"],
    "don_age40": df["Donor_age40"], "related_donor": df["Related"],
    "don_overweight": df["Donor_overweight"], "don_gfr100": df["Donor_GFR100"],
})
dst = os.path.join(os.path.dirname(os.path.abspath(src)) if os.path.dirname(src) else ".", "jakarta_clean.csv")
dst = "jakarta_clean.csv"
out.to_csv(dst, index=False)
print(f"wrote {dst}: n={len(out)}")
print(f"  death-censored graft loss: {int(out.DC_Graft.sum())} events ({out.DC_Graft.mean()*100:.1f}%)")
print(f"  all-cause graft failure:   {int(out.AC_Graft.sum())} events ({out.AC_Graft.mean()*100:.1f}%)")
print(f"  predictors: 12 pre-transplant (leakage vars excluded)")
