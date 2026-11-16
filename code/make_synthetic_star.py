"""
Generate a synthetic STAR-SHAPED CSV to validate star_pipeline.py before the real extract arrives.
Columns mimic OPTN/UNOS STAR fields (renamed generically); outcomes carry a shared frailty, nonlinear
structure, and informative (covariate-dependent) censoring so the pipeline's models are exercised
realistically. This is NOT real data --- it only checks the pipeline runs end-to-end and the metrics
are well-formed. Swap in the real file and the documented column mapping to run for real.
"""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(20261116)
N = 30000
rec_age = np.clip(rng.normal(50, 14, N), 18, 85)
don_age = np.clip(rng.normal(42, 16, N), 1, 80)
hla_mm = rng.integers(0, 7, N).astype(float)
cpra = np.where(rng.random(N) < 0.2, rng.uniform(50, 99, N), rng.beta(1.3, 6, N)*100)
cold_isch = np.clip(rng.gamma(3, 4, N), 1, 40)          # hours
dial_vintage = np.clip(rng.gamma(2, 1.8, N), 0, 20)     # years
don_bmi = np.clip(rng.normal(27, 5, N), 15, 50)
rec_bmi = np.clip(rng.normal(28, 6, N), 15, 55)
diabetes = rng.binomial(1, 0.33, N)
abo_mm = rng.binomial(1, 0.12, N)
dcd = rng.binomial(1, 0.22, N)                           # donation after circulatory death
regraft = rng.binomial(1, 0.11, N)

u = rng.normal(0, 1, N)                                  # shared frailty (unobserved)
# log-hazard with nonlinear + interaction structure Cox cannot fully represent
lp = (0.028*(rec_age-50) + 0.020*(don_age-42) + 0.14*hla_mm + 0.012*cpra
      + 0.035*cold_isch + 0.06*dial_vintage + 0.015*(rec_bmi-28)
      + 0.45*diabetes + 0.5*abo_mm + 0.35*dcd + 0.6*regraft
      + 0.18*(hla_mm/6)*(cpra>80) + 0.03*np.maximum(cold_isch-18, 0) + 0.25*(dial_vintage>8)
      + 0.7*u)
base = 0.012
T = rng.exponential(1.0/(base*np.exp(lp - lp.mean())))   # event time (years)
# informative censoring: sicker (higher lp) followed slightly less, plus admin censor at 12 y
C = np.minimum(rng.exponential(10.0/np.exp(0.15*(lp-lp.mean()))), 12.0)
GTIME_Y = np.minimum(T, C)
GSTATUS = (T <= C).astype(int)

df = pd.DataFrame(dict(GTIME_Y=np.round(GTIME_Y, 3), GSTATUS=GSTATUS,
    rec_age=np.round(rec_age, 1), don_age=np.round(don_age, 1), hla_mm=hla_mm.astype(int),
    cpra=np.round(cpra, 1), cold_isch=np.round(cold_isch, 1), dial_vintage=np.round(dial_vintage, 2),
    don_bmi=np.round(don_bmi, 1), rec_bmi=np.round(rec_bmi, 1),
    diabetes=diabetes, abo_mm=abo_mm, dcd=dcd, regraft=regraft))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_star.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: n={N}, events={GSTATUS.sum()} ({GSTATUS.mean()*100:.1f}%), median follow-up={np.median(GTIME_Y):.1f} y")
