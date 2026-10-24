"""Global (no airframe fixed effects) fit with RAW log(mass) for the cross-experiment transfer table.

Six factors (turbulence, wind, log mass, T/W, inertia, |com_y|/wheelbase), no airframe dummies, HC3.
Reports the coefficients so the table's global log-mass column can be regenerated.
usage: TDR_METRIC=cross_track_rmse global_logmass.py <analysis_csv> <label>
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robust_stats import ols_hc3  # noqa: E402
from refit_robust import load, num  # noqa: E402

METRIC = os.environ.get("TDR_METRIC", "cross_track_rmse")
rows = load(sys.argv[1])
y = np.array([num(r, METRIC) for r in rows])
cols = {
    "turbulence": [num(r, "turbulence_rms_m_s") for r in rows],
    "mean_wind": [num(r, "mean_wind_m_s") for r in rows],
    "log_mass": [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rows],
    "thrust_to_weight": [num(r, "thrust_to_weight") for r in rows],
    "inertia_scale": [num(r, "inertia_scale") for r in rows],
    "abs_com_y_frac": [abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9) for r in rows],
}
names = list(cols)
M = np.column_stack([np.ones(len(rows))] + [np.array(cols[k], float) for k in names])
ok = (y > 0) & np.isfinite(M).all(axis=1)
fit = ols_hc3(M[ok], np.log(y[ok]))
print(f"GLOBAL (no FE, raw log mass) {sys.argv[2]}: n={int(ok.sum())} R2={fit['r2']:.3f}")
for nm, b, t in zip(["intercept"] + names, fit["beta"], fit["t"]):
    print(f"  {nm:<18}{b:>+9.4f}  (t={t:+.1f})")
