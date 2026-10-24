"""Does needing a retry predict lower (artificially favourable) tracking error?  (primary metric, HC3)

log(3-D SPE) ~ six design factors + airframe FE + RETRIED, on the canonical analysis set, where
RETRIED = 1 if the usable flight succeeded after at least one failed attempt (succeeded_after_retry).
A positive coefficient means retried flights fly slightly WORSE than their conditions predict (no
favourable-survivor effect). usage: TDR_METRIC=cross_track_rmse retry_selection_test.py <analysis_csv> <label>
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
af = sorted({r.get("airframe_id") for r in rows if r.get("airframe_id")})
cols = [np.ones(len(rows))]
names = ["intercept"]
for k, fn in (("turbulence", lambda r: num(r, "turbulence_rms_m_s")), ("mean_wind", lambda r: num(r, "mean_wind_m_s")),
              ("log_payload", lambda r: math.log1p(num(r, "payload_mass_pct") / 100.0)),
              ("thrust_to_weight", lambda r: num(r, "thrust_to_weight")),
              ("inertia_scale", lambda r: num(r, "inertia_scale")),
              ("abs_com_y_frac", lambda r: abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9))):
    cols.append(np.array([fn(r) for r in rows], float)); names.append(k)
for a in af[1:]:
    cols.append(np.array([1.0 if r.get("airframe_id") == a else 0.0 for r in rows])); names.append("AF[" + a + "]")
retried = np.array([1.0 if str(r.get("succeeded_after_retry", "")).strip().lower() in ("true", "1") else 0.0 for r in rows])
cols.append(retried); names.append("RETRIED")
X = np.column_stack(cols)
ok = (y > 0) & np.isfinite(X).all(axis=1)
fit = ols_hc3(X[ok], np.log(y[ok]))
i = names.index("RETRIED")
print(f"{sys.argv[2]}: n={int(ok.sum())}  retried flights={int(retried[ok].sum())} ({100*retried[ok].mean():.2f}%)  "
      f"RETRIED coef={fit['beta'][i]:+.4f}  HC3 se={fit['se'][i]:.4f}  t={fit['t'][i]:+.2f}")
