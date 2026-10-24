"""v3-D check: global fits (no airframe FE, raw log mass) on the v2 paired subset, v3-D, and full v2; and the
mean paired change in log(3-D SPE) Dryden - white on the paired subset. usage: v3d_pair_check.py"""
import csv
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robust_stats import ols_hc3  # noqa: E402

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
M = "cross_track_rmse"
v3d = list(csv.DictReader(open("results/campaign_v3d/analysis_cruise.csv", encoding="utf-8")))
v2all = list(csv.DictReader(open("results/campaign_v2/analysis_cruise.csv", encoding="utf-8")))
v2ids = {r["run_id"]: r for r in v2all}
v2all_combined = {r["run_id"]: r for r in csv.DictReader(open("results/campaign_v2/combined_cruise.csv", encoding="utf-8"))}


def num(r, k):
    try:
        return float(r[k])
    except Exception:
        return float("nan")


def fit(rows, label):
    y = np.array([num(r, M) for r in rows])
    X = np.column_stack([np.ones(len(rows)),
                         [num(r, "turbulence_rms_m_s") for r in rows], [num(r, "mean_wind_m_s") for r in rows],
                         [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rows],
                         [num(r, "thrust_to_weight") for r in rows], [num(r, "inertia_scale") for r in rows]])
    ok = (y > 0) & np.isfinite(X).all(axis=1)
    f = ols_hc3(X[ok], np.log(y[ok]))
    b, t = f["beta"], f["t"]
    print(f"{label}: n={int(ok.sum())} R2={f['r2']:.3f} | turb {b[1]:+.3f} (t {t[1]:+.0f}) wind {b[2]:+.3f} (t {t[2]:+.0f}) "
          f"log mass {b[3]:+.3f} (t {t[3]:+.0f}) T/W {b[4]:+.3f} (t {t[4]:+.1f}) inertia {b[5]:+.3f} (t {t[5]:+.1f})")


pairs = []
for r in v3d:
    c = v2all_combined.get(r.get("paired_with_run_id") or "")
    if c is not None and num(r, M) > 0 and num(c, M) > 0:
        pairs.append((r, c))
print("v3-D rows", len(v3d), "paired with a v2 control", len(pairs),
      "controls also in canonical v2 set:", sum(1 for _, c in pairs if c["run_id"] in v2ids))
fit(v2all, "v2 full set (white)       ")
fit([c for _, c in pairs], "v2 paired subset (white)  ")
fit([r for r, _ in pairs], "v3-D Dryden (paired)     ")
d = [math.log(num(r, M)) - math.log(num(c, M)) for r, c in pairs]
m = statistics.mean(d); se = statistics.stdev(d) / math.sqrt(len(d))
print(f"mean paired change log(SPE) Dryden - white = {m:+.4f} (SE {se:.4f}, t {m/se:+.0f}); ratio exp = {math.exp(m):.3f}; "
      f"median d = {statistics.median(d):+.4f}")
