"""v2 same-seed replicate noise (cross-check 1).

v2's duplicate stratum (repeat 2) re-flies 8,190 configs with identical parameters AND identical wind_seed.
For every config whose repeat-1 and repeat-2 flights are both valid (primary metric finite >0, not a failed-attempt row,
not a 2x-takeoff row), d = log(3-D SPE)_rep2 - log(3-D SPE)_rep1.  Noise variance = Var(d)/2.
Compared with (a) total Var(log SPE) of the v2 analysis set and (b) the residual variance of the primary model (six factors + AF FE).
Also noise variance by turbulence tercile and by mean-wind tercile.  usage: python scripts/analysis/v2_replicate_noise.py
"""
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robust_stats import ols_hc3_rank  # noqa: E402

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
M = "cross_track_rmse"


def num(r, k):
    try:
        v = float(r[k]); return v if math.isfinite(v) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


comb = list(csv.DictReader(open("results/campaign_v2/combined_cruise.csv", encoding="utf-8")))


def ok(r):
    return r.get("from_failed_attempt") != "1" and not (r.get("takeoff_sim_speed_factor") or "").startswith("2") and num(r, M) > 0


by = {}
for r in comb:
    if ok(r):
        by.setdefault((r["airframe_id"], r["sample_index"]), {})[r["repeat"]] = r
pairs = [(v["1"], v["2"]) for v in by.values() if "1" in v and "2" in v]
d = np.array([math.log(num(b, M)) - math.log(num(a, M)) for a, b in pairs])
n = len(d)
print(f"configs with both repeats valid: n = {n}")
print(f"  seed identical in all pairs: {all(a['wind_seed'] == b['wind_seed'] for a, b in pairs)}; params identical (wind, turb, payload): "
      f"{all(a['mean_wind_m_s'] == b['mean_wind_m_s'] and a['turbulence_rms_m_s'] == b['turbulence_rms_m_s'] and a['payload_mass_pct'] == b['payload_mass_pct'] for a, b in pairs)}")
print(f"  mean d = {d.mean():+.5f} (SE {d.std(ddof=1)/math.sqrt(n):.5f}); sd(d) = {d.std(ddof=1):.5f}; fraction d == 0 exactly: {(d == 0).mean():.3f}")
noise = d.var(ddof=1) / 2.0
print(f"  same-seed noise variance Var(d)/2 = {noise:.6f}  (noise sd {math.sqrt(noise):.4f} log units = {100*(math.exp(math.sqrt(noise))-1):.1f}% per flight)")
an = [r for r in csv.DictReader(open("results/campaign_v2/analysis_cruise.csv", encoding="utf-8")) if num(r, M) > 0]
y = np.log(np.array([num(r, M) for r in an]))
tot = y.var(ddof=1)
print(f"  total Var(log SPE) of the v2 analysis set (n={len(y)}) = {tot:.5f} -> noise share {100*noise/tot:.3f}%")
afs = sorted({r["airframe_id"] for r in an})
cols = [np.ones(len(an))]
for k, fn in (("turb", lambda r: num(r, "turbulence_rms_m_s")), ("wind", lambda r: num(r, "mean_wind_m_s")),
              ("lpay", lambda r: math.log1p(num(r, "payload_mass_pct") / 100)), ("tw", lambda r: num(r, "thrust_to_weight")),
              ("inert", lambda r: num(r, "inertia_scale")), ("comf", lambda r: abs(num(r, "com_y_m")) / num(r, "wheelbase_m"))):
    cols.append(np.array([fn(r) for r in an]))
for a in afs[1:]:
    cols.append(np.array([1.0 if r["airframe_id"] == a else 0.0 for r in an]))
f = ols_hc3_rank(np.column_stack(cols), y, ["x%d" % i for i in range(len(cols))])
resid_var = float(f["resid"] @ f["resid"]) / (len(y) - len(f["names"]))
print(f"  residual variance of the primary model (six factors + AF FE; R2 {f['r2']:.4f}) = {resid_var:.5f} -> noise share {100*noise/resid_var:.2f}% of residual variance")
print(f"  (explained by model: {100*f['r2']:.1f}% of total; residual {100*(1-f['r2']):.1f}% of total; noise is {100*noise/tot:.2f}% of total)")
for nm, col in (("turbulence", "turbulence_rms_m_s"), ("mean wind", "mean_wind_m_s")):
    x = np.array([num(a, col) for a, b in pairs])
    q = np.quantile(x, [1 / 3, 2 / 3])
    print(f"  noise variance Var(d)/2 and mean |d| by {nm} tercile (cut points {q[0]:.2f}, {q[1]:.2f}):")
    for lbl, m in (("low", x <= q[0]), ("mid", (x > q[0]) & (x <= q[1])), ("high", x > q[1])):
        print(f"      {lbl:<5} n={int(m.sum()):>5}  Var(d)/2={d[m].var(ddof=1)/2:.6f}  mean|d|={np.abs(d[m]).mean():.5f}")
