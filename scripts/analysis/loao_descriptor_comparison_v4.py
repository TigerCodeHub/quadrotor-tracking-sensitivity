"""LOAO airframe-descriptor comparison on the v4 (delivery) primary-footing analysis set (c1-c4); common features add 9 plan dummies + chunk dummies.
Mirrors loao_descriptor_comparison.py (v2):  LOAO airframe-descriptor comparison on the v2 primary-footing analysis set (same setup as primary_footing.py section b, WITH direction terms).

Features: intercept, six primary factors (turb, wind, log1p(payload), T/W, inertia, |com_y|/wheelbase), 3 direction level shifts (tailwind base),
wind x dir and turb x dir.  OLS, train on 4 airframes, held-out R2 = 1 - SS_res/SS_tot about the held-out airframe's own mean (raw, no recentring).
V0 none | V1 + log nominal_mass_kg | V2 + log wheelbase + log disk loading (current paper model) | V3 + log m0 + log wheelbase + log disk loading
CEILING: same factors+direction fit and scored in-sample on each airframe.  Also pooled in-sample R2 with airframe FE.
usage (fleet): python scripts/analysis/loao_descriptor_comparison.py > results/campaign_v4/loao_descriptor_comparison.txt
"""
import csv
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
M = os.environ.get("TDR_METRIC", "cross_track_rmse")


def num(r, k):
    try:
        v = float(r[k])
        return v if math.isfinite(v) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


rows = [r for r in csv.DictReader(open("results/campaign_v4/analysis_cruise.csv", encoding="utf-8")) if num(r, M) > 0]
n = len(rows)
y = np.log(np.array([num(r, M) for r in rows]))
arf = np.array([r["airframe_id"] for r in rows])
afs = sorted(set(arf))
col = lambda f: np.array([f(r) for r in rows])
turb = col(lambda r: num(r, "turbulence_rms_m_s")); wind = col(lambda r: num(r, "mean_wind_m_s"))
lpay = col(lambda r: math.log1p(num(r, "payload_mass_pct") / 100.0)); tw = col(lambda r: num(r, "thrust_to_weight"))
inert = col(lambda r: num(r, "inertia_scale")); comf = col(lambda r: abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9))
d = col(lambda r: int(round(num(r, "wind_direction_deg"))) % 360)
lm0 = col(lambda r: math.log(num(r, "nominal_mass_kg")))
lwb = col(lambda r: math.log(max(num(r, "wheelbase_m"), 1e-9)))
ldl = col(lambda r: math.log(max(num(r, "disk_loading_kg_m2"), 1e-9)))
dircols = []
for k in (90, 180, 270):
    dircols += [(d == k).astype(float), (d == k) * wind, (d == k) * turb]
plan = col(lambda r: r["plan"]); chunk = col(lambda r: r["route_chunk"])
fe = [(plan == a).astype(float) for a in sorted(set(plan))[1:]] + [(chunk == a).astype(float) for a in sorted(set(chunk))[1:]]
core = [np.ones(n), turb, wind, lpay, tw, inert, comf] + dircols + fe
VAR = {"V0 none": [], "V1 + log m0": [lm0], "V2 + log wheelbase + log disk loading": [lwb, ldl],
       "V3 + log m0 + log wheelbase + log disk loading": [lm0, lwb, ldl]}
print(f"n={n}; airframes {afs}; counts {[int((arf == a).sum()) for a in afs]}")


def r2(yt, p):
    return 1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


print(f"\n{'variant':<50}" + "".join(f"{a:>9}" for a in afs) + f"{'MEAN':>9}{'min':>8}{'max':>8}")
for nm, extra in VAR.items():
    X = np.column_stack(core + extra)
    out = []
    for h in afs:
        te = arf == h
        b, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None)
        out.append(r2(y[te], X[te] @ b))
    print(f"{nm:<50}" + "".join(f"{v:>9.3f}" for v in out) + f"{np.mean(out):>9.3f}{min(out):>8.3f}{max(out):>8.3f}")
X = np.column_stack(core)
out = []
for h in afs:
    te = arf == h
    b, *_ = np.linalg.lstsq(X[te], y[te], rcond=None)
    out.append(r2(y[te], X[te] @ b))
print(f"{'CEILING (in-sample per airframe)':<50}" + "".join(f"{v:>9.3f}" for v in out) + f"{np.mean(out):>9.3f}{min(out):>8.3f}{max(out):>8.3f}")
Xfe = np.column_stack(core + [(arf == a).astype(float) for a in afs[1:]])
b, *_ = np.linalg.lstsq(Xfe, y, rcond=None)
print(f"\npooled in-sample R2, factors + direction + AF dummies: {r2(y, Xfe @ b):.4f}   (without AF dummies: "
      f"{r2(y, X @ np.linalg.lstsq(X, y, rcond=None)[0]):.4f})")
print("airframe-level descriptors:")
for a in afs:
    m = arf == a
    print(f"   {a:<8} m0={np.exp(lm0[m][0]):.3f} kg  wheelbase={np.exp(lwb[m][0]):.3f} m  disk loading={np.exp(ldl[m][0]):.2f} kg/m2")

print("\nPER-HELD-OUT-AIRFRAME bias = mean(y - yhat) [log units], level error 100*(exp(-bias)-1)% of the prediction, RMSE, within-airframe SD(y)")
print("(level error: prediction = truth * exp(-bias), so the model is off by 100*(exp(-bias)-1)% relative to truth)")
for nm, extra in VAR.items():
    X = np.column_stack(core + extra)
    print(f"\n{nm}\n   {'held out':<10}{'bias':>9}{'level err %':>13}{'RMSE':>9}{'SD(y)':>9}")
    bs_, rm_ = [], []
    for h in afs:
        te = arf == h
        b, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None)
        res = y[te] - X[te] @ b
        bias = res.mean(); rmse = math.sqrt((res ** 2).mean())
        bs_.append(bias); rm_.append(rmse)
        print(f"   {h:<10}{bias:>+9.3f}{100 * (math.exp(-bias) - 1):>+12.1f}%{rmse:>9.3f}{y[te].std():>9.3f}")
    print(f"   {'mean|bias|':<10}{np.mean(np.abs(bs_)):>9.3f}{'':>13}{np.mean(rm_):>9.3f}")
