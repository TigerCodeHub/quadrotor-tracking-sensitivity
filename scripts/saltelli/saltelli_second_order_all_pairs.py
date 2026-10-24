"""v5 full pairwise second-order matrix: all 28 factor pairs, same estimator as saltelli_crosscheck_second_order.py.

C_ij = mean(f(BA_i) f(AB_j) - f(A) f(B)) / Var(Y), uncentered, pairwise-complete (A, B, BA_i, AB_j all valid);
Var(Y) = valid A+B rows pooled, ddof=1;  S2_ij = C_ij - S1_i - S1_j with given-data bias-corrected K=50 S1.
95% CIs: 1000 base-block bootstrap resamples (S1 and C recomputed per resample).
Primary orientation is (i, j) in the order ORDER below (i earlier = BA_i, j later = AB_j); the reversed orientation
(BA_j, AB_i) is also reported as a consistency check (both estimate Var(E[Y|X_i,X_j])).
usage: python3 scripts/saltelli/saltelli_second_order_all_pairs.py > results/campaign_saltelli_merged/second_order_all_pairs.txt
"""
import csv
import json
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
D = 8
B = 2 * D + 2
SEED_MAX = 2 ** 31 - 2
NBOOT = 1000
KMAIN = 50
prob = json.load(open("generated/designs/saltelli/saltelli_problem.json"))["problem"]
names = prob["names"]
design = list(csv.DictReader(open("generated/designs/saltelli/saltelli_all.csv", encoding="utf-8")))
nrun = len(design)
X = np.zeros((nrun, D))
for r in design:
    i = int(r["saltelli_index"])
    X[i] = [float(r["payload_mass_pct"]) / 100.0, float(r["turbulence_rms_m_s"]), float(r["mean_wind_m_s"]),
            float(r["thrust_to_weight"]), float(r["inertia_scale"]), float(r["com_y_m"]) / float(r["wheelbase_m"]),
            float(r["wind_direction_deg"]), float(r["wind_seed"]) / SEED_MAX]
rid2i = {r["run_id"]: int(r["saltelli_index"]) for r in design}
y = np.full(nrun, np.nan)
for r in csv.DictReader(open(os.environ.get("TDR_V5_CSV", "results/campaign_saltelli_merged/analysis_cruise.csv"), encoding="utf-8")):
    try:
        v = float(r.get("cross_track_rmse") or "nan")
    except ValueError:
        continue
    if r["run_id"] in rid2i and math.isfinite(v) and v > 0:
        y[rid2i[r["run_id"]]] = math.log(v)
nbl = nrun // B
Xb = X.reshape(nbl, B, D)
yb = y.reshape(nbl, B)
print(f"problem names (column order): {names}")
print(f"valid runs {int(np.isfinite(y).sum())} of {nrun}; blocks {nbl}")

# requested factor order -> column indices in the design
ORDER = [("turbulence", 1), ("mean wind", 2), ("gust seed", 7), ("T/W", 3), ("CoM", 5), ("payload", 0),
         ("wind direction", 6), ("inertia", 4)]
PAIRS = [(a, b) for a in range(D) for b in range(a + 1, D)]  # positions in ORDER


def s1_bc(xcol, ycol, K=KMAIN):
    o = np.argsort(xcol, kind="stable"); yo = ycol[o]; n = len(yo)
    e = np.linspace(0, n, K + 1).astype(int)
    means = np.array([yo[e[k]:e[k + 1]].mean() for k in range(K)])
    within = np.array([yo[e[k]:e[k + 1]].var(ddof=1) / (e[k + 1] - e[k]) for k in range(K)])
    w = np.diff(e) / n
    return float((w * ((means - ycol.mean()) ** 2 - within)).sum() / ycol.var())


def stats(jb):
    A = yb[jb, 0]; Bv = yb[jb, B - 1]; AB = yb[jb, 1:1 + D]; BA = yb[jb, 1 + D:1 + 2 * D]
    varY = np.concatenate([A[np.isfinite(A)], Bv[np.isfinite(Bv)]]).var(ddof=1)
    yy = yb[jb].reshape(-1); mk = np.isfinite(yy)
    S1 = np.array([s1_bc(Xb[jb][:, :, i].reshape(-1)[mk], yy[mk]) for i in range(D)])
    C = {}; nC = {}
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            m = np.isfinite(A) & np.isfinite(Bv) & np.isfinite(BA[:, i]) & np.isfinite(AB[:, j])
            C[(i, j)] = (BA[m, i] * AB[m, j] - A[m] * Bv[m]).mean() / varY; nC[(i, j)] = int(m.sum())
    return {"varY": varY, "S1": S1, "C": C, "nC": nC}


rng = np.random.default_rng(20261012)
pt = stats(np.arange(nbl))
print(f"Var(Y) = {pt['varY']:.5f}")
boot = [stats(rng.integers(0, nbl, nbl)) for _ in range(NBOOT)]


def ci(getter):
    v = np.array([getter(s) for s in boot])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


print("\nS1 (given-data bias-corrected, K=50):")
for nm, c in ORDER:
    print(f"   {nm:<16}{pt['S1'][c]:+.4f}")
print(f"\nS2_ij = C_ij - S1_i - S1_j, primary orientation BA_i x AB_j with i earlier in the requested order ({NBOOT} block resamples)")
print(f"{'pair':<30}{'C_ij':>9}{'S2_ij':>9}{'95% CI':>20}{'n blocks':>10}{'S2 reversed':>13}{'rev n':>8}")
for a, b in PAIRS:
    (ni, i), (nj, j) = ORDER[a], ORDER[b]
    s2 = pt["C"][(i, j)] - pt["S1"][i] - pt["S1"][j]
    lo, hi = ci(lambda s, i=i, j=j: s["C"][(i, j)] - s["S1"][i] - s["S1"][j])
    s2r = pt["C"][(j, i)] - pt["S1"][i] - pt["S1"][j]
    print(f"{ni + ' x ' + nj:<30}{pt['C'][(i, j)]:>9.4f}{s2:>+9.4f}   [{lo:+.4f},{hi:+.4f}]{pt['nC'][(i, j)]:>9}{s2r:>+13.4f}{pt['nC'][(j, i)]:>8}")
