"""v5 cross-check 2: closed second-order indices from the Saltelli BA_i rows, pairwise-complete.

Block layout (SALib, second order): [A, AB_1..AB_D, BA_1..BA_D, B].  AB_j = A with column j from B; BA_i = B with column i from A.
f(BA_i) and f(AB_j) (i != j) share exactly coordinates i and j, so
    Vc_ij = mean( f(BA_i) f(AB_j) - f(A) f(B) )  = Var(E[Y | X_i, X_j])      (closed second-order variance),   C_ij = Vc_ij / Var(Y)
over every base block in which A, B, BA_i and AB_j are all valid (pairwise-complete).  Var(Y) = sample variance (ddof=1) of the valid A and B rows.
Interaction S2_ij = C_ij - S1_i - S1_j with the given-data bias-corrected K=50 S1 (equal-count binning on all valid runs).
Compared with ST_i - S1_i (Jansen, pairwise-complete) for j = gust seed; and 1 - ST_seed.  Block bootstrap, 1000 resamples (S1, ST, C all recomputed per resample).
usage: python3 scripts/saltelli/saltelli_crosscheck_second_order.py
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
# structure checks
ok_ab = ok_ba = 0
for b in range(200):
    A_, B_ = Xb[b, 0], Xb[b, B - 1]
    for i in range(D):
        e1 = A_.copy(); e1[i] = B_[i]
        e2 = B_.copy(); e2[i] = A_[i]
        ok_ab += int(np.allclose(Xb[b, 1 + i], e1, rtol=1e-6, atol=1e-9))
        ok_ba += int(np.allclose(Xb[b, 1 + D + i], e2, rtol=1e-6, atol=1e-9))
print(f"structure check (200 blocks x 8): AB_i == A with col i from B: {ok_ab}/1600; BA_i == B with col i from A: {ok_ba}/1600")
print(f"valid runs {int(np.isfinite(y).sum())} of {nrun}; blocks {nbl}")


def s1_bc(xcol, ycol, K=KMAIN):
    o = np.argsort(xcol, kind="stable"); yo = ycol[o]; n = len(yo)
    e = np.linspace(0, n, K + 1).astype(int)
    means = np.array([yo[e[k]:e[k + 1]].mean() for k in range(K)])
    within = np.array([yo[e[k]:e[k + 1]].var(ddof=1) / (e[k + 1] - e[k]) for k in range(K)])
    w = np.diff(e) / n
    return float((w * ((means - ycol.mean()) ** 2 - within)).sum() / ycol.var())


JS = {"seed": 7, "turbulence": 1, "mean wind": 2}


def stats(jb):
    """All statistics on the block resample jb (array of block indices). Returns dict."""
    A = yb[jb, 0]; Bv = yb[jb, B - 1]; AB = yb[jb, 1:1 + D]; BA = yb[jb, 1 + D:1 + 2 * D]
    varY = np.concatenate([A[np.isfinite(A)], Bv[np.isfinite(Bv)]]).var(ddof=1)
    yy = yb[jb].reshape(-1); mk = np.isfinite(yy)
    S1 = np.array([s1_bc(Xb[jb][:, :, i].reshape(-1)[mk], yy[mk]) for i in range(D)])
    ST = np.empty(D); nST = np.empty(D, int)
    for i in range(D):
        m = np.isfinite(A) & np.isfinite(AB[:, i])
        ST[i] = (((A[m] - AB[m, i]) ** 2).sum() / (2.0 * m.sum())) / varY; nST[i] = int(m.sum())
    C = {}; nC = {}
    for jn, j in JS.items():
        for i in range(D):
            if i == j:
                continue
            m = np.isfinite(A) & np.isfinite(Bv) & np.isfinite(BA[:, i]) & np.isfinite(AB[:, j])
            q = BA[m, i] * AB[m, j] - A[m] * Bv[m]
            C[(i, jn)] = q.mean() / varY; nC[(i, jn)] = int(m.sum())
    return {"varY": varY, "S1": S1, "ST": ST, "nST": nST, "C": C, "nC": nC}


rng = np.random.default_rng(20261012)
pt = stats(np.arange(nbl))
print(f"Var(Y) (valid A and B rows pooled, ddof=1) = {pt['varY']:.5f}")
boot = []
for b in range(NBOOT):
    boot.append(stats(rng.integers(0, nbl, nbl)))


def ci(getter):
    v = np.array([getter(s) for s in boot])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


print("\nPoint estimates (given-data bias-corrected K=50 S1; pairwise Jansen S_T):")
for i, nm in enumerate(names):
    print(f"   {nm:<20} S1={pt['S1'][i]:+.4f}  S_T={pt['ST'][i]:.4f}  S_T-S1={pt['ST'][i]-pt['S1'][i]:.4f}")
for jn, j in JS.items():
    print(f"\nCLOSED second-order with j = {jn}: C_ij = Var(E[Y|X_i,X_j])/Var(Y); S2_ij = C_ij - S1_i - S1_j   (95% block-bootstrap CI, {NBOOT} resamples)")
    print(f"   {'i':<20}{'C_ij':>9}{'S1_i':>9}{'S1_j':>9}{'S2_ij':>9}{'S2 CI':>20}{'n blocks':>10}")
    for i, nm in enumerate(names):
        if i == j:
            continue
        c = pt["C"][(i, jn)]
        s2 = c - pt["S1"][i] - pt["S1"][j]
        lo, hi = ci(lambda s, i=i, j=j, jn=jn: s["C"][(i, jn)] - s["S1"][i] - s["S1"][j])
        print(f"   {nm:<20}{c:>9.4f}{pt['S1'][i]:>9.4f}{pt['S1'][j]:>9.4f}{s2:>+9.4f}   [{lo:+.4f},{hi:+.4f}]{pt['nC'][(i, jn)]:>9}")
print("\nSEED INTERACTION SHARE: S2_{i,seed} vs ST_i - S1_i (the factor's whole non-first-order mass)")
print(f"   {'factor':<20}{'ST_i-S1_i':>11}{'S2_i,seed':>11}{'CI of S2':>22}{'share':>8}")
for i, nm in enumerate(names):
    if i == 7:
        continue
    nf = pt["ST"][i] - pt["S1"][i]
    s2 = pt["C"][(i, "seed")] - pt["S1"][i] - pt["S1"][7]
    lo, hi = ci(lambda s, i=i: s["C"][(i, "seed")] - s["S1"][i] - s["S1"][7])
    print(f"   {nm:<20}{nf:>11.4f}{s2:>+11.4f}   [{lo:+.4f},{hi:+.4f}]{s2/nf:>8.2f}")
lo, hi = ci(lambda s: 1 - s["ST"][7])
print(f"\n1 - ST_seed (share of variance in the seed-averaged response) = {1-pt['ST'][7]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]   (ST_seed = {pt['ST'][7]:.4f})")
print(f"sum of S1 = {pt['S1'].sum():.4f}; sum of ST = {pt['ST'].sum():.4f}")
