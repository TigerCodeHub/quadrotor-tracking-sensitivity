"""v5 given-data 2-D second-order indices, all 28 pairs, from all valid runs (no block structure in the estimator).

C_ij = Var(E[Y|X_i,X_j]) / Var(Y) by equal-count binning (NB bins per factor -> NB*NB cells; bins are equal-count RANK bins from a
stable sort, so tied values (e.g. the 4 cardinal wind directions) are split across bins by row order), with the ANOVA within-cell bias
correction used for S1:  (SS_between - (cells_used - 1) * MS_within) / SS_total,  MS_within = SS_within / (N - cells_used).
S1 is recomputed with the same NB bins (1-D, NB cells) so binning bias is consistent.  Var(Y) population variance of valid runs.
S2_ij = C_ij - S1_i - S1_j.  95% CIs: 1000 base-block bootstrap resamples (bins, S1, C recomputed per resample).
usage: python3 scripts/saltelli/saltelli_second_order_given_data.py > results/campaign_saltelli_merged/second_order_given_data.txt
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
NB = 10
prob = json.load(open("generated/designs/saltelli/saltelli_problem.json"))["problem"]
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
print(f"valid runs {int(np.isfinite(y).sum())} of {nrun}; blocks {nbl}; bins per factor {NB}")

ORDER = [("turbulence", 1), ("mean wind", 2), ("gust seed", 7), ("T/W", 3), ("CoM", 5), ("payload", 0),
         ("wind direction", 6), ("inertia", 4)]
PAIRS = [(a, b) for a in range(D) for b in range(a + 1, D)]


def bins(xcol):
    """equal-count rank bins 0..NB-1 (stable sort)."""
    n = len(xcol)
    o = np.argsort(xcol, kind="stable")
    bi = np.empty(n, np.int64)
    e = np.linspace(0, n, NB + 1).astype(int)
    for k in range(NB):
        bi[o[e[k]:e[k + 1]]] = k
    return bi


def bc_share(cell, ncell, yv, sst, gm):
    cnt = np.bincount(cell, minlength=ncell)
    sm = np.bincount(cell, weights=yv, minlength=ncell)
    sq = np.bincount(cell, weights=yv * yv, minlength=ncell)
    used = cnt > 0
    k = int(used.sum())
    mean = sm[used] / cnt[used]
    ssb = float((cnt[used] * (mean - gm) ** 2).sum())
    ssw = float((sq[used] - cnt[used] * mean ** 2).sum())
    msw = ssw / (len(yv) - k)
    return (ssb - (k - 1) * msw) / sst


def stats(jb):
    yy = yb[jb].reshape(-1); mk = np.isfinite(yy)
    yv = yy[mk]
    Xr = Xb[jb].reshape(-1, D)[mk]
    gm = yv.mean(); sst = float(((yv - gm) ** 2).sum())
    bi = [bins(Xr[:, c]) for c in range(D)]
    S1 = np.array([bc_share(bi[c], NB, yv, sst, gm) for c in range(D)])
    C = {}
    for a, b in PAIRS:
        i, j = ORDER[a][1], ORDER[b][1]
        C[(a, b)] = bc_share(bi[i] * NB + bi[j], NB * NB, yv, sst, gm)
    return {"S1": S1, "C": C, "n": len(yv)}


rng = np.random.default_rng(20261012)
pt = stats(np.arange(nbl))
print(f"N valid = {pt['n']}")
boot = [stats(rng.integers(0, nbl, nbl)) for _ in range(NBOOT)]


def ci(getter, point):
    """Normal-approximation CI: point +/- 1.96 * bootstrap SD.  The percentile interval is NOT used: block resampling duplicates rows,
    which shrinks within-cell variance and shifts the bias-corrected statistic upward (~+0.001-0.002 per index), so percentile CIs
    do not cover the point estimate.  The bootstrap SD is unaffected by that shift."""
    v = np.array([getter(s) for s in boot])
    sd = float(v.std(ddof=1))
    return point - 1.96 * sd, point + 1.96 * sd


print(f"\n{NB}-bin bias-corrected S1 (95% normal CI from SD of {NBOOT} block resamples):")
for a, (nm, c) in enumerate(ORDER):
    lo, hi = ci(lambda s, c=c: s["S1"][c], pt["S1"][c])
    print(f"   {nm:<16}{pt['S1'][c]:+.4f}   [{lo:+.4f},{hi:+.4f}]")
print(f"\nS2_ij = C_ij - S1_i - S1_j, {NB}x{NB} bins")
print(f"{'pair':<30}{'C_ij':>9}{'S2_ij':>9}{'95% CI':>20}")
tot = 0.0
for a, b in PAIRS:
    i, j = ORDER[a][1], ORDER[b][1]
    s2 = pt["C"][(a, b)] - pt["S1"][i] - pt["S1"][j]
    tot += s2
    lo, hi = ci(lambda s, a=a, b=b, i=i, j=j: s["C"][(a, b)] - s["S1"][i] - s["S1"][j], s2)
    print(f"{ORDER[a][0] + ' x ' + ORDER[b][0]:<30}{pt['C'][(a, b)]:>9.4f}{s2:>+9.4f}   [{lo:+.4f},{hi:+.4f}]")
sa = pt["S1"].sum()
lo, hi = ci(lambda s: s["S1"].sum() + sum(s["C"][(a, b)] - s["S1"][ORDER[a][1]] - s["S1"][ORDER[b][1]] for a, b in PAIRS), sa + tot)
print(f"\nsum S1 = {sa:.4f};  sum S2 (28 pairs) = {tot:+.4f};  sum S1 + sum S2 = {sa + tot:.4f}  95% CI [{lo:.4f},{hi:.4f}]")
