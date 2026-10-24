"""Independent cross-check of the v5 (Saltelli) indices without the complete-block requirement.

Response y = log(cross_track_rmse) over the VALID primary-metric runs of results/campaign_saltelli_merged/analysis_cruise.csv.
  (2) GIVEN-DATA first-order S1_i = Var(E[Y | X_i]) / Var(Y): equal-count binning of X_i (K bins) on ALL valid runs;
      non-negative by construction; bootstrap over runs.
  (3) PAIRWISE-COMPLETE Jansen total effect S_Ti = (1/(2 n_i)) sum_b (y_A,b - y_ABi,b)^2 / Var(Y), over every base block b whose A row
      and AB_i row are both valid; Var(Y) = sample variance (ddof=1) of the valid A and B rows pooled; bootstrap over base blocks.
Block layout (SALib, second order): rows [A, AB_1..AB_D, BA_1..BA_D, B] per base sample (18 rows for D=8).
usage: python3 scripts/saltelli/saltelli_crosscheck_given_data.py   (run on a fleet host)"""
import csv
import json
import math
import os

import numpy as np

EXCL = "TDR_V5_CSV" in os.environ  # cleaned set (analysis_cruise_clean.csv: Dryden-plugin-failure flights removed)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
D = 8
B = 2 * D + 2
SEED_MAX = 2 ** 31 - 2
prob = json.load(open("generated/designs/saltelli/saltelli_problem.json"))["problem"]
names = prob["names"]
design = list(csv.DictReader(open("generated/designs/saltelli/saltelli_all.csv", encoding="utf-8")))
nrun = len(design)
idx = np.array([int(r["saltelli_index"]) for r in design])
assert sorted(idx) == list(range(nrun))
X = np.zeros((nrun, D))
for r, i in zip(design, idx):
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
valid = np.isfinite(y)
print(f"design runs {nrun}; valid primary-metric runs used {int(valid.sum())} ({100*valid.mean():.1f}%)")

# ---- verify block structure on the design values: AB_i = A with column i taken from B
Xb = X.reshape(-1, B, D)
nbl = Xb.shape[0]
Xb_all = Xb
yb_all = y.reshape(nbl, B)
ok = 0
for b in range(min(nbl, 200)):
    A, Bm = Xb[b, 0], Xb[b, B - 1]
    for i in range(D):
        AB = Xb[b, 1 + i]
        expect = A.copy(); expect[i] = Bm[i]
        ok += int(np.allclose(AB, expect, rtol=1e-6, atol=1e-9))
print(f"structure check (first 200 blocks): AB_i == A with column i from B in {ok}/{min(nbl,200)*D} cases")

# ---- (2) given-data S1 by equal-count binning
yv = y[valid]
Xv = X[valid]
N = len(yv)
rng = np.random.default_rng(20261012)


def s1_binned(xcol, ycol, K):
    order = np.argsort(xcol, kind="stable")
    yo = ycol[order]
    n = len(yo)
    edges = np.linspace(0, n, K + 1).astype(int)
    means = np.array([yo[edges[k]:edges[k + 1]].mean() for k in range(K)])
    w = np.diff(edges) / n
    gm = ycol.mean()
    return float((w * (means - gm) ** 2).sum() / ycol.var())


def s1_bias_corrected(xcol, ycol, K):
    order = np.argsort(xcol, kind="stable")
    yo = ycol[order]
    n = len(yo)
    edges = np.linspace(0, n, K + 1).astype(int)
    means = np.array([yo[edges[k]:edges[k + 1]].mean() for k in range(K)])
    within = np.array([yo[edges[k]:edges[k + 1]].var(ddof=1) / (edges[k + 1] - edges[k]) for k in range(K)])
    w = np.diff(edges) / n
    gm = ycol.mean()
    return float((w * ((means - gm) ** 2 - within)).sum() / ycol.var())


KMAIN = 50
nboot = 1000  # resamples for BOTH the S1 (base-block bootstrap) and S_Ti (base-block bootstrap) intervals
print(f"\n(2) GIVEN-DATA S1: equal-count binning K={KMAIN} on all {N} valid runs; Var(Y) = population variance of valid log-metric = {yv.var():.5f}")
res2 = {}
for i, nm in enumerate(names):
    s = s1_binned(Xv[:, i], yv, KMAIN)
    s20, s100 = s1_binned(Xv[:, i], yv, 20), s1_binned(Xv[:, i], yv, 100)
    sbc = s1_bias_corrected(Xv[:, i], yv, KMAIN)
    # BASE-BLOCK bootstrap: rows within a base block share coordinates (AB_i = A except column i), so the resampling unit is
    # the base block (all of its valid runs), not the individual run.
    boots = np.empty(nboot); boots_bc = np.empty(nboot); boots_run = np.empty(nboot)
    for b in range(nboot):
        jb = rng.integers(0, nbl, nbl)
        yy = yb_all[jb].reshape(-1); xx = Xb_all[jb][:, :, i].reshape(-1)
        mk = np.isfinite(yy)
        boots[b] = s1_binned(xx[mk], yy[mk], KMAIN)
        boots_bc[b] = s1_bias_corrected(xx[mk], yy[mk], KMAIN)
        j = rng.integers(0, N, N)  # run bootstrap kept only for comparison
        boots_run[b] = s1_bias_corrected(Xv[j, i], yv[j], KMAIN)
    # Resampling base blocks with replacement duplicates rows, which the binned estimator reads as extra between-bin
    # variance: the bootstrap replicates sit ~0.003 above the point estimate for every factor. Bias-adjusted percentile
    # interval: shift the percentiles by the bootstrap bias (mean replicate - point estimate).
    sh, sh_bc = boots.mean() - s, boots_bc.mean() - sbc
    res2[nm] = (s, np.percentile(boots, 2.5) - sh, np.percentile(boots, 97.5) - sh, s20, s100, sbc,
                float(np.percentile(boots_bc, 2.5) - sh_bc), float(np.percentile(boots_bc, 97.5) - sh_bc),
                float(np.percentile(boots_run, 2.5)), float(np.percentile(boots_run, 97.5)))
    print(f"   {nm:<20} S1={s:.4f}  95% CI (block bootstrap, plug-in, bias-adjusted by {-sh:+.4f}) [{res2[nm][1]:.4f}, {res2[nm][2]:.4f}]; bias-corrected S1 block-bootstrap CI (adjusted by {-sh_bc:+.4f}) [{res2[nm][6]:.4f}, {res2[nm][7]:.4f}]; run-bootstrap CI [{res2[nm][8]:.4f}, {res2[nm][9]:.4f}]   (K=20: {s20:.4f}, K=100: {s100:.4f}; bias-corrected K=50: {sbc:+.4f})")

# ---- (3) pairwise-complete Jansen total effects
yb = y.reshape(nbl, B)
A = yb[:, 0]; Bv = yb[:, B - 1]; AB = yb[:, 1:1 + D]
vA, vB = np.isfinite(A), np.isfinite(Bv)
print(f"\n(3) PAIRWISE-COMPLETE Jansen S_Ti over {nbl} base blocks; valid A rows {int(vA.sum())}, valid B rows {int(vB.sum())}")
pool = np.concatenate([A[vA], Bv[vB]])
print(f"    Var(Y) = sample variance (ddof=1) of the {len(pool)} valid A and B rows pooled = {pool.var(ddof=1):.5f}")


def jansen(Ai, Bi, ABi, idxs):
    a, bb, ab = Ai[idxs], Bi[idxs], ABi[idxs]
    varY = np.concatenate([a[np.isfinite(a)], bb[np.isfinite(bb)]]).var(ddof=1)
    out = []
    for i in range(D):
        m = np.isfinite(a) & np.isfinite(ab[:, i])
        n = int(m.sum())
        out.append((((a[m] - ab[m, i]) ** 2).sum() / (2.0 * n) / varY, n))
    return out


point = jansen(A, Bv, AB, np.arange(nbl))
bs = np.empty((nboot, D))
for b in range(nboot):
    j = rng.integers(0, nbl, nbl)
    r = jansen(A, Bv, AB, j)
    bs[b] = [x[0] for x in r]
cb = {r["factor"]: (float(r["S1"]), float(r["S1_conf"]), float(r["ST"]), float(r["ST_conf"]))
      for r in csv.DictReader(open("results/campaign_saltelli_final/saltelli_cb_indices_cruise_cross_track_rmse.csv"))}
print(f"    {'factor':<20}{'S_Ti pairwise':>14}{'95% CI (block bootstrap)':>28}{'n pairs':>9}")
for i, nm in enumerate(names):
    st, n = point[i]
    print(f"    {nm:<20}{st:>14.4f}   [{np.percentile(bs[:, i], 2.5):.4f}, {np.percentile(bs[:, i], 97.5):.4f}]{n:>12}")
print("\nFINAL TABLE: factor | S1 given-data (CI) | S_Ti pairwise (CI) | n pairs || reference complete-block (SALib, complete blocks only): S1 +-conf, ST +-conf")
for i, nm in enumerate(names):
    s1, lo1, hi1 = res2[nm][:3]
    st, n = point[i]
    c = cb.get(nm, (float("nan"),) * 4)
    print(f"{nm:<20} | {s1:.3f} [{lo1:.3f},{hi1:.3f}] | {st:.3f} [{np.percentile(bs[:, i], 2.5):.3f},{np.percentile(bs[:, i], 97.5):.3f}] | {n} || "
          f"S1 {c[0]:+.3f}+-{c[1]:.3f}, ST {c[2]:.3f}+-{c[3]:.3f}")

# ---- machine-readable output for the paper figure/table
import csv as _csv
out = "results/campaign_saltelli_merged/saltelli_v5_given_data_indices" + ("_" + os.environ.get("TDR_V5_TAG", "clean") if EXCL else "") + ".csv"
with open(out, "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["factor", "S1_biascorrected_K50", "S1_lo95", "S1_hi95", "S1_plugin_K50", "ST_pairwise_jansen", "ST_lo95", "ST_hi95",
                "n_pairs", "n_valid_runs", "varY_runs", "varY_AB", "S1_lo95_runbootstrap", "S1_hi95_runbootstrap", "boot_resamples_S1_blocks", "boot_resamples_ST_blocks"])
    for i, nm in enumerate(names):
        r = res2[nm]; st, n = point[i]
        w.writerow([nm, f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}", f"{r[0]:.6f}", f"{st:.6f}",
                    f"{np.percentile(bs[:, i], 2.5):.6f}", f"{np.percentile(bs[:, i], 97.5):.6f}", n, N,
                    f"{yv.var():.6f}", f"{pool.var(ddof=1):.6f}", f"{r[8]:.6f}", f"{r[9]:.6f}", nboot, nboot])
print("wrote", out)
