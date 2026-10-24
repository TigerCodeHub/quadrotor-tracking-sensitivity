"""Does route amplify the wind slope? (audit pass 3 #3)

  (1) log(PTE) ~ six factors + airframe FE + plan(route x variant) FE + chunk FE   (v4)
  (2) wind x route and turbulence x route interactions; per-route slopes, HC3 CI, omnibus Wald/F
  (3) is route assignment correlated with the Sobol predictors? (eta^2 by route/plan/chunk)
  (4) paired c1 delta log(PTE) (v4 route - matched straight control) ~ wind + turb + route
      + wind x route + airframe FE; per-route penalty slope in wind.
All inference HC3 (robust_stats.ols_hc3).  Metric = TDR_METRIC (default cross_track_rmse).

Usage (c1; pass v3d csv as 3rd arg): TDR_METRIC=cross_track_rmse python3 scripts/v4/route_wind_interaction.py \
        results/campaign_v4/analysis_cruise.csv results/campaign_v3d/analysis_cruise.csv [--json out.json]
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from robust_stats import ols_hc3  # noqa: E402
from refit_robust import load, factor_matrix, FACTORS, num  # noqa: E402

METRIC = os.environ.get("TDR_METRIC", "cross_track_rmse")
OUT = {}
PAIR_SUFFIX = os.environ.get("TDR_PAIR_SUFFIX", "_v4c1")  # c1 -> v3-D _dw controls; "" = all chunks (adds v2 controls)


def hc3_cov(fit, X):
    XtX_inv, u = fit["XtX_inv"], fit["resid"]
    X = np.asarray(X, float)
    h = np.clip(np.einsum("ij,jk,ik->i", X, XtX_inv, X), 0, 1 - 1e-9)
    w = (u / (1 - h)) ** 2
    return XtX_inv @ ((X * w[:, None]).T @ X) @ XtX_inv


def chi2_sf(x, k):
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, k))
    except Exception:
        return float("nan")


def f_sf(x, a, b):
    try:
        from scipy.stats import f
        return float(f.sf(x, a, b))
    except Exception:
        return float("nan")


def wald(beta, cov, idx):
    """H0: all coefficients in idx equal."""
    idx = list(idx)
    q = len(idx) - 1
    R = np.zeros((q, len(beta)))
    for i in range(q):
        R[i, idx[0]] = 1.0
        R[i, idx[i + 1]] = -1.0
    d = R @ beta
    W = float(d @ np.linalg.solve(R @ cov @ R.T, d))
    return W, q


def dummies(labels, drop_first=True):
    lv = sorted(set(labels))
    use = lv[1:] if drop_first else lv
    return use, [np.array([1.0 if l == v else 0.0 for l in labels]) for v in use]


def archetype(p):
    for a in ("dlv_a", "dlv_b", "dlv_c"):
        if (p or "").startswith(a):
            return a
    return "other"


def interaction_block(base_cols, base_names, route_labels, xvars):
    """base + route FE + (x_centered * full route dummy set) for each x: coefficients = per-route slopes."""
    rl, rd = dummies(route_labels, drop_first=True)
    full_l, full_d = dummies(route_labels, drop_first=False)
    cols = list(base_cols) + rd
    names = list(base_names) + [f"route[{l}]" for l in rl]
    slope_idx = {}
    for xn, xv in xvars.items():
        xc = xv - xv.mean()
        slope_idx[xn] = []
        for l, d in zip(full_l, full_d):
            cols.append(xc * d)
            names.append(f"{xn}_c*{l}")
            slope_idx[xn].append(len(cols) - 1)
    return np.column_stack(cols), names, slope_idx, full_l


def report_slopes(tag, fit, X, slope_idx, full_l, nobs):
    cov = hc3_cov(fit, X)
    res = {}
    for xn, ids in slope_idx.items():
        W, q = wald(fit["beta"], cov, ids)
        res[xn] = {"slopes": {}, "wald_chi2": W, "df": q, "p_chi2": chi2_sf(W, q),
                   "F": W / q, "p_F": f_sf(W / q, q, nobs - X.shape[1])}
        print(f"  {xn} x {tag}: per-route slope (HC3 95% CI)")
        for l, i in zip(full_l, ids):
            b, s = fit["beta"][i], math.sqrt(cov[i, i])
            res[xn]["slopes"][l] = [b, s, b - 1.96 * s, b + 1.96 * s]
            print(f"    {l:<10}{b:>9.4f}  se {s:.4f}  [{b-1.96*s:+.4f}, {b+1.96*s:+.4f}]")
        sl = np.array([fit["beta"][i] for i in ids])
        print(f"    range {sl.min():.4f}..{sl.max():.4f}; omnibus equal-slopes Wald chi2({q})={W:.2f} "
              f"p={res[xn]['p_chi2']:.3g}; F={W/q:.2f} p={res[xn]['p_F']:.3g}")
    return res


def eta2(x, labels):
    m = np.isfinite(x)
    x = x[m]
    lab = np.array(labels)[m]
    gm = x.mean()
    sst = ((x - gm) ** 2).sum()
    ssb = sum(((lab == l).sum()) * (x[lab == l].mean() - gm) ** 2 for l in sorted(set(lab)))  # sorted: deterministic sum order
    k, n = len(set(lab)), len(x)
    F = (ssb / (k - 1)) / ((sst - ssb) / (n - k))
    return ssb / sst, F, f_sf(F, k - 1, n - k)


def main():
    v4, v2 = sys.argv[1], sys.argv[2]  # argv[2] = control csv (v3d for c1); extra .csv args pooled as controls
    jp = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    print(f"metric = {METRIC}")

    rows = load(v4)
    fac = factor_matrix(rows)
    y_raw = np.array([num(r, METRIC) for r in rows])
    ok = (y_raw > 0)
    for f in FACTORS:
        ok &= np.isfinite(fac[f])
    rows = [r for r, k in zip(rows, ok) if k]
    fac = {f: fac[f][ok] for f in FACTORS}
    y = np.log(y_raw[ok])
    n = len(y)
    af = [r.get("airframe_id") for r in rows]
    plan = [r.get("plan") for r in rows]
    arch = [archetype(p) for p in plan]
    chunk = [r.get("route_chunk") or "NA" for r in rows]
    print("=" * 90)
    print(f"v4 rows used: {n}; plans={len(set(plan))}, archetypes={len(set(arch))}, chunks={sorted(set(chunk))}")
    one = np.ones(n)
    Fcols = [fac[f] for f in FACTORS]
    al, ad = dummies(af)
    pl, pd_ = dummies(plan)
    cl, cd = dummies(chunk)
    OUT["n_v4"] = n

    def fit_print(name, cols, names):
        X = np.column_stack(cols)
        fit = ols_hc3(X, y)
        i, j = names.index("mean_wind"), names.index("turbulence_rms")
        print(f"  {name:<34} wind {fit['beta'][i]:+.4f} (se {fit['se'][i]:.4f}) "
              f"turb {fit['beta'][j]:+.4f} (se {fit['se'][j]:.4f}) R2={fit['r2']:.3f}")
        return {"wind": [fit["beta"][i], fit["se"][i]], "turb": [fit["beta"][j], fit["se"][j]],
                "r2": fit["r2"]}

    print("\n(1) Wind coefficient under alternative route controls (log-PTE, HC3)")
    fe_names = [f"AF[{a}]" for a in al]
    base_c = [one] + Fcols + ad
    base_n = ["intercept"] + FACTORS + fe_names
    pn = [f"plan[{l}]" for l in pl]
    cn = [f"chunk[{l}]" for l in cl]
    OUT["refit"] = {
        "six+AF": fit_print("six factors + airframe FE", base_c, base_n),
        "+plan FE": fit_print("+ plan FE (9)", base_c + pd_, base_n + pn),
        "+plan+chunk FE": fit_print("+ plan FE + chunk FE", base_c + pd_ + cd, base_n + pn + cn),
    }
    fit = ols_hc3(np.column_stack(base_c + pd_ + cd), y)
    nm = base_n + pn + cn
    print("  full coefficients, plan+chunk FE model:")
    for k, nme in enumerate(nm):
        if nme in FACTORS or nme.startswith("plan") or nme.startswith("chunk"):
            print(f"    {nme:<20}{fit['beta'][k]:>9.4f}  se {fit['se'][k]:.4f}")

    wind, turb = fac["mean_wind"], fac["turbulence_rms"]
    oth = [f for f in FACTORS if f not in ("mean_wind", "turbulence_rms")]
    other_c = [one] + [fac[f] for f in oth] + ad + cd
    other_n = ["intercept"] + oth + fe_names + cn
    OUT["interactions"] = {}
    for tag, labels in (("plan", plan), ("archetype", arch)):
        print("\n" + "=" * 90)
        print(f"(2) wind x {tag} and turbulence x {tag} (centered; + other 4 factors, airframe FE, chunk FE, {tag} FE)")
        X, names, sidx, full_l = interaction_block(other_c, other_n, labels, {"wind": wind, "turb": turb})
        fit = ols_hc3(X, y)
        print(f"  n={n} k={X.shape[1]} R2={fit['r2']:.3f}")
        OUT["interactions"][tag] = report_slopes(tag, fit, X, sidx, full_l, n)
        print(f"  -- wind-only interaction (turbulence entered as a common slope):")
        X1, n1, s1, fl = interaction_block(other_c + [turb], other_n + ["turbulence_rms"], labels, {"wind": wind})
        f1 = ols_hc3(X1, y)
        OUT["interactions"][tag + "_wind_only"] = report_slopes(tag, f1, X1, s1, fl, n)
        Xn = np.column_stack(other_c + [turb, wind] + dummies(labels)[1])
        fn = ols_hc3(Xn, y)
        print(f"  R2 no-interaction {fn['r2']:.4f} -> with wind x {tag} {f1['r2']:.4f}")

    print("\n" + "=" * 90)
    print("(3) Is route assignment correlated with predictors?  eta^2 (share of predictor variance explained by label), F-test p")
    OUT["eta2"] = {}
    pred = dict(fac)
    pred["wind_dir_deg"] = np.array([num(r, "wind_direction_deg") for r in rows])
    for tag, labels in (("plan", plan), ("archetype", arch), ("chunk", chunk), ("airframe", af)):
        print(f"  by {tag}:")
        OUT["eta2"][tag] = {}
        for k, v in pred.items():
            e, F, p = eta2(v, labels)
            OUT["eta2"][tag][k] = [e, F, p]
            print(f"    {k:<18} eta2={e:.5f}  F={F:.2f}  p={p:.3g}")
    print("  plan counts:", {p_: plan.count(p_) for p_ in sorted(set(plan))})

    print("\n" + "=" * 90)
    # controls: c1 pairs to v3-D '_dw' straight runs; other chunks pair to v2 '_n01' straight runs.
    # Both control CSVs are pooled (run_ids are disjoint).
    base = {}
    for cpath in [v2] + [a for a in sys.argv[3:] if a.endswith(".csv")]:
        for r in csv.DictReader(open(cpath, encoding="utf-8")):
            rid = r.get("run_id")
            if not rid or rid in base or r.get("from_failed_attempt") == "1":
                continue
            v = num(r, METRIC)
            if math.isfinite(v) and v > 0:
                base[rid] = v
    pairs, seen = [], set()
    for r in csv.DictReader(open(v4, encoding="utf-8")):
        rid = r.get("run_id") or ""
        if not rid.endswith(PAIR_SUFFIX) or rid in seen or r.get("from_failed_attempt") == "1":
            continue
        b = r.get("paired_with_run_id") or ""
        v = num(r, METRIC)
        w_, t_ = num(r, "mean_wind_m_s"), num(r, "turbulence_rms_m_s")
        if b not in base or not (v > 0) or not (math.isfinite(w_) and math.isfinite(t_)):
            continue
        seen.add(rid)
        pairs.append((math.log(v) - math.log(base[b]), r.get("plan"), r.get("airframe_id"), w_, t_,
                      math.log(v), math.log(base[b])))
    m = len(pairs)
    d = np.array([p[0] for p in pairs])
    pw = np.array([p[3] for p in pairs])
    pt = np.array([p[4] for p in pairs])
    paf = [p[2] for p in pairs]
    print(f"(4) Paired c1: {m} pairs (v4 route minus the matched straight control (v3-D Dryden '_dw' for c1; v2 for other chunks), same config + wind seed)")
    OUT["paired"] = {"n": m}
    print(f"  mean delta={d.mean():.4f}; corr(delta, wind)={np.corrcoef(d, pw)[0,1]:.3f}")
    a_l, a_d = dummies(paf)
    po = np.ones(m)
    for tag, labels in (("archetype", [archetype(p[1]) for p in pairs]), ("plan", [p[1] for p in pairs])):
        print(f"\n  delta ~ wind + turb + {tag} + wind x {tag} + airframe FE   [wind/turb centered]")
        cols = [po, pt - pt.mean()] + a_d
        names = ["intercept", "turb_c"] + [f"AF[{a}]" for a in a_l]
        rl, rd = dummies(labels)
        Xm = np.column_stack(cols + [pw - pw.mean()] + rd)
        fm = ols_hc3(Xm, d)
        iw = len(cols)
        print(f"    no-interaction: wind coef on delta {fm['beta'][iw]:+.4f} (se {fm['se'][iw]:.4f}); R2={fm['r2']:.3f}")
        X, nms, sidx, fl = interaction_block(cols, names, labels, {"wind": pw})
        fit = ols_hc3(X, d)
        print(f"    interaction model n={m} R2={fit['r2']:.3f}")
        OUT["paired"][tag] = {"wind_main_no_int": [fm["beta"][iw], fm["se"][iw]],
                              **report_slopes(tag, fit, X, sidx, fl, m)}
        print("    (slope = d delta/d wind per route = growth of route penalty with wind, log units per m/s)")
    if jp:
        json.dump(OUT, open(jp, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
