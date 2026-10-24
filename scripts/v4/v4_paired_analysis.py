"""Paired route-effect contrast for v4 c1 (audit #7).

The headline v4 fit (v4_c1234_analysis.py) is a POOLED cross-campaign OLS: it compares the
v4 route coefficient to the v3-D straight coefficient across two different campaigns, so the
"routes amplify wind ~1.7x" number conflates route geometry with altitude, speed, phase mix
and campaign. This script does the predesigned PAIRED contrast instead:

    delta_i = log SOB2(config i on Dryden route)  -  log SOB2(config i on Dryden straight)

using paired_with_run_id to match each v4 c1 flight to its v3-D `_dw` straight control (same
Sobol draws + wind seed). The within-config difference removes the mass/wind/turb/airframe
confounds, leaving the route effect. delta is then modelled on route archetype + airframe
with HC3 and an airframe cluster-bootstrap CI.

NOTE: v4 bundles route geometry WITH per-archetype altitude/speed, so delta is the TOTAL
effect of flying archetype X vs the straight baseline for a matched config -- not route
curvature in isolation. A within-config contrast (a second route for the same config) is
left as future work.

Usage:
  python3 scripts/v4/v4_paired_analysis.py <v4_combined_cruise.csv> <v3d_combined_cruise.csv>
"""
import collections
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from robust_stats import ols_hc3, cluster_bootstrap_se, finite_mask  # noqa: E402

METRIC = os.environ.get("TDR_METRIC", "sob2")  # response column (sob2 | cross_track_rmse | ...)


def num(r, k):
    v = r.get(k)
    try:
        return float(v) if v not in (None, "", "nan") else float("nan")
    except ValueError:
        return float("nan")


def sob2_map(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rid = r.get("run_id")
        if not rid or rid in out or r.get("from_failed_attempt") == "1":
            continue
        v = num(r, METRIC)
        if math.isfinite(v) and v > 0:
            out[rid] = v
    return out


def route(r):
    p = r.get("plan") or ""
    for a in ("dlv_a", "dlv_b", "dlv_c"):
        if p.startswith(a):
            return a
    return "other"


def main():
    v4_csv, v3d_csv = sys.argv[1], sys.argv[2]
    base = sob2_map(v3d_csv)
    pairs = []
    seen = set()
    for r in csv.DictReader(open(v4_csv, encoding="utf-8")):
        rid = r.get("run_id") or ""
        if not rid.endswith("_v4c1") or rid in seen or r.get("from_failed_attempt") == "1":
            continue
        base_id = r.get("paired_with_run_id") or ""
        v = num(r, METRIC)
        if base_id not in base or not (math.isfinite(v) and v > 0):
            continue
        seen.add(rid)
        pairs.append({"delta": math.log(v) - math.log(base[base_id]), "route": route(r),
                      "airframe": r.get("airframe_id") or "?",
                      "wind": num(r, "mean_wind_m_s"), "turb": num(r, "turbulence_rms_m_s")})
    print("=" * 80)
    print(f"PAIRED v4 c1 route contrast: {len(pairs)} matched pairs (Dryden route - Dryden straight)")
    print("=" * 80)
    if not pairs:
        print("no matched pairs found (check paired_with_run_id / _v4c1 ids)")
        return

    # per-route paired mean delta + airframe cluster-bootstrap CI
    by = collections.defaultdict(list)
    for p in pairs:
        by[p["route"]].append(p)
    print(f"\n{'route':<8}{'n':>8}{'mean dlog':>12}{'exp(mean)':>11}  (within-config route effect)")
    for rt in sorted(by):
        d = np.array([p["delta"] for p in by[rt]])
        print(f"{rt:<8}{len(d):>8}{d.mean():>12.4f}{math.exp(d.mean()):>11.3f}")

    # model: delta ~ route dummies (dlv_a baseline) + airframe FE
    routes = sorted({p["route"] for p in pairs})
    afs = sorted({p["airframe"] for p in pairs})
    base_route = "dlv_a" if "dlv_a" in routes else routes[0]
    y = np.array([p["delta"] for p in pairs])
    names = ["intercept"]
    cols = [np.ones(len(pairs))]
    for rt in routes:
        if rt == base_route:
            continue
        names.append(f"route[{rt}]")
        cols.append(np.array([1.0 if p["route"] == rt else 0.0 for p in pairs]))
    for a in afs[1:]:
        names.append(f"AF[{a}]")
        cols.append(np.array([1.0 if p["airframe"] == a else 0.0 for p in pairs]))
    X = np.column_stack(cols)
    m = finite_mask(y, *[X[:, i] for i in range(X.shape[1])])
    X, y = X[m], y[m]
    clusters = np.array([p["airframe"] for p in pairs])[m]
    fit = ols_hc3(X, y)
    se_boot, _ = cluster_bootstrap_se(X, y, clusters, n_boot=1000, seed=7)
    print(f"\nmodel: delta ~ route + airframe FE  (baseline route={base_route}, n={fit['n']}, R^2={fit['r2']:.3f})")
    print(f"  {'term':<16}{'b':>10}{'HC3 se':>10}{'clusterSE':>11}{'t_HC3':>8}")
    for i, nm in enumerate(names):
        print(f"  {nm:<16}{fit['beta'][i]:>10.4f}{fit['se'][i]:>10.4f}"
              f"{se_boot[i]:>11.4f}{fit['t'][i]:>8.1f}")
    print("\nInterpretation: route[dlv_b]/[dlv_c] coefficients are the added log-SOB2 of")
    print("flying the suburban/multi-stop archetype vs the urban baseline, for matched")
    print("configs. Cluster SE (by airframe, 5 clusters) is a conservative dependence check.")


if __name__ == "__main__":
    main()
