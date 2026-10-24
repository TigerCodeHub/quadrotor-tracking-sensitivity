"""Robust re-fit of the sensitivity model for any campaign (audit #4/#9/#10/#11/#12).

Supersedes the classical-OLS, raw-|b|-ranked, nan->0 fits in the per-campaign scripts.
For each response metric it fits  log(response) ~ design factors + airframe fixed effects
on config-mean rows, and reports:

  * HC3 robust SEs (not homoskedastic) and a 95% normal CI per coefficient;
  * the effect across each factor's INTERQUARTILE range (interpretable, unit-free-ish),
    which is the ranking key -- NOT raw |b| (audit #9);
  * drop-one delta-R^2 and conventional partial R^2 per factor;
  * standard tracking metrics (rms_error, cross_track_rmse) alongside sob2, so the story
    does not depend on the SOB2 norm or its duration normalization (audit #4).

Rows with any non-finite predictor or non-positive response are DROPPED and counted
(audit #12), never zero-filled.

Usage:
  python3 scripts/analysis/refit_robust.py <combined_seg.csv> <label> \
      [--responses sob2 rms_error cross_track_rmse max_error] [--json out.json]
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robust_stats import ols_hc3, importance, finite_mask  # noqa: E402

# log_payload = log(1 + payload fraction) and abs_com_y_frac = |com_y|/wheelbase are the DESIGN
# variables (identical ranges in every airframe), so the pooled IQR equals the within-airframe
# IQR that the airframe-fixed-effect coefficient is identified from (audit pass 3 #1). Raw
# log(mass_kg) / |com_y| in metres mixed in between-airframe nominal-mass / wheelbase spread.
FACTORS = ["turbulence_rms", "mean_wind", "log_payload", "thrust_to_weight",
           "inertia_scale", "abs_com_y_frac"]
DEFAULT_RESPONSES = ["sob2", "rms_error", "cross_track_rmse"]


def num(r, k):
    v = r.get(k)
    if v in (None, "", "nan"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load(path):
    """Dedup by run_id; drop failed attempts and 2x-takeoff runs; keep dominant repeats."""
    allrows = list(csv.DictReader(open(path, encoding="utf-8")))
    counts = collections.Counter(r.get("repeat") for r in allrows if r.get("repeat"))
    reps = None
    if counts:
        top = max(counts.values())
        reps = {k for k, v in counts.items() if v >= 0.99 * top}
    rows, seen = [], set()
    for r in allrows:
        if reps is not None and r.get("repeat") not in reps:
            continue
        if r.get("from_failed_attempt") == "1":
            continue
        if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            continue
        rid = r.get("run_id") or ""
        if rid and rid in seen:
            continue
        if rid:
            seen.add(rid)
        rows.append(r)
    return rows


def factor_matrix(rows):
    f = {
        "turbulence_rms": np.array([num(r, "turbulence_rms_m_s") for r in rows]),
        "mean_wind": np.array([num(r, "mean_wind_m_s") for r in rows]),
        "log_payload": np.array([math.log1p(num(r, "payload_mass_pct") / 100.0) for r in rows]),
        "thrust_to_weight": np.array([num(r, "thrust_to_weight") for r in rows]),
        "inertia_scale": np.array([num(r, "inertia_scale") for r in rows]),
        "abs_com_y_frac": np.array([abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9)
                                    for r in rows]),
    }
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("label")
    ap.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES)
    ap.add_argument("--no-fe", action="store_true",
                    help="omit airframe fixed effects (global fit, for cross-campaign comparison)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load(args.csv)
    afs = sorted({r.get("airframe_id") for r in rows if r.get("airframe_id")})
    fac = factor_matrix(rows)
    af_cols = {a: np.array([1.0 if r.get("airframe_id") == a else 0.0 for r in rows])
               for a in afs[1:]}

    print("=" * 100)
    print(f"ROBUST REFIT -- {args.label}: {len(rows)} config rows, {len(afs)} airframes")
    print("HC3 SEs; ranking by |IQR effect|; airframe fixed effects included")
    print("=" * 100)

    try:
        src_sha = hashlib.sha256(open(args.csv, "rb").read()).hexdigest()
    except OSError:
        src_sha = None
    results = {"label": args.label, "n_rows": len(rows),
               "source_csv": args.csv, "source_sha256": src_sha,
               "fixed_effects": (not args.no_fe), "factors_in_model": FACTORS,
               "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "responses": {}}
    for resp in args.responses:
        yv = np.array([num(r, resp) for r in rows])
        base_mask = yv > 0
        fmask = finite_mask(*[fac[f] for f in FACTORS])
        mask = base_mask & fmask
        dropped = int((~mask).sum())
        if mask.sum() < 1000:
            print(f"\n[{resp}] only {int(mask.sum())} usable rows -- SKIPPED")
            continue
        y = np.log(yv[mask])
        names = ["intercept"] + FACTORS + ([] if args.no_fe else [f"AF[{a}]" for a in afs[1:]])
        cols = [np.ones(int(mask.sum()))]
        cols += [fac[f][mask] for f in FACTORS]
        if not args.no_fe:
            cols += [af_cols[a][mask] for a in afs[1:]]
        X = np.column_stack(cols)
        imp, r2 = importance(X, y, names, FACTORS)
        print(f"\n[{resp}]  n={int(mask.sum())} (dropped {dropped})  R^2={r2:.3f}")
        print(f"  {'factor':<18}{'b':>10}{'HC3 se':>10}{'t':>8}"
              f"{'IQR eff':>10}{'dR2(drop)':>11}{'partR2':>9}")
        for d in imp:
            fit_se = None
            print(f"  {d['factor']:<18}{d['beta']:>10.4f}"
                  f"{abs(d['beta'] / d['t']) if d['t'] else float('nan'):>10.4f}"
                  f"{d['t']:>8.1f}{d['iqr_effect']:>10.4f}{d['drop_one_dr2']:>11.4f}{d['partial_r2']:>9.4f}")
        ranking = " > ".join(d["factor"] for d in imp)
        print(f"  RANK (|IQR effect|): {ranking}")
        results["responses"][resp] = {"n": int(mask.sum()), "dropped": dropped,
                                      "r2": round(r2, 4), "ranking": ranking,
                                      "factors": imp}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as h:
            json.dump(results, h, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
