"""v3-B analysis: paired treatment-vs-control comparison against the v2 campaign.

Every v3 row is a v2 config flown with ONE field changed (same model, same wind_seed),
so each v3 run pairs exactly with its v2 control via paired_with_run_id and the response
of interest is the within-pair delta  d = log(sob2_v3) - log(sob2_v2),  which removes
config effects entirely.

Two blocks, two questions:
  b1 (plan=straight_hXXX, 7 headings): does the v2 "crosswind is worst" finding survive
     when course heading varies independently of wind direction? In v2 every flight went
     due east, so wind-direction dummies were confounded with course-relative geometry.
     Here relative angle = (wind_direction_deg - course_heading_deg) mod 360 separates
     from absolute direction. If the effect is physical (airframe vs relative wind),
     RELATIVE classes explain it and ABSOLUTE classes add nothing on top.
  b2 (plan=turn_lXXX / climb_sX): do turns and climbs wake the v2 nulls (inertia_scale,
     thrust_to_weight)? The paired delta regressions test whether maneuver-added error
     trends with those design variables.

Usage: v3b_analysis.py <combined_v3_seg.csv> <combined_v2_seg.csv> <label>
"""
import collections
import csv
import math
import sys

import numpy as np

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from robust_stats import ols_hc3_rank, wald_block  # noqa: E402
# audit pass-2 #1: response column must be set explicitly -- NO sob2 default,
# so a forgotten env var fails loudly instead of silently reverting to SOB2.
METRIC = os.environ.get("TDR_METRIC")
if not METRIC:
    raise SystemExit("set TDR_METRIC (e.g. cross_track_rmse); refusing to default to sob2")

TURN_PLANS = ("turn_l100", "turn_l250", "turn_l400")
CLIMB_PLANS = ("climb_s1", "climb_s2")


def num(r, k):
    try:
        v = float(r.get(k))
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    """Same eligibility rules as fe_analysis.py: canonical rows, 1x takeoff, sob2>0."""
    rows, seen = [], set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("from_failed_attempt") == "1":
            continue
        if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            continue
        rid = r.get("run_id") or ""
        if not rid or rid in seen:
            continue
        s = num(r, METRIC)
        if not (s > 0):
            continue
        seen.add(rid)
        rows.append(r)
    return rows


def rel_class(wind_deg, heading_deg):
    """Wind direction RELATIVE to course, folded to the v2 categories.

    Wind vector is 'toward' in ENU; heading is the course direction. rel=0 means the air
    moves the same way the vehicle flies (tailwind), 180 headwind, 90/270 crosswind.
    v3 combinations are multiples of 45; the odd multiples are oblique classes v2 never
    sampled, kept separate rather than pooled into their neighbours.
    """
    rel = (wind_deg - heading_deg) % 360.0
    lab = {0: "tail", 45: "tail-cross", 90: "cross", 135: "head-cross", 180: "head",
           225: "head-cross", 270: "cross", 315: "tail-cross"}
    key = int(round(rel / 45.0)) % 8 * 45
    return lab.get(key, "oblique")


def ols(y, X, names, quiet=False):
    """Rank-aware HC3 OLS (audit pass 3 #7/#11). Aliased columns are dropped (earlier columns
    win) and listed; matrix rank and condition number are printed. Returns (beta, r2);
    the full fit dict is stored in ols.last."""
    f = ols_hc3_rank(X, y, names)
    ols.last = f
    if quiet:
        return f["beta"], f["r2"]
    print("   [rank %d/%d, cond(X)=%.1f%s]" % (f["rank_full"], f["k_full"], f["cond"],
          "; ALIASED dropped: %s" % f["aliased"] if f["aliased"] else ""))
    for nm, b, s, t in zip(f["names"], f["beta"], f["se"], f["t"]):
        star = " ***" if abs(t) > 3.29 else (" **" if abs(t) > 2.58 else (" *" if abs(t) > 1.96 else ""))
        print("   %-22s %+9.4f  (HC3 se %.4f, t %+6.2f)%s" % (nm, b, s, t, star))
    print("   n=%d  R2=%.4f  resid sd=%.4f" % (f["n"], f["r2"], float(np.sqrt(f["resid"] @ f["resid"] / max(f["n"] - len(f["names"]), 1)))))
    return f["beta"], f["r2"]


def nested(y, base_cols, base_names, add_cols, add_names, label):
    """Compare [base] vs [base + add] with the same rank-aware code: delta R2, classical
    nested F (reference), and the ROBUST (HC3) Wald omnibus test on the added block."""
    Xb = np.column_stack(base_cols); fb = ols_hc3_rank(Xb, y, base_names)
    Xf = np.column_stack(base_cols + add_cols); ff = ols_hc3_rank(Xf, y, base_names + add_names)
    chi2, df, p, F = wald_block(ff, add_names)
    dr2 = ff["r2"] - fb["r2"]
    n, kf, kb = ff["n"], len(ff["names"]), len(fb["names"])
    Fc = (dr2 / max(kf - kb, 1)) / ((1 - ff["r2"]) / max(n - kf, 1)) if kf > kb else float("nan")
    print("   %-34s R2 %.4f -> %.4f  dR2=%+.4f | HC3 Wald chi2=%.2f df=%d p=%.3g (F=%.2f) | "
          "classical F=%.2f | rank %d/%d cond=%.1f%s" % (
              label, fb["r2"], ff["r2"], dr2, chi2, df, p, F, Fc, ff["rank_full"], ff["k_full"], ff["cond"],
              " ALIASED=%s" % ff["aliased"] if ff["aliased"] else ""))
    return dr2, p


def covariate_design(rows, extra_cols=()):
    cols, names = [], []

    def add(nm, v):
        cols.append(np.nan_to_num(np.array(v, float), nan=0.0))
        names.append(nm)

    add("mean_wind", [num(r, "mean_wind_m_s") for r in rows])
    add("turb_rms", [num(r, "turbulence_rms_m_s") for r in rows])
    add("log_mass", [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rows])
    add("thrust_to_weight", [num(r, "thrust_to_weight") for r in rows])
    add("inertia_scale", [num(r, "inertia_scale") for r in rows])
    add("abs_com_y", [abs(num(r, "com_y_m")) for r in rows])
    for nm, v in extra_cols:
        add(nm, v)
    return np.column_stack([np.ones(len(rows))] + cols), ["intercept"] + names


def dummies(labels, base):
    levels = sorted(set(labels) - {base})
    return [("[%s]" % lv, [1.0 if l == lv else 0.0 for l in labels]) for lv in levels]


def main():
    v3_path, v2_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    v3 = load(v3_path)
    v2 = {r["run_id"]: r for r in load(v2_path)}
    print("=" * 78)
    print("V3-B PAIRED ANALYSIS -- %s segment" % label)
    print("=" * 78)
    print("v3 eligible rows: %d   v2 control pool: %d" % (len(v3), len(v2)))

    pairs = []
    unmatched = 0
    for r in v3:
        c = v2.get(r.get("paired_with_run_id") or "")
        if c is None:
            unmatched += 1
            continue
        pairs.append((r, c, math.log(num(r, METRIC)) - math.log(num(c, METRIC))))
    print("paired: %d   v3 rows without an eligible v2 control: %d" % (len(pairs), unmatched))
    by_block = collections.Counter(r.get("v3_block") or "?" for r, _c, _d in pairs)
    print("by block: %s" % dict(sorted(by_block.items())))

    # ---------------------------------------------------------------- b1: headings
    b1 = [(r, c, d) for r, c, d in pairs if (r.get("v3_block") or "").startswith("b1")]
    if b1:
        rows = [r for r, _c, _d in b1]
        y = np.array([math.log(num(r, METRIC)) for r in rows])
        relc = [rel_class(num(r, "wind_direction_deg"), num(r, "course_heading_deg")) for r in rows]
        absc = ["abs%d" % int(num(r, "wind_direction_deg")) for r in rows]
        hdgc = ["h%03d" % int(num(r, "course_heading_deg")) for r in rows]

        print("\n" + "-" * 78)
        print("B1: COURSE HEADINGS (n=%d) -- is wind direction course-relative or absolute?" % len(rows))
        print("-" * 78)
        print("relative-class counts: %s" % dict(sorted(collections.Counter(relc).items())))

        print("\n1a. covariates + RELATIVE wind classes (base=tail):")
        X, names = covariate_design(rows, dummies(relc, "tail"))
        _, r2_rel = ols(y, X, names)

        print("\n1b. covariates + ABSOLUTE wind classes (base=abs0):")
        X, names = covariate_design(rows, dummies(absc, "abs0"))
        _, r2_abs = ols(y, X, names)

        print("\n1c. covariates + BOTH (do absolute classes add anything over relative?):")
        X, names = covariate_design(rows, dummies(relc, "tail") + dummies(absc, "abs0"))
        _, r2_both = ols(y, X, names)
        print("\n   R2: relative=%.4f  absolute=%.4f  both=%.4f" % (r2_rel, r2_abs, r2_both))
        print("   -> crosswind is COURSE-RELATIVE if R2(rel) ~= R2(both) >> R2(abs)")

        print("\n1d. OMNIBUS NESTED-MODEL COMPARISON (audit pass 3 #7). Full-rank parameterisation via")
        print("    rank-aware HC3 fit: aliased columns are dropped (base columns kept first) and named.")
        Xc, cn = covariate_design(rows)
        cb = [Xc[:, i] for i in range(Xc.shape[1])]
        rel_d = dummies(relc, "tail"); abs_d = dummies(absc, "abs0"); hdg_d = dummies(hdgc, "h045")
        rc = [np.array(v, float) for _n, v in rel_d]; rn = ["rel" + n for n, _v in rel_d]
        ac = [np.array(v, float) for _n, v in abs_d]; an = ["abs" + n for n, _v in abs_d]
        hc = [np.array(v, float) for _n, v in hdg_d]; hn = ["hdg" + n for n, _v in hdg_d]
        print("    design ranks: relative=%d dummies, absolute=%d, heading=%d (levels: %s)" % (
            len(rc), len(ac), len(hc), sorted(set(hdgc))))
        print("  Which wind description is needed? (covariates always in the base)")
        nested(y, cb, cn, rc, rn, "[cov] -> [cov + RELATIVE]")
        nested(y, cb, cn, ac, an, "[cov] -> [cov + ABSOLUTE]")
        nested(y, cb + rc, cn + rn, ac, an, "[cov+REL] -> [+ ABSOLUTE]")
        nested(y, cb + ac, cn + an, rc, rn, "[cov+ABS] -> [+ RELATIVE]")
        print("  Does HEADING add anything after conditioning on course-relative wind?")
        nested(y, cb + rc, cn + rn, hc, hn, "[cov+REL] -> [+ HEADING]")
        nested(y, cb + rc + ac, cn + rn + an, hc, hn, "[cov+REL+ABS] -> [+ HEADING]")
        print("  Full fit of [cov + REL + HEADING] (rank diagnostics + aliased columns):")
        X, names = covariate_design(rows, rel_d + hdg_d)
        ols(y, X, names)

        print("\n1e. paired deltas by relative-class transition (v2 control -> v3):")
        trans = collections.defaultdict(list)
        for r, c, d in b1:
            c_rel = rel_class(num(c, "wind_direction_deg"), 0.0)
            trans[(c_rel, rel_class(num(r, "wind_direction_deg"), num(r, "course_heading_deg")))].append(d)
        print("   %-24s %6s %9s %9s" % ("control->v3", "n", "mean d", "se"))
        for k in sorted(trans):
            v = np.array(trans[k])
            se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
            print("   %-24s %6d %+9.4f %9.4f" % ("%s->%s" % k, len(v), v.mean(), se))

    # ---------------------------------------------------------------- b2: maneuvers
    b2 = [(r, c, d) for r, c, d in pairs if (r.get("v3_block") or "").startswith("b2")]
    if b2:
        print("\n" + "-" * 78)
        print("B2: TURNS + CLIMBS (n=%d) -- do maneuvers wake inertia / thrust-to-weight?" % len(b2))
        print("-" * 78)
        fams = [("turns", TURN_PLANS), ("climbs", CLIMB_PLANS)]
        for fam, plans in fams:
            sub = [(r, c, d) for r, c, d in b2 if r.get("plan") in plans]
            if not sub:
                print("\n%s: no rows" % fam)
                continue
            rows = [r for r, _c, _d in sub]
            d = np.array([dd for _r, _c, dd in sub])
            print("\n2.%s paired delta ~ design vars (n=%d, plans=%s)" % (fam, len(sub),
                  dict(collections.Counter(r.get("plan") for r in rows))))
            print("   mean delta = %+7.4f (maneuver-added log-error vs straight control)" % d.mean())
            X, names = covariate_design(rows, dummies([r.get("plan") for r in rows],
                                                      sorted({r.get("plan") for r in rows})[0]))
            ols(d, X, names)

        print("\n2c. LEVEL fit on v3-b2 alone (comparable to the v2 main-effects table):")
        rows = [r for r, _c, _d in b2]
        y = np.array([math.log(num(r, METRIC)) for r in rows])
        wdir = ["w%d" % int(num(r, "wind_direction_deg")) for r in rows]
        X, names = covariate_design(rows, dummies([r.get("plan") for r in rows], "turn_l100")
                                    + dummies(wdir, "w0"))
        ols(y, X, names)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main()
