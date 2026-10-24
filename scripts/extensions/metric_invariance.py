"""Are the conclusions an artefact of the SOB2 metric?

SOB2 = sqrt(sum|dev|^2 + sum|d dev|^2) / steps is not a metric anyone else reports, so
"the authors invented the metric that produced the result" is a cheap and damaging
criticism. It costs nothing to pre-empt: compute_metrics() already emits standard
alternatives for every run, so the same model can be refit on each and the signs,
significance and RANKING of the design factors compared.

Metrics compared (all already in the combined CSV -- no reprocessing needed):
  sob2             the Sobolev-style norm used throughout (reference)
  rms_error        RMS 3-D deviation from the commanded setpoint -- the field default
  cross_track_rmse RMS distance to the commanded PATH (ignores along-track lag)
  max_error        worst-case deviation -- the safety-relevant one
  cross_track_max  worst-case cross-track deviation
  altitude_rmse    vertical only, a partly independent axis

Note SOB2's two known mechanical sensitivities, which is why cross_track_* matters as a
check: the derivative term depends on the sample interval, and dividing by `steps` couples
the score to flight duration, which itself varies with wind (headwind runs take longer).
cross_track_rmse has neither property.

Usage: python scripts/extensions/metric_invariance.py <combined_seg.csv> <label>
"""
import os, csv, glob, math, sys, collections
import numpy as np

# Robust SEs + a proper (standardized / IQR / partial-R^2) importance ranking live in
# scripts/analysis/refit_robust.py. This script's purpose is the metric-INVARIANCE check
# (sign + significance agreement across error metrics); it now uses HC3 SEs for that, and
# its factor ordering below is by standardized |b| (NOT raw |b|, audit #9) and is only a
# consistency check, not an absolute importance claim.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from robust_stats import ols_hc3  # noqa: E402

METRICS = ["sob2", "rms_error", "cross_track_rmse", "max_error", "cross_track_max", "altitude_rmse"]
FACTORS = ["mean_wind", "turb_rms", "log_mass", "thrust_to_weight", "inertia_scale", "abs_com_y"]
DIRNAME = {"90.0": "crossL", "180.0": "head", "270.0": "crossR"}


def _pick_repeats(all_rows, min_frac=0.99):
    env = os.environ.get("TDR_REPEATS", "").strip()
    counts = collections.Counter(r.get("repeat") for r in all_rows if r.get("repeat"))
    if env:
        return tuple(x.strip() for x in env.split(",") if x.strip())
    if not counts:
        return ()
    top = max(counts.values())
    return tuple(sorted((k for k, v in counts.items() if v >= min_frac * top),
                        key=lambda z: int(z) if z.isdigit() else 999))


def load(pattern):
    allrows = []
    for f in glob.glob(pattern):
        try:
            allrows.extend(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:
            continue
    reps = _pick_repeats(allrows)
    rows, seen = [], set()
    for r in allrows:
        if r.get("repeat") not in reps or r.get("from_failed_attempt") == "1":
            continue
        if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            continue
        rid = r.get("run_id") or ""
        if rid in seen:
            continue
        seen.add(rid)
        rows.append(r)
    return rows


num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")


def ols(X, y):
    # HC3 heteroskedasticity-consistent SEs (audit #10); classical homoskedastic SEs were
    # anticonservative at these n.
    r = ols_hc3(X, y)
    return r["beta"], r["se"], r["r2"]


def pval(t):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


rows = load(sys.argv[1])
LABEL = sys.argv[2]
afs = sorted({r["airframe_id"] for r in rows})

print("=" * 108)
print("METRIC INVARIANCE -- %s segment, %d rows loaded" % (LABEL, len(rows)))
print("=" * 108)

for with_direction in (False, True):
    print("\n" + "-" * 108)
    print("MODEL: main effects + airframe dummies%s" % (" + direction terms" if with_direction else ""))
    print("-" * 108)
    hdr = "   %-17s %8s" % ("metric", "n")
    for f in FACTORS:
        hdr += " %>18s" % f if False else " %18s" % f
    hdr += " %7s" % "R2"
    print(hdr)

    table = {}
    for metric in METRICS:
        sub = []
        for r in rows:
            try:
                v = float(r.get(metric) or 0)
            except Exception:
                continue
            if v > 0:
                sub.append(r)
        if len(sub) < 1000:
            print("   %-17s SKIPPED (only %d usable rows)" % (metric, len(sub)))
            continue
        y = np.array([math.log(float(r[metric])) for r in sub])
        cols, names = [], []

        def add(nm, v):
            cols.append(np.nan_to_num(np.array(v, float), nan=0.0)); names.append(nm)

        add("mean_wind", [num(r, "mean_wind_m_s") for r in sub])
        add("turb_rms", [num(r, "turbulence_rms_m_s") for r in sub])
        add("log_mass", [math.log(max(num(r, "mass_kg"), 1e-9)) for r in sub])
        add("thrust_to_weight", [num(r, "thrust_to_weight") for r in sub])
        add("inertia_scale", [num(r, "inertia_scale") for r in sub])
        add("abs_com_y", [abs(num(r, "com_y_m")) for r in sub])
        if with_direction:
            d = [("%.1f" % (num(r, "wind_direction_deg") % 360.0)) for r in sub]
            for key in ("90.0", "180.0", "270.0"):
                ind = np.array([1.0 if x == key else 0.0 for x in d])
                add("DIR[%s]" % DIRNAME[key], ind)
                add("wind x %s" % DIRNAME[key], ind * np.array([num(r, "mean_wind_m_s") for r in sub]))
                add("turb x %s" % DIRNAME[key], ind * np.array([num(r, "turbulence_rms_m_s") for r in sub]))
        for a in afs[1:]:
            add("AF[%s]" % a, [1.0 if r["airframe_id"] == a else 0.0 for r in sub])
        X = np.column_stack([np.ones(len(sub))] + cols)
        b, se, r2 = ols(X, y)
        idx = {nm: i for i, nm in enumerate(["intercept"] + names)}
        line = "   %-17s %8d" % (metric, len(sub))
        vals = {}
        for f in FACTORS:
            i = idx[f]
            t = b[i] / se[i] if se[i] else 0.0
            p = pval(t)
            star = "***" if p < 1e-3 else ("**" if p < 0.01 else ("*" if p < 0.05 else " ns"))
            line += " %+11.5f%-6s" % (b[i], star)
            vals[f] = (b[i], p, float(np.std(cols[idx[f] - 1])))
        line += " %7.4f" % r2
        print(line)
        table[metric] = vals

    if not table:
        continue
    print("\n   SIGN AND SIGNIFICANCE AGREEMENT vs sob2 (the reference metric):")
    ref = table.get("sob2")
    for metric, vals in table.items():
        if metric == "sob2":
            continue
        same_sign = sum(1 for f in FACTORS if np.sign(vals[f][0]) == np.sign(ref[f][0]))
        same_sig = sum(1 for f in FACTORS if (vals[f][1] < 0.05) == (ref[f][1] < 0.05))
        disagree = [f for f in FACTORS if np.sign(vals[f][0]) != np.sign(ref[f][0])]
        print("   %-17s sign %d/%d, significance %d/%d%s" % (
            metric, same_sign, len(FACTORS), same_sig, len(FACTORS),
            ("   DISAGREES ON: " + ", ".join(disagree)) if disagree else ""))

    print("\n   FACTOR ORDER (by standardized |b| = |b*std(x)|; a cross-metric CONSISTENCY")
    print("   check, NOT an absolute importance ranking -- use scripts/analysis/refit_robust.py")
    print("   for IQR effects + partial R^2):")
    for metric, vals in table.items():
        order = sorted(FACTORS, key=lambda f: -abs(vals[f][0] * vals[f][2]))
        print("   %-17s %s" % (metric, " > ".join(order)))

print("\n" + "=" * 108)
print("A conclusion that only holds for sob2 is a metric artefact. Identical signs and")
print("near-identical significance across metrics means the findings are about the vehicle,")
print("not about the norm. cross_track_* is the sharpest check: unlike sob2 it has no")
print("derivative term and no division by sample count, so it cannot inherit either")
print("sampling-rate or flight-duration sensitivity.")
print("=" * 108)
