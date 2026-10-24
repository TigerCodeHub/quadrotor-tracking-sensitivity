"""Does wind DIRECTION matter -- as a level shift, and as a change in wind sensitivity?

Every fit so far pooled the four cardinal directions, so the published wind and
turbulence coefficients are direction-averaged. Direction is a balanced 4-level
categorical drawn from its own Sobol dimension, so pooling biases nothing -- but it
hides head/tail/cross structure and inflates residual variance.

Geometry (verified from the world SDF and straight.plan -- do not re-derive): the world
frame is ENU, the wind vector is written as (V cos t, V sin t, 0) = the direction the air
moves TOWARD (not the meteorological "from"), and the mission cruises DUE EAST. Hence:

    0 deg   = TAILWIND          180 deg = HEADWIND
    90 deg  = CROSSWIND (left/north)    270 deg = CROSSWIND (right/south)

Four questions, in order of what a reviewer would ask:
  1. Level:       does direction shift baseline error? (3 dummies, base = tailwind)
  2. Sensitivity: does direction change the wind/turbulence SLOPES? (interactions)
  3. Stratified:  fit each direction separately -- the assumption-free version of 2.
  4. Mechanism:   |com_y| is a BODY-frame lateral offset and the aircraft flies due
                  east, so the 90/270 crosswinds push along that same body axis.
                  com_y x crosswind therefore has a real mechanism, unlike the
                  withdrawn wind x log(mass) term that only proxied airframe identity.

Usage: python scripts/v2/wind_direction_analysis.py <combined_seg.csv> <label>
"""
import os, csv, glob, math, sys, collections
import numpy as np

# audit pass-2 #1: response column must be set explicitly -- NO sob2 default,
# so a forgotten env var fails loudly instead of silently reverting to SOB2.
METRIC = os.environ.get("TDR_METRIC")
if not METRIC:
    raise SystemExit("set TDR_METRIC (e.g. cross_track_rmse); refusing to default to sob2")

try:
    from scipy.stats import f as _fdist
except Exception:
    _fdist = None

DIRNAME = {"0.0": "tailwind(0)", "90.0": "cross-left(90)",
           "180.0": "headwind(180)", "270.0": "cross-right(270)"}
CROSS = {"90.0", "270.0"}


def _norm_dir(v):
    """Canonicalise wind_direction_deg to the string keys above ('0' -> '0.0')."""
    try:
        return "%.1f" % (float(v) % 360.0)
    except Exception:
        return "?"


def _collapse_to_configs(rs):
    """One row per config: geometric mean of SOB2 across replicates.

    Same rule as fe_analysis.py -- design covariates are constant within a config
    (replicates differ only by wind seed), so run-level rows are not independent and
    run-level SEs come out sqrt(1+(m-1)*ICC) too small. Equal weights on purpose.
    """
    g = collections.defaultdict(list)
    for r in rs:
        g[r.get("model_name") or r.get("run_id", "")[:-4]].append(r)
    out = []
    for _k, v in g.items():
        rep = dict(v[0])
        rep[METRIC] = math.exp(sum(math.log(float(r[METRIC])) for r in v) / len(v))
        rep["_nrep"] = len(v)
        out.append(rep)
    return out


def _pick_repeats(all_rows, min_frac=0.99):
    env = os.environ.get("TDR_REPEATS", "").strip()
    counts = collections.Counter(r.get("repeat") for r in all_rows if r.get("repeat"))
    if env:
        sel = tuple(x.strip() for x in env.split(",") if x.strip())
        src = "TDR_REPEATS"
    elif counts:
        top = max(counts.values())
        sel = tuple(sorted((k for k, v in counts.items() if v >= min_frac * top),
                           key=lambda z: int(z) if z.isdigit() else 999))
        src = "auto (>=%.0f%% of %d rows)" % (100 * min_frac, top)
    else:
        sel, src = (), "empty"
    print("   replicates used: %s   [%s]" % (",".join(sel) or "none", src))
    excl = {k: v for k, v in sorted(counts.items()) if k not in sel}
    if excl:
        print("   excluded (incomplete): %s" % excl)
    return sel


def pvalue_t(t):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    n, k = X.shape
    rss = float((res ** 2).sum())
    s2 = rss / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    return b, se, rss, r2, n, k


def report(names, b, se, r2, title):
    print("\n%s  R2=%.4f" % (title, r2))
    for nm, bb, ss in zip(names, b, se):
        t = bb / ss if ss else 0.0
        p = pvalue_t(t)
        star = "***" if p < 1e-3 else ("**" if p < 0.01 else ("*" if p < 0.05 else "   "))
        print("     %-26s %+.5f  se=%.5f  t=%+8.2f  p=%-11.3g %s" % (nm, bb, ss, t, p, star))


def ftest(rss_r, k_r, rss_f, k_f, n, label):
    q = k_f - k_r
    if q <= 0 or rss_f <= 0:
        return
    F = ((rss_r - rss_f) / q) / (rss_f / (n - k_f))
    p = _fdist.sf(F, q, n - k_f) if _fdist is not None else float("nan")
    print("   JOINT F-TEST %s: F(%d,%d)=%.2f  p=%.3g%s" % (
        label, q, n - k_f, F, p, "" if _fdist else "  (scipy missing -> p unavailable)"))


# ------------------------------------------------------------------ load + hygiene
_all = []
for f in glob.glob(sys.argv[1]):
    try:
        _all.extend(csv.DictReader(open(f, encoding="utf-8")))
    except Exception:
        continue
REPEATS = _pick_repeats(_all)

rows, seen = [], set()
for r in _all:
    if r.get("repeat") not in REPEATS:
        continue
    if r.get("from_failed_attempt") == "1":
        continue
    if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
        continue
    rid = r.get("run_id") or ""
    if rid in seen:
        continue
    try:
        s = float(r.get(METRIC) or 0)
    except Exception:
        continue
    if s <= 0:
        continue
    seen.add(rid)
    rows.append(r)

LABEL = sys.argv[2]
_nruns = len(rows)
rows = _collapse_to_configs(rows)
print("   collapsed %d runs -> %d configs (config-level SEs; run-level would be ~%.2fx too small)"
      % (_nruns, len(rows), math.sqrt(_nruns / max(len(rows), 1))))

for r in rows:
    r["_dir"] = _norm_dir(r.get("wind_direction_deg"))
rows = [r for r in rows if r["_dir"] in DIRNAME]

num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
y = np.array([math.log(num(r, METRIC)) for r in rows])
afs = sorted({r["airframe_id"] for r in rows})
base_af = afs[0]
DIRS = ["0.0", "90.0", "180.0", "270.0"]
present = [d for d in DIRS if any(r["_dir"] == d for r in rows)]

print("=" * 78)
print("WIND DIRECTION ANALYSIS -- %s segment, n=%d configs" % (LABEL, len(rows)))
print("=" * 78)
print("\n0. BALANCE (direction is its own Sobol dimension, so this should be ~even)")
cnt = collections.Counter(r["_dir"] for r in rows)
for d in present:
    sub = [r for r in rows if r["_dir"] == d]
    gm = math.exp(sum(math.log(num(r, METRIC)) for r in sub) / len(sub))
    print("   %-18s n=%-7d geometric-mean SOB2=%.5f  mean log=%+.4f" % (
        DIRNAME[d], cnt[d], gm, float(np.mean([math.log(num(r, METRIC)) for r in sub]))))


def base_cols(rs):
    """Main-effects design shared by every model here (matches fe_analysis.py)."""
    cols, names = [], []

    def add(nm, v):
        cols.append(np.nan_to_num(np.array(v, float), nan=0.0))
        names.append(nm)

    add("mean_wind", [num(r, "mean_wind_m_s") for r in rs])
    add("turb_rms", [num(r, "turbulence_rms_m_s") for r in rs])
    add("log_mass(within)", [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rs])
    add("thrust_to_weight", [num(r, "thrust_to_weight") for r in rs])
    add("inertia_scale", [num(r, "inertia_scale") for r in rs])
    add("abs_com_y", [abs(num(r, "com_y_m")) for r in rs])
    for af in afs[1:]:
        add("AF[%s]" % af, [1.0 if r["airframe_id"] == af else 0.0 for r in rs])
    return cols, names


# ------------------------------------------------- 1. pooled (reference) + level shifts
cols, names = base_cols(rows)
X0 = np.column_stack([np.ones(len(rows))] + cols)
n0 = ["intercept"] + names
b0, se0, rss0, r20, n, k0 = ols(X0, y)
report(n0, b0, se0, r20, "1. POOLED MAIN EFFECTS (reference -- direction not in model)")

lvl = list(cols)
lvl_n = list(names)
for d in present[1:]:
    lvl.append(np.array([1.0 if r["_dir"] == d else 0.0 for r in rows]))
    lvl_n.append("DIR[%s]" % DIRNAME[d])
X1 = np.column_stack([np.ones(len(rows))] + lvl)
b1, se1, rss1, r21, _, k1 = ols(X1, y)
report(["intercept"] + lvl_n, b1, se1, r21,
       "2. + DIRECTION LEVEL SHIFTS (base = %s)" % DIRNAME[present[0]])
ftest(rss0, k0, rss1, k1, n, "direction level shifts")

# ------------------------------------------------- 3. direction-specific wind slopes
inter = list(lvl)
inter_n = list(lvl_n)
for d in present[1:]:
    ind = np.array([1.0 if r["_dir"] == d else 0.0 for r in rows])
    inter.append(ind * np.array([num(r, "mean_wind_m_s") for r in rows]))
    inter_n.append("wind x %s" % DIRNAME[d])
for d in present[1:]:
    ind = np.array([1.0 if r["_dir"] == d else 0.0 for r in rows])
    inter.append(ind * np.array([num(r, "turbulence_rms_m_s") for r in rows]))
    inter_n.append("turb x %s" % DIRNAME[d])
X2 = np.column_stack([np.ones(len(rows))] + inter)
b2, se2, rss2, r22, _, k2 = ols(X2, y)
report(["intercept"] + inter_n, b2, se2, r22,
       "3. + DIRECTION-SPECIFIC WIND/TURB SLOPES (base slope = %s)" % DIRNAME[present[0]])
ftest(rss1, k1, rss2, k2, n, "direction-specific slopes (over level shifts)")
ftest(rss0, k0, rss2, k2, n, "ALL direction terms (over pooled model)")

# ------------------------------------------------- 4. stratified: one fit per direction
print("\n4. STRATIFIED -- an independent fit per direction (assumption-free version of 3)")
print("   %-18s %-7s %-24s %-24s %s" % ("direction", "n", "mean_wind", "turb_rms", "R2"))
for d in present:
    sub = [r for r in rows if r["_dir"] == d]
    if len(sub) < 50:
        print("   %-18s n=%-6d too few configs" % (DIRNAME[d], len(sub)))
        continue
    ysub = np.array([math.log(num(r, METRIC)) for r in sub])
    csub, nsub = base_cols(sub)
    Xs = np.column_stack([np.ones(len(sub))] + csub)
    bs, ses, _rss, r2s, _n, _k = ols(Xs, ysub)
    idx = {nm: i for i, nm in enumerate(["intercept"] + nsub)}
    iw, it = idx["mean_wind"], idx["turb_rms"]
    print("   %-18s %-7d %+.5f (se %.5f)     %+.5f (se %.5f)     %.4f" % (
        DIRNAME[d], len(sub), bs[iw], ses[iw], bs[it], ses[it], r2s))

# ------------------------------------------------- 5. mechanism: com_y x crosswind
print("\n5. MECHANISM TEST -- |com_y| x crosswind")
print("   com_y is a BODY-frame lateral offset; flying due east means the 90/270")
print("   crosswinds push along that same body axis, so this interaction has a")
print("   physical mechanism (unlike the withdrawn wind x log(mass) term).")
cx = np.array([1.0 if r["_dir"] in CROSS else 0.0 for r in rows])
acom = np.array([abs(num(r, "com_y_m")) for r in rows])
wind = np.array([num(r, "mean_wind_m_s") for r in rows])
mcols = list(cols) + [cx, cx * acom, cx * wind * acom]
mnames = list(names) + ["crosswind", "crosswind x |com_y|", "crosswind x wind x |com_y|"]
X3 = np.column_stack([np.ones(len(rows))] + [np.nan_to_num(c, nan=0.0) for c in mcols])
b3, se3, rss3, r23, _, k3 = ols(X3, y)
report(["intercept"] + mnames, b3, se3, r23, "   model")
ftest(rss0, k0, rss3, k3, n, "crosswind x com_y block")

print("\n" + "=" * 78)
print("Reading guide: a coefficient b on log(SOB2) multiplies path-deviation error by")
print("e^b. Direction terms are deviations FROM %s." % DIRNAME[present[0]])
print("=" * 78)
