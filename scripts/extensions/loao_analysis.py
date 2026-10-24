"""Leave-one-airframe-out: can the model predict a quad it has never seen?

The reported R2 (0.76-0.91) is IN-SAMPLE and includes a free
per-airframe intercept (gamma_airframe), so it cannot answer the question a practitioner
actually asks: given a new vehicle's parameters and the wind, what tracking error should I
expect? Here each airframe is held out in turn, the model is fit on the other four, and the
held-out airframe is predicted.

Three prediction modes, because they answer different questions:

  RAW        no intercept for the held-out airframe (it gets the base level). Tests whether
             the absolute error LEVEL of an unseen vehicle is predictable. Expected to be
             poor -- gamma spans ~0.19 log units across the 5 airframes.
  RECENTRED  allow ONE free intercept for the held-out airframe, fit on its own data, then
             judge the rest. Tests whether the RESPONSE TO CONDITIONS transfers even if the
             level does not. This is the scientifically interesting question, and it is the
             honest way to report "the shape generalises, the offset does not".
  ORACLE     in-sample fit on the held-out airframe alone. The ceiling: no model trained
             elsewhere can beat this. Included so RAW/RECENTRED are judged against what is
             achievable rather than against zero.

Usage: python scripts/extensions/loao_analysis.py <combined_seg.csv> <label>
"""
import os, csv, glob, math, sys, collections
import numpy as np

# audit pass-2 #1: response column must be set explicitly -- NO sob2 default,
# so a forgotten env var fails loudly instead of silently reverting to SOB2.
METRIC = os.environ.get("TDR_METRIC")
if not METRIC:
    raise SystemExit("set TDR_METRIC (e.g. cross_track_rmse); refusing to default to sob2")

DIRNAME = {"0.0": "tail", "90.0": "crossL", "180.0": "head", "270.0": "crossR"}


def _pick_repeats(all_rows, min_frac=0.99):
    env = os.environ.get("TDR_REPEATS", "").strip()
    counts = collections.Counter(r.get("repeat") for r in all_rows if r.get("repeat"))
    if env:
        sel = tuple(x.strip() for x in env.split(",") if x.strip())
    elif counts:
        top = max(counts.values())
        sel = tuple(sorted((k for k, v in counts.items() if v >= min_frac * top),
                           key=lambda z: int(z) if z.isdigit() else 999))
    else:
        sel = ()
    print("   replicates used: %s" % (",".join(sel) or "none"))
    return sel


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
        if r.get("repeat") not in reps:
            continue
        if r.get("from_failed_attempt") == "1":
            continue
        if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            continue
        rid = r.get("run_id") or ""
        if rid in seen:
            continue
        try:
            if float(r.get(METRIC) or 0) <= 0:
                continue
        except Exception:
            continue
        seen.add(rid)
        rows.append(r)
    # one run per config in v2, but collapse anyway so this works on v1 too
    g = collections.defaultdict(list)
    for r in rows:
        g[r.get("model_name") or r.get("run_id", "")[:-4]].append(r)
    out = []
    for _k, v in g.items():
        rep = dict(v[0])
        rep[METRIC] = math.exp(sum(math.log(float(x[METRIC])) for x in v) / len(v))
        out.append(rep)
    print("   %d runs -> %d configs" % (len(rows), len(out)))
    return out


num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")


def features(rs, with_direction):
    """Design matrix WITHOUT airframe dummies -- that is the point of the exercise."""
    cols, names = [], []

    def add(nm, v):
        cols.append(np.nan_to_num(np.array(v, float), nan=0.0))
        names.append(nm)

    add("mean_wind", [num(r, "mean_wind_m_s") for r in rs])
    add("turb_rms", [num(r, "turbulence_rms_m_s") for r in rs])
    add("log_mass", [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rs])
    add("thrust_to_weight", [num(r, "thrust_to_weight") for r in rs])
    add("inertia_scale", [num(r, "inertia_scale") for r in rs])
    add("abs_com_y", [abs(num(r, "com_y_m")) for r in rs])
    add("log_wheelbase", [math.log(max(num(r, "wheelbase_m"), 1e-9)) for r in rs])
    add("log_disk_loading", [math.log(max(num(r, "disk_loading_kg_m2"), 1e-9)) for r in rs])
    if with_direction:
        d = [("%.1f" % (num(r, "wind_direction_deg") % 360.0)) for r in rs]
        for key in ("90.0", "180.0", "270.0"):
            ind = np.array([1.0 if x == key else 0.0 for x in d])
            add("DIR[%s]" % DIRNAME[key], ind)
            add("wind x %s" % DIRNAME[key], ind * np.array([num(r, "mean_wind_m_s") for r in rs]))
            add("turb x %s" % DIRNAME[key], ind * np.array([num(r, "turbulence_rms_m_s") for r in rs]))
    return np.column_stack([np.ones(len(rs))] + cols), ["intercept"] + names


def fit(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def stats(y, yhat):
    resid = y - yhat
    rmse = float(np.sqrt((resid ** 2).mean()))
    bias = float(resid.mean())
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return rmse, bias, 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


rows = load(sys.argv[1])
LABEL = sys.argv[2]
afs = sorted({r["airframe_id"] for r in rows})
y_all = np.array([math.log(num(r, METRIC)) for r in rows])

print("=" * 96)
print("LEAVE-ONE-AIRFRAME-OUT -- %s segment, n=%d configs, %d airframes" % (LABEL, len(rows), len(afs)))
print("=" * 96)
print("Response: log(SOB2). RMSE is in log units -- 0.10 means a typical ~10%% error in SOB2.")

for with_direction in (False, True):
    tag = "WITH direction terms" if with_direction else "WITHOUT direction terms"
    print("\n" + "-" * 96)
    print("%s" % tag)
    print("-" * 96)
    print("   %-10s %6s | %-22s | %-22s | %-14s" % (
        "held out", "n", "RAW (no intercept)", "RECENTRED (1 offset)", "ORACLE"))
    print("   %-10s %6s | %8s %7s %6s | %8s %7s %6s | %8s %6s" % (
        "", "", "rmse", "bias", "R2", "rmse", "bias", "R2", "rmse", "R2"))
    agg = collections.defaultdict(list)
    for held in afs:
        tr = [r for r in rows if r["airframe_id"] != held]
        te = [r for r in rows if r["airframe_id"] == held]
        Xtr, _ = features(tr, with_direction)
        Xte, _ = features(te, with_direction)
        ytr = np.array([math.log(num(r, METRIC)) for r in tr])
        yte = np.array([math.log(num(r, METRIC)) for r in te])

        b = fit(Xtr, ytr)
        yhat = Xte @ b
        raw = stats(yte, yhat)
        # RECENTRED: single additive offset fitted on the held-out airframe
        rec = stats(yte, yhat + (yte - yhat).mean())
        # ORACLE: fit on the held-out airframe itself
        b_or = fit(Xte, yte)
        orc = stats(yte, Xte @ b_or)

        print("   %-10s %6d | %8.4f %+7.4f %6.3f | %8.4f %+7.4f %6.3f | %8.4f %6.3f" % (
            held, len(te), raw[0], raw[1], raw[2], rec[0], rec[1], rec[2], orc[0], orc[2]))
        agg["raw"].append(raw[0]); agg["rec"].append(rec[0]); agg["orc"].append(orc[0])
        agg["raw_r2"].append(raw[2]); agg["rec_r2"].append(rec[2]); agg["orc_r2"].append(orc[2])
    print("   %-10s %6s | %8.4f %7s %6.3f | %8.4f %7s %6.3f | %8.4f %6.3f" % (
        "MEAN", "", np.mean(agg["raw"]), "", np.mean(agg["raw_r2"]),
        np.mean(agg["rec"]), "", np.mean(agg["rec_r2"]),
        np.mean(agg["orc"]), np.mean(agg["orc_r2"])))

print("\n" + "=" * 96)
print("How to read this:")
print("  RAW R2 near or below 0 => the absolute error level of an unseen airframe is NOT")
print("  predictable (expected: the per-airframe intercepts span ~0.19 log units).")
print("  RECENTRED R2 close to ORACLE R2 => the RESPONSE to wind/turbulence/design DOES")
print("  transfer across airframes, and only the offset needs calibration. That is a much")
print("  more useful and defensible claim than the in-sample R2.")
print("=" * 96)
