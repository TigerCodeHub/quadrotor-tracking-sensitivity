"""Dimensionless collapse and theory-predicted exponents.

Two questions:

3a. COLLAPSE. The dimensional model needs a free intercept per airframe (gamma spans ~0.19
    log units). If the right dimensionless groups are used instead, do the 5 airframes fall
    on ONE curve -- i.e. do the per-airframe intercepts stop being necessary? That is the
    difference between "we measured 5 quads" and "here is a scaling law". Operationalised as:
    fit WITHOUT airframe dummies and measure how much per-airframe structure is left in the
    residuals (spread of per-airframe mean residual, and the F-test for adding dummies back).

3b. EXPONENTS. Physics makes falsifiable predictions that the linear-in-sigma model hides:
      * Gust force on a bluff body ~ rho*A*sigma_v^2, so if deviation follows the forcing,
        d log(dev) / d log(sigma_v) = 2.
      * That force produces acceleration F/m, so for a FIXED gust,
        d log(dev) / d log(m) = -1.
    Measured values quantify how much the controller rejects. A measured mass exponent of
    -0.087 against a predicted -1 says the loop cancels ~91% of the mass effect -- a real
    mechanistic statement, and a better figure than a coefficient table.

Reference speed: the commanded cruise speed is used, NOT the achieved ground speed
(1000 m / duration). Achieved speed is endogenous -- headwind slows the vehicle -- so
dividing by it would put the treatment on both sides of the equation. V_cmd is estimated
from the calmest runs and held fixed.

Usage: python scripts/extensions/dimensionless_analysis.py <combined_seg.csv> <label>
"""
import os, csv, glob, math, sys, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from robust_stats import ols_hc3  # HC3 SEs for the exponent fits (audit pass-3 #4)

# audit pass-3 #4: response column must be set explicitly -- NO sob2 default.
# Legacy SOB2 result: TDR_METRIC=sob2.  Paper primary 3-D PTE metric: cross_track_rmse.
METRIC = os.environ.get("TDR_METRIC")
if not METRIC:
    raise SystemExit("set TDR_METRIC (e.g. cross_track_rmse); refusing to default to sob2")

RHO = 1.225   # kg/m^3, sea-level air density (the worlds are at 488 m MSL; 1.18 would be
              # closer, but the constant only shifts intercepts, never a collapse or slope)
G = 9.80665
TAU_GUST = 2.0  # s, the wind plugin's time_for_rise -- the only gust timescale it defines


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
        try:
            if float(r.get(METRIC) or 0) <= 0:
                continue
        except Exception:
            continue
        seen.add(rid)
        rows.append(r)
    g = collections.defaultdict(list)
    for r in rows:
        g[r.get("model_name") or r.get("run_id", "")[:-4]].append(r)
    out = []
    for _k, v in g.items():
        rep = dict(v[0])
        rep[METRIC] = math.exp(sum(math.log(float(x[METRIC])) for x in v) / len(v))
        out.append(rep)
    return out


num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    n, k = X.shape
    rss = float((res ** 2).sum())
    se = np.sqrt(np.diag((rss / (n - k)) * np.linalg.pinv(X.T @ X)))
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    return b, se, res, rss, r2, n, k


def pval(t):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


rows = load(sys.argv[1])
LABEL = sys.argv[2]
afs = sorted({r["airframe_id"] for r in rows})

# Commanded cruise speed from the calmest decile: achieved speed there is not yet distorted
calm = sorted(rows, key=lambda r: num(r, "mean_wind_m_s") + num(r, "turbulence_rms_m_s"))
calm = calm[: max(50, len(calm) // 20)]
durs = [num(r, "duration_s") for r in calm if num(r, "duration_s") > 0]
V_CMD = 1000.0 / float(np.median(durs))

print("=" * 100)
print("DIMENSIONLESS ANALYSIS -- %s segment, n=%d configs, response=log(%s)" % (LABEL, len(rows), METRIC))
print("=" * 100)
print("   V_cmd = %.3f m/s (median of calmest 5%%: 1000 m / %.1f s). rho=%.3f, tau_gust=%.1f s"
      % (V_CMD, float(np.median(durs)), RHO, TAU_GUST))

y = np.array([math.log(num(r, METRIC)) for r in rows])


def dim_cols(rs):
    """Dimensional predictors (the current model, minus airframe dummies)."""
    out, nm = [], []

    def add(n, v):
        out.append(np.nan_to_num(np.array(v, float), nan=0.0)); nm.append(n)

    add("mean_wind", [num(r, "mean_wind_m_s") for r in rs])
    add("turb_rms", [num(r, "turbulence_rms_m_s") for r in rs])
    add("log_mass", [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rs])
    add("thrust_to_weight", [num(r, "thrust_to_weight") for r in rs])
    add("inertia_scale", [num(r, "inertia_scale") for r in rs])
    add("abs_com_y", [abs(num(r, "com_y_m")) for r in rs])
    return out, nm


def nondim_cols(rs):
    """Dimensionless groups.

    I   = sigma_v / V_cmd                     turbulence intensity
    W   = mean_wind / V_cmd                   steady-wind ratio
    Gs  = rho * sigma_v^2 * b^2 / (m g)       gust force / weight (b = wheelbase)
    Gw  = rho * U^2 * b^2 / (m g)             steady gust force / weight
    TWm = T/W - 1                             thrust margin available for rejection
    St  = tau_att / tau_gust                  attitude timescale vs gust timescale,
          tau_att = sqrt(I_xx / (T_max * b/2)) -- rotational response time
    DL* = disk_loading * g / (rho * V_cmd^2)  dimensionless disk loading
    """
    out, nm = [], []

    def add(n, v):
        out.append(np.nan_to_num(np.array(v, float), nan=0.0)); nm.append(n)

    I, W, Gs, Gw, TWm, St, DL = [], [], [], [], [], [], []
    for r in rs:
        m = max(num(r, "mass_kg"), 1e-9)
        b = max(num(r, "wheelbase_m"), 1e-9)
        sig = num(r, "turbulence_rms_m_s")
        U = num(r, "mean_wind_m_s")
        ixx = max(num(r, "inertia_xx_kg_m2"), 1e-12)
        tmax = max(num(r, "max_thrust_n"), 1e-9)
        I.append(sig / V_CMD)
        W.append(U / V_CMD)
        Gs.append(RHO * sig * sig * b * b / (m * G))
        Gw.append(RHO * U * U * b * b / (m * G))
        TWm.append(max(num(r, "thrust_to_weight") - 1.0, 1e-6))
        St.append(math.sqrt(ixx / (tmax * b / 2.0)) / TAU_GUST)
        DL.append(num(r, "disk_loading_kg_m2") * G / (RHO * V_CMD * V_CMD))
    add("I=sig/V", I)
    add("W=U/V", W)
    add("Gs=rho.sig2.b2/mg", Gs)
    add("Gw=rho.U2.b2/mg", Gw)
    add("TWmargin", TWm)
    add("St=tau_att/tau_g", St)
    add("DL*", DL)
    add("abs_com_y/b", [abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9) for r in rs])
    return out, nm


print("\n1. COLLAPSE: how much per-airframe structure survives, with NO airframe dummies?")
print("   %-26s %7s %9s %11s %9s" % ("model", "R2", "resid_sd", "per-AF sd", "F(dummies)"))
for label, builder in (("dimensional", dim_cols), ("dimensionless", nondim_cols)):
    cols, nm = builder(rows)
    X = np.column_stack([np.ones(len(rows))] + cols)
    b, se, res, rss, r2, n, k = ols(X, y)
    per = [float(np.mean(res[[i for i, r in enumerate(rows) if r["airframe_id"] == a]])) for a in afs]
    # add dummies back: does it still help?
    cols2 = list(cols) + [np.array([1.0 if r["airframe_id"] == a else 0.0 for r in rows]) for a in afs[1:]]
    X2 = np.column_stack([np.ones(len(rows))] + cols2)
    _b2, _se2, _res2, rss2, r22, n2, k2 = ols(X2, y)
    q = k2 - k
    F = ((rss - rss2) / q) / (rss2 / (n2 - k2))
    print("   %-26s %7.4f %9.4f %11.4f %9.1f" % (label, r2, float(res.std()), float(np.std(per)), F))
    print("        per-airframe mean residual: %s" % ", ".join(
        "%s=%+.3f" % (a, v) for a, v in zip(afs, per)))
    print("        R2 with dummies re-added: %.4f (gain %+.4f)" % (r22, r22 - r2))

print("\n2. LOG-LOG EXPONENTS vs physical prediction")
print("   Fit log(%s) = c + p*log(x) + airframe dummies (HC3 robust SE, 95%% CI), compare p to theory." % METRIC)
tests = [
    ("sigma_v (turb RMS)", "turbulence_rms_m_s", 2.0,
     "gust force ~ rho A sigma^2, deviation follows forcing"),
    ("mean wind U", "mean_wind_m_s", 2.0, "same drag law for the steady component"),
]
for name, col, predicted, why in tests:
    sub = [r for r in rows if num(r, col) > 0.2]   # log needs a floor; 0.2 m/s keeps >90%
    if len(sub) < 500:
        print("   %-22s too few rows (%d)" % (name, len(sub)))
        continue
    ys = np.array([math.log(num(r, METRIC)) for r in sub])
    lx = np.array([math.log(num(r, col)) for r in sub])
    cols = [lx] + [np.array([1.0 if r["airframe_id"] == a else 0.0 for r in sub]) for a in afs[1:]]
    X = np.column_stack([np.ones(len(sub))] + cols)
    fit = ols_hc3(X, ys)
    b, se = fit["beta"], fit["se"]
    t_vs_pred = (b[1] - predicted) / se[1]
    print("   %-22s n=%-6d measured p=%+.4f (HC3 se %.4f, 95%% CI [%+.4f, %+.4f])  predicted %+.1f  -> %.0f%% of theory"
          % (name, len(sub), b[1], se[1], b[1] - 1.96 * se[1], b[1] + 1.96 * se[1], predicted, 100 * b[1] / predicted))
    print("        deviation from theory: t=%+.1f (p=%.3g) -- %s" % (t_vs_pred, pval(t_vs_pred), why))

# mass exponent must be WITHIN airframe, else it is airframe identity
sub = rows
ys = np.array([math.log(num(r, METRIC)) for r in sub])
lm = np.array([math.log(max(num(r, "mass_kg"), 1e-9)) for r in sub])
cols = [lm] + [np.array([1.0 if r["airframe_id"] == a else 0.0 for r in sub]) for a in afs[1:]]
X = np.column_stack([np.ones(len(sub))] + cols)
fit = ols_hc3(X, ys)
b, se = fit["beta"], fit["se"]
t_vs_pred = (b[1] - (-1.0)) / se[1]
print("   %-22s n=%-6d measured p=%+.4f (HC3 se %.4f, 95%% CI [%+.4f, %+.4f])  predicted %+.1f  -> %.1f%% of theory"
      % ("mass (within AF)", len(sub), b[1], se[1], b[1] - 1.96 * se[1], b[1] + 1.96 * se[1], -1.0, 100 * b[1] / -1.0))
print("        deviation from theory: t=%+.1f (p=%.3g)" % (t_vs_pred, pval(t_vs_pred)))
print("        => the control loop rejects ~%.0f%% of the open-loop mass effect"
      % (100 * (1 - abs(b[1]) / 1.0)))

print("\n" + "=" * 100)
print("Reading guide: a collapse is credible only if the dimensionless fit leaves LITTLE")
print("per-airframe structure (small per-AF sd, small R2 gain when dummies are re-added).")
print("A large F for the dummies means the groups are still missing airframe physics.")
print("=" * 100)
