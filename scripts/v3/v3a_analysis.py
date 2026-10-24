"""v3-A analysis: does per-config PX4 autotune remove the v2 MASS effect?

Every _at run is a v2 config flown with controller_mode=autotune (attitude/rate loops
retuned in flight) instead of the fixed controller, SAME model_name + wind_seed, straight
plan. It pairs with its v2 control via v3_a.csv paired_with_run_id. The within-pair delta
  d = ln(sob2_autotune) - ln(sob2_fixed)
removes config identity entirely, so d ~ design-var slopes isolate what RETUNING changed.

v2 headline (cruise): log_mass -0.0873 (heavier tracked slightly better). Autotune touches
only the attitude/rate loops, NOT the MPC position loop, so the a-priori expectation is the
mass effect PERSISTS: d should not trend with mass (slope ~ 0), and the autotune mass slope
should match the fixed mass slope.

Inference (audit pass 3 #6, #11): HC3 SEs throughout (robust_stats.ols_hc3_rank). Results
are reported three ways: GLOBAL (raw log mass, the old estimand that mixes within- and
between-airframe mass variation), AIRFRAME-FE (log1p(payload fraction), |com_y|/wheelbase +
airframe fixed effects = the paper's within-airframe estimand), and PER-AIRFRAME payload
slopes. Because d = ln(at) - ln(fixed), the coefficient of any predictor in the d-regression
IS (autotune slope - fixed slope) exactly.

Usage: v3a_analysis.py <combined_v3a_seg.csv> <combined_v2_seg.csv> <v3_a_design.csv> <label>
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from robust_stats import ols_hc3_rank, wald_block  # noqa: E402

# audit pass-2 #1: response column must be set explicitly -- NO sob2 default,
# so a forgotten env var fails loudly instead of silently reverting to SOB2.
METRIC = os.environ.get("TDR_METRIC")
if not METRIC:
    raise SystemExit("set TDR_METRIC (e.g. cross_track_rmse); refusing to default to sob2")

COV_NAMES = ["turb", "wind", "t_to_w", "inertia", "abs_com_y_frac"]


def num(r, k):
    try:
        v = float(r.get(k))
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_sob2(path):
    """Canonical rows only, 1x takeoff, metric>0 -> {run_id: row} (first canonical wins)."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("from_failed_attempt") == "1":
            continue
        if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            continue
        rid = r.get("run_id") or ""
        if not rid or rid in out:
            continue
        s = num(r, METRIC)
        if not (s > 0):
            continue
        out[rid] = r
    return out


def fit(title, y, cols, names, show=None, quiet=False):
    """HC3 rank-aware fit; prints coefficients (all, or only those in `show`)."""
    X = np.column_stack([np.ones(len(y))] + list(cols))
    f = ols_hc3_rank(X, y, ["const"] + list(names))
    if not quiet:
        print(f"\n{title}  (n={f['n']}, R2={f['r2']:.4f}, HC3, rank={f['rank_full']}/{f['k_full']}, "
              f"cond={f['cond']:.1f})")
        if f["aliased"]:
            print(f"   ALIASED/dropped: {f['aliased']}")
        for nm, b, s, tv in zip(f["names"], f["beta"], f["se"], f["t"]):
            if show is None or nm in show:
                print(f"   {nm:<20} {b:+.4f}  se={s:.4f}  t={tv:+.2f}")
    return f


def coef(f, nm):
    i = f["names"].index(nm)
    return f["beta"][i], f["se"][i], f["t"][i]


def gap_report(design, at):
    """Which designed _at configs have no valid scored flight? By airframe / design position."""
    miss, tot = {}, {}
    rows = []
    for rid, r in design.items():
        a = r.get("airframe_id") or "?"
        tot[a] = tot.get(a, 0) + 1
        got = rid in at
        if not got:
            miss[a] = miss.get(a, 0) + 1
        rows.append((num(r, "execution_order"), num(r, "sample_index"), got, r.get("wind_level")))
    print("\n[GAP] designed vs scored by airframe:")
    for a in sorted(tot):
        m = miss.get(a, 0)
        print(f"    {a:<10} designed={tot[a]:6d} scored={tot[a]-m:6d} unscored={m:6d} ({100*m/tot[a]:.1f}%)")
    got = np.array([g for _e, _s, g, _w in rows])
    for label, k in (("execution_order", 0), ("sample_index", 1)):
        v = np.array([x[k] for x in rows])
        ok = np.isfinite(v)
        if not ok.any():
            continue
        qs = np.quantile(v[ok], np.linspace(0, 1, 11))
        print(f"[GAP] scored fraction by {label} decile:")
        for i in range(10):
            hi = (v <= qs[i + 1]) if i == 9 else (v < qs[i + 1])
            m = ok & (v >= qs[i]) & hi
            if m.any():
                print(f"    decile {i}: [{qs[i]:.0f},{qs[i+1]:.0f})  n={int(m.sum()):5d}  "
                      f"scored={100*got[m].mean():5.1f}%")
    wl = {}
    for _e, _s, g, w in rows:
        a = wl.setdefault(w, [0, 0]); a[0] += 1; a[1] += int(g)
    print("[GAP] scored fraction by wind_level: " +
          "  ".join(f"{k}: {100*v[1]/v[0]:.1f}% (n={v[0]})" for k, v in sorted(wl.items(), key=lambda z: str(z[0]))))


def main():
    v3a_path, v2_path, design_path, label = sys.argv[1:5]
    at = load_sob2(v3a_path)
    v2 = load_sob2(v2_path)
    design = {r["run_id"]: r for r in csv.DictReader(open(design_path, encoding="utf-8"))}
    n_design = len(design)

    pairs = []
    n_at_valid = n_no_ctrl = n_no_at = 0
    for rid, drow in design.items():
        atrow = at.get(rid)
        if atrow is None:
            n_no_at += 1
            continue
        n_at_valid += 1
        cid = drow.get("paired_with_run_id") or ""
        ctrl = v2.get(cid)
        if ctrl is None:
            n_no_ctrl += 1
            continue
        s_at = num(atrow, METRIC)
        s_c = num(ctrl, METRIC)
        if not (s_at > 0 and s_c > 0):
            continue
        pairs.append((drow, s_at, s_c))

    print(f"=== v3-A autotune analysis [{label}]  metric={METRIC}  (HC3) ===")
    print(f"designed _at configs      : {n_design}")
    print(f"_at with valid {METRIC}: {n_at_valid}  ({100*n_at_valid/n_design:.1f}% of design; "
          f"{n_no_at} design rows have NO valid scored _at flight)")
    print(f"  of those, no v2 control : {n_no_ctrl}")
    print(f"paired _at<->v2 usable     : {len(pairs)}")
    if len(pairs) < 50:
        print("too few pairs -- stopping"); return

    gap_report(design, at)

    d = np.array([math.log(s_at) - math.log(s_c) for (_, s_at, s_c) in pairs])
    drows = [dr for (dr, _, _) in pairs]
    logm = np.array([math.log(num(dr, "mass_kg")) for dr in drows])
    logp = np.array([math.log1p(num(dr, "payload_mass_pct") / 100.0) for dr in drows])
    turb = np.array([num(dr, "turbulence_rms_m_s") for dr in drows])
    wind = np.array([num(dr, "mean_wind_m_s") for dr in drows])
    tw = np.array([num(dr, "thrust_to_weight") for dr in drows])
    inert = np.array([num(dr, "inertia_scale") for dr in drows])
    comy = np.array([abs(num(dr, "com_y_m")) for dr in drows])
    comf = np.array([abs(num(dr, "com_y_m")) / max(num(dr, "wheelbase_m"), 1e-9) for dr in drows])
    lsob_at = np.array([math.log(s_at) for (_, s_at, _) in pairs])
    lsob_c = np.array([math.log(s_c) for (_, _, s_c) in pairs])
    afid = np.array([dr.get("airframe_id") or dr.get("model_name", "").rsplit("_", 1)[0] for dr in drows])
    afname = {dr.get("airframe_id"): dr.get("airframe_name") for dr in drows}
    afs = sorted(set(afid))
    fe_cols = [(afid == a).astype(float) for a in afs[1:]]
    fe_names = ["AF[%s]" % a for a in afs[1:]]

    print("\n[0] n per airframe (paired, usable):")
    for a in afs:
        print(f"    {a:<10} {afname.get(a, ''):<18} n={int((afid == a).sum())}")
    print(f"    total n={len(d)}")

    fit("[1] mean paired delta d = ln(autotune) - ln(fixed)  (const = mean d; neg = retuning IMPROVED)",
        d, [], [])
    print(f"    median d={np.median(d):+.4f}   frac improved (d<0)={float((d < 0).mean()):.3f}")

    # ============ GLOBAL (old estimand: raw log mass, no airframe FE) ============
    print("\n" + "#" * 72 + "\n# GLOBAL slopes (raw log mass, NO airframe FE): mixes within- and"
          "\n# between-airframe mass variation. Kept for comparison with earlier numbers.\n" + "#" * 72)
    gcov = [turb, wind, tw, inert, comy]
    gnames = ["log_mass", "turb", "wind", "t_to_w", "inertia", "abs_com_y"]
    fit("[G2] GLOBAL paired delta  d ~ log_mass", d, [logm], ["log_mass"])
    fit("[G3] GLOBAL paired delta  d ~ log_mass + covariates", d, [logm] + gcov, gnames)
    ba = fit("[G4a] GLOBAL fixed:    ln(metric_v2) ~ log_mass + covars", lsob_c, [logm] + gcov, gnames)
    bb = fit("[G4b] GLOBAL autotune: ln(metric_at) ~ log_mass + covars", lsob_at, [logm] + gcov, gnames)
    fa, fb = coef(ba, "log_mass"), coef(bb, "log_mass")
    print(f"\n    GLOBAL fixed log_mass slope    = {fa[0]:+.4f} (HC3 t={fa[2]:+.2f})")
    print(f"    GLOBAL autotune log_mass slope = {fb[0]:+.4f} (HC3 t={fb[2]:+.2f})")
    print(f"    GLOBAL change (autotune-fixed) = {fb[0]-fa[0]:+.4f}")

    # ============ AIRFRAME FIXED EFFECTS (within-airframe payload slope) ============
    print("\n" + "#" * 72 + "\n# AIRFRAME-FE: log1p(payload fraction), |com_y|/wheelbase + airframe FE"
          "\n# (the paper's within-airframe estimand)\n" + "#" * 72)
    cov = [turb, wind, tw, inert, comf]
    fcols = [logp] + cov + fe_cols
    fnames = ["log_payload"] + COV_NAMES + fe_names
    shw = {"const", "log_payload"} | set(COV_NAMES)
    fit("[F2] FE: paired delta  d ~ log_payload + covariates + airframe FE", d, fcols, fnames, show=shw)
    fx = fit("[F4a] FE fixed:    ln(metric_v2) ~ log_payload + covars + AF FE", lsob_c, fcols, fnames, show=shw)
    fy = fit("[F4b] FE autotune: ln(metric_at) ~ log_payload + covars + AF FE", lsob_at, fcols, fnames, show=shw)
    xa, xb = coef(fx, "log_payload"), coef(fy, "log_payload")
    cd = coef(fit("", d, fcols, fnames, quiet=True), "log_payload")
    print(f"\n    FE fixed log_payload slope    = {xa[0]:+.4f} (se {xa[1]:.4f}, t={xa[2]:+.2f})")
    print(f"    FE autotune log_payload slope = {xb[0]:+.4f} (se {xb[1]:.4f}, t={xb[2]:+.2f})")
    print(f"    FE autotune-fixed (= paired-delta slope, HC3) = {cd[0]:+.4f} (se {cd[1]:.4f}, t={cd[2]:+.2f})")

    # ============ AIRFRAME x PAYLOAD interaction ============
    print("\n" + "#" * 72 + "\n# AIRFRAME x PAYLOAD interaction (payload slope may differ by airframe)\n" + "#" * 72)
    ia_cols = [(afid == a).astype(float) * logp for a in afs[1:]]
    ia_names = ["AFxpay[%s]" % a for a in afs[1:]]
    icols = [logp] + cov + fe_cols + ia_cols
    inames = ["log_payload"] + COV_NAMES + fe_names + ia_names
    for ttl, yy in (("delta d", d), ("fixed ln(metric_v2)", lsob_c), ("autotune ln(metric_at)", lsob_at)):
        fi = fit("", yy, icols, inames, quiet=True)
        chi2, df, p, _F = wald_block(fi, ia_names)
        print(f"[I] {ttl}: joint HC3 Wald, airframe x payload interactions: chi2={chi2:.2f} df={df} "
              f"p={p:.3g}  (R2={fi['r2']:.4f})")
    print("\n[I2] PER-AIRFRAME payload slope (separate fit per airframe: y ~ log_payload + covars, HC3)")
    print(f"    {'airframe':<10} {'n':>5}  {'fixed':>8} {'(t)':>8}  {'autotune':>9} {'(t)':>8}  "
          f"{'at-fixed':>9} {'se':>7} {'t':>7}")
    for a in afs:
        m = afid == a
        c_ = [turb[m], wind[m], tw[m], inert[m], comf[m]]
        nm_ = ["log_payload"] + COV_NAMES
        p1 = coef(fit("", lsob_c[m], [logp[m]] + c_, nm_, quiet=True), "log_payload")
        p2 = coef(fit("", lsob_at[m], [logp[m]] + c_, nm_, quiet=True), "log_payload")
        p3 = coef(fit("", d[m], [logp[m]] + c_, nm_, quiet=True), "log_payload")
        print(f"    {a:<10} {int(m.sum()):>5}  {p1[0]:>+8.4f} {p1[2]:>+8.2f}  {p2[0]:>+9.4f} {p2[2]:>+8.2f}  "
              f"{p3[0]:>+9.4f} {p3[1]:>7.4f} {p3[2]:>+7.2f}")

    print("\n[5] mean paired delta by airframe")
    for a in afs:
        dd = d[afid == a]
        se = dd.std(ddof=1) / math.sqrt(len(dd))
        print(f"    {afname.get(a, a):<18} n={len(dd):5d}  mean_d={dd.mean():+.4f}  se={se:.4f}  "
              f"median={np.median(dd):+.4f}")


if __name__ == "__main__":
    main()
