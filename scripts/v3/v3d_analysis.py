"""v3-D analysis: does an accepted-model (MIL-STD Dryden) turbulence field change the
v2 results vs the crude white-noise WindEffects model?

Every _dw run is a v2 config flown with turb_model=dryden (mean + anisotropic Dryden
gusts, airspeed axes, altitude length scales, NO sinusoid) instead of the v2 white model
(mean + 20 s sinusoid + gaussian white noise). SAME model_name + wind_seed + plan +
mean + direction + turbulence_rms, paired via v3_d.csv paired_with_run_id. Within-pair
delta  d = ln(sob2_dryden) - ln(sob2_white)  reads as "crude white model -> Dryden model".

v2 headline (cruise): turbulence is the dominant effect (turb ~ +0.95). The question:
does a physically-correct turbulence spectrum change that sensitivity or the deviation?

Usage: v3d_analysis.py <combined_v3d_seg.csv> <combined_v2_seg.csv> <v3_d_design.csv> <label>
"""
import csv
import math
import os
import sys

import numpy as np

METRIC = os.environ.get("TDR_METRIC", "sob2")  # response column (sob2 | cross_track_rmse | ...)


def num(r, k):
    try:
        v = float(r.get(k))
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_sob2(path):
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


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(1, n - k)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    t = beta / np.where(se > 0, se, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, t, r2, n


def report_ols(title, y, cols, names):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, se, t, r2, n = ols(y, X)
    print(f"\n{title}  (n={n}, R2={r2:.4f})")
    for nm, b, s, tv in zip(["const"] + names, beta, se, t):
        print(f"   {nm:<14} {b:+.4f}  se={s:.4f}  t={tv:+.2f}")
    return beta, t, names


def main():
    v3d_path, v2_path, design_path, label = sys.argv[1:5]
    dry = load_sob2(v3d_path)
    v2 = load_sob2(v2_path)
    design = {r["run_id"]: r for r in csv.DictReader(open(design_path, encoding="utf-8"))}
    n_design = len(design)

    pairs = []
    n_dry_valid = n_no_ctrl = 0
    for rid, drow in design.items():
        drow_out = dry.get(rid)
        if drow_out is None:
            continue
        n_dry_valid += 1
        cid = drow.get("paired_with_run_id") or ""
        ctrl = v2.get(cid)
        if ctrl is None:
            n_no_ctrl += 1
            continue
        s_d = num(drow_out, METRIC)
        s_w = num(ctrl, METRIC)
        if not (s_d > 0 and s_w > 0):
            continue
        pairs.append((drow, s_d, s_w))

    print(f"=== v3-D Dryden analysis [{label}] ===")
    print(f"designed _dw configs   : {n_design}")
    print(f"_dw with valid sob2    : {n_dry_valid}  ({100*n_dry_valid/n_design:.1f}%)")
    print(f"  of those no v2 ctrl  : {n_no_ctrl}")
    print(f"paired dryden<->white   : {len(pairs)}")
    if len(pairs) < 50:
        print("too few pairs -- stopping"); return

    d = np.array([math.log(sd) - math.log(sw) for (_, sd, sw) in pairs])
    turb = np.array([num(dr, "turbulence_rms_m_s") for (dr, _, _) in pairs])
    wind = np.array([num(dr, "mean_wind_m_s") for (dr, _, _) in pairs])
    logm = np.array([math.log(num(dr, "mass_kg")) for (dr, _, _) in pairs])
    tw = np.array([num(dr, "thrust_to_weight") for (dr, _, _) in pairs])
    inert = np.array([num(dr, "inertia_scale") for (dr, _, _) in pairs])
    comy = np.array([abs(num(dr, "com_y_m")) for (dr, _, _) in pairs])
    lsob_d = np.array([math.log(sd) for (_, sd, _) in pairs])
    lsob_w = np.array([math.log(sw) for (_, _, sw) in pairs])
    af = [dr.get("airframe_name") or dr.get("model_name", "") for (dr, _, _) in pairs]

    cov = [turb, wind, logm, tw, inert, comy]
    covn = ["turb", "wind", "log_mass", "t_to_w", "inertia", "abs_com_y"]

    mean_d = float(d.mean()); se_d = float(d.std(ddof=1) / math.sqrt(len(d)))
    print(f"\n[1] mean paired delta d = ln(dryden) - ln(white): {mean_d:+.4f}"
          f"  se={se_d:.4f}  t={mean_d/se_d:+.2f}   (>0 = Dryden gives MORE deviation)")
    print(f"    median d={np.median(d):+.4f}   frac d>0={float((d>0).mean()):.3f}"
          f"   mean |d|={float(np.abs(d).mean()):.4f}")

    report_ols("[2] paired delta  d ~ turb", d, [turb], ["turb"])
    report_ols("[3] paired delta  d ~ covariates", d, cov, covn)

    print("\n[4] does the turbulence sensitivity change? ln(sob2) ~ covariates, same configs")
    bw, tw_, _ = report_ols("    white:  ln(sob2_v2)  ~ covars", lsob_w, cov, covn)
    bd, td_, _ = report_ols("    dryden: ln(sob2_dw)  ~ covars", lsob_d, cov, covn)
    iturb = covn.index("turb")
    print(f"\n    white turb slope  = {bw[iturb+1]:+.4f} (t={tw_[iturb+1]:+.2f})")
    print(f"    dryden turb slope = {bd[iturb+1]:+.4f} (t={td_[iturb+1]:+.2f})")
    print(f"    change (dryden-white) = {bd[iturb+1]-bw[iturb+1]:+.4f}")

    print("\n[5] mean paired delta by airframe")
    for a in sorted(set(af)):
        dd = d[[i for i, x in enumerate(af) if x == a]]
        print(f"    {a:<20} n={len(dd):5d}  mean_d={dd.mean():+.4f}  median={np.median(dd):+.4f}")


if __name__ == "__main__":
    main()
