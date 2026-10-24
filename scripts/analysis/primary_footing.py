"""Re-run every regression the paper quotes on the PRIMARY footing.

Primary footing = canonical analysis set (analysis_cruise.csv, no geometric-mean collapse), response
log(3-D SPE) (cross_track_rmse), HC3 SEs (rank-aware), covariates turbulence, mean wind, log1p(payload),
T/W, inertia, |com_y|/wheelbase, + airframe fixed effects (except leave-one-airframe-out, which has none by design).

usage (fleet): TDR_METRIC=cross_track_rmse python scripts/analysis/primary_footing.py [section ...]
sections: a (v2 direction) b (LOAO) c (v3-B) d (v3-C) e (v3-D) f (exponents) g (metric table); default all.
"""
import csv
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v3"))
from robust_stats import ols_hc3_rank, wald_block  # noqa: E402

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
M = os.environ.get("TDR_METRIC", "cross_track_rmse")
Z = 1.96
FAC = ["turb", "wind", "lpay", "tw", "inert", "comf"]
FAC_LABEL = {"turb": "turbulence", "wind": "mean wind", "lpay": "log(1+payload)", "tw": "T/W", "inert": "inertia", "comf": "|com_y|/wheelbase"}


def num(r, k):
    try:
        v = float(r[k])
        return v if math.isfinite(v) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load(path, metric=M, block=None):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if block is not None and r.get("v3_block") != block:
            continue
        if num(r, metric) > 0:
            rows.append(r)
    return rows


def covs(rows):
    return {"turb": np.array([num(r, "turbulence_rms_m_s") for r in rows]),
            "wind": np.array([num(r, "mean_wind_m_s") for r in rows]),
            "lpay": np.array([math.log1p(num(r, "payload_mass_pct") / 100.0) for r in rows]),
            "tw": np.array([num(r, "thrust_to_weight") for r in rows]),
            "inert": np.array([num(r, "inertia_scale") for r in rows]),
            "comf": np.array([abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9) for r in rows])}


def afdummies(rows, afs=None):
    afs = afs or sorted({r["airframe_id"] for r in rows})
    return {("AF[%s]" % a): np.array([1.0 if r["airframe_id"] == a else 0.0 for r in rows]) for a in afs[1:]}


def build(rows, extra=None, fac=FAC, af=True):
    c = covs(rows)
    cols, names = [np.ones(len(rows))], ["intercept"]
    for k in fac:
        cols.append(c[k]); names.append(k)
    if af:
        for n, v in afdummies(rows).items():
            cols.append(v); names.append(n)
    for n, v in (extra or {}).items():
        cols.append(np.asarray(v, float)); names.append(n)
    return np.column_stack(cols), names


def fit(rows, extra=None, fac=FAC, af=True, y=None):
    X, names = build(rows, extra, fac, af)
    if y is None:
        y = np.log(np.array([num(r, M) for r in rows]))
    return ols_hc3_rank(X, y, names)


def coef(f, nm):
    i = f["names"].index(nm)
    return f["beta"][i], f["se"][i], f["t"][i], f["p"][i]


def show(f, only=None, label=""):
    print(f"   {label} n={f['n']} R2={f['r2']:.4f} rank {len(f['names'])}/{f['k_full']}" + (f" ALIASED {f['aliased']}" if f["aliased"] else ""))
    for nm, b, s, t, p in zip(f["names"], f["beta"], f["se"], f["t"], f["p"]):
        if only is None or nm in only:
            print(f"      {nm:<22}{b:+10.4f}  se {s:.4f}  t {t:+8.2f}  p {p:.2g}")


def pw(x):
    return math.erfc(abs(x) / math.sqrt(2))


# ---------------------------------------------------------------- (a) v2 direction
def sec_a():
    print("=" * 100); print("(a) v2 WIND DIRECTION on the primary footing  (analysis_cruise, HC3 rank-aware, AF FE)"); print("=" * 100)
    rows = load("results/campaign_v2/analysis_cruise.csv")
    d = np.array([int(round(num(r, "wind_direction_deg"))) % 360 for r in rows])
    dn = {0: "tailwind(0)", 90: "cross-left(90)", 180: "headwind(180)", 270: "cross-right(270)"}
    print("   direction counts:", {dn[k]: int((d == k).sum()) for k in dn})
    lvl = {f"DIR[{dn[k]}]": (d == k).astype(float) for k in (90, 180, 270)}
    c = covs(rows)
    inter = {}
    for k in (90, 180, 270):
        inter[f"wind x {dn[k]}"] = (d == k) * c["wind"]
        inter[f"turb x {dn[k]}"] = (d == k) * c["turb"]
    f1 = fit(rows); f2 = fit(rows, lvl); f3 = fit(rows, {**lvl, **inter})
    print(f"   M1 primary six factors + AF FE: R2 {f1['r2']:.4f} (n={f1['n']})")
    print(f"   M2 + direction level shifts (base tailwind): R2 {f2['r2']:.4f}")
    print(f"   M3 + direction-specific wind and turbulence slopes: R2 {f3['r2']:.4f}")
    ch, dfv, p, F = wald_block(f2, list(lvl))
    print(f"   HC3 Wald M1->M2 (level shifts): chi2={ch:.1f} df={dfv} p={p:.3g}")
    ch, dfv, p, F = wald_block(f3, list(inter))
    print(f"   HC3 Wald M2->M3 (slopes): chi2={ch:.1f} df={dfv} p={p:.3g}")
    print("   model-based error ratios vs tailwind = exp(level-shift coefficient in M2), HC3 95% CI:")
    for k in (90, 180, 270):
        b, s, t, pp = coef(f2, f"DIR[{dn[k]}]")
        print(f"      {dn[k]:<18} coef {b:+.4f} (se {s:.4f})  ratio {math.exp(b):.3f}  CI [{math.exp(b-Z*s):.3f}, {math.exp(b+Z*s):.3f}]")
    y = np.log(np.array([num(r, M) for r in rows]))
    gm = {k: float(y[d == k].mean()) for k in dn}
    print("   unadjusted geometric-mean ratios vs tailwind:", {dn[k]: round(math.exp(gm[k] - gm[0]), 3) for k in (90, 180, 270)},
          "| geometric means:", {dn[k]: round(math.exp(gm[k]), 4) for k in dn})
    cr = math.exp((gm[90] + gm[270]) / 2 - gm[0])
    print(f"   crosswind (avg of 90/270) / tailwind unadjusted: {cr:.3f}")
    print("   direction-specific slopes (M3; slope = base + interaction; HC3 se via covariance):")
    names = f3["names"]; cov = f3["cov"]
    for fnm, base in (("wind", "wind"), ("turb", "turb")):
        ib = names.index(base)
        b0, s0 = f3["beta"][ib], f3["se"][ib]
        print(f"      {fnm} tailwind: {b0:+.4f} (se {s0:.4f})")
        for k in (90, 180, 270):
            ii = names.index(f"{fnm} x {dn[k]}")
            bs = f3["beta"][ib] + f3["beta"][ii]
            ss = math.sqrt(cov[ib, ib] + cov[ii, ii] + 2 * cov[ib, ii])
            print(f"      {fnm} {dn[k]}: {bs:+.4f} (se {ss:.4f})")


# ---------------------------------------------------------------- (b) LOAO
def sec_b():
    print("=" * 100); print("(b) LEAVE-ONE-AIRFRAME-OUT on the primary footing (analysis_cruise; no airframe FE by design)"); print("=" * 100)
    rows = load("results/campaign_v2/analysis_cruise.csv")
    afs = sorted({r["airframe_id"] for r in rows})
    yall = np.log(np.array([num(r, M) for r in rows]))
    c = covs(rows)
    d = np.array([int(round(num(r, "wind_direction_deg"))) % 360 for r in rows])
    base = [np.ones(len(rows)), c["turb"], c["wind"], c["lpay"], c["tw"], c["inert"], c["comf"],
            np.array([math.log(max(num(r, "wheelbase_m"), 1e-9)) for r in rows]),
            np.array([math.log(max(num(r, "disk_loading_kg_m2"), 1e-9)) for r in rows])]
    dircols = []
    for k in (90, 180, 270):
        dircols += [(d == k).astype(float), (d == k) * c["wind"], (d == k) * c["turb"]]
    rng = np.random.default_rng(20261012)
    for with_dir in (False, True):
        X = np.column_stack(base + (dircols if with_dir else []))
        print(f"\n   {'WITH' if with_dir else 'WITHOUT'} direction terms  (features: six primary factors, log wheelbase, log disk loading)")
        print(f"   {'held out':<10}{'n':>7}{'raw R2':>9}{'raw RMSE':>10}{'bias':>8}{'in-sample R2':>14}{'offset(10% split) R2':>22}")
        res = defaultdict(list)
        arf = np.array([r["airframe_id"] for r in rows])
        for h in afs:
            te = arf == h; tr = ~te
            b, *_ = np.linalg.lstsq(X[tr], yall[tr], rcond=None)
            pred = X[te] @ b; yt = yall[te]
            sst = ((yt - yt.mean()) ** 2).sum()
            r2 = 1 - ((yt - pred) ** 2).sum() / sst
            rmse = math.sqrt(((yt - pred) ** 2).mean()); bias = float((yt - pred).mean())
            b2, *_ = np.linalg.lstsq(X[te], yt, rcond=None)
            r2in = 1 - ((yt - X[te] @ b2) ** 2).sum() / sst
            idx = np.where(te)[0]; rng.shuffle(idx)
            cal = idx[: max(1, len(idx) // 10)]; ev = idx[len(cal):]
            off = float((yall[cal] - X[cal] @ b).mean())
            ye = yall[ev]; r2off = 1 - ((ye - (X[ev] @ b + off)) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
            print(f"   {h:<10}{int(te.sum()):>7}{r2:>9.3f}{rmse:>10.4f}{bias:>+8.3f}{r2in:>14.3f}{r2off:>22.3f}")
            res["raw"].append(r2); res["in"].append(r2in); res["off"].append(r2off)
        print(f"   {'MEAN':<10}{'':>7}{np.mean(res['raw']):>9.3f}{'':>10}{'':>8}{np.mean(res['in']):>14.3f}{np.mean(res['off']):>22.3f}"
              f"   raw range [{min(res['raw']):.3f}, {max(res['raw']):.3f}]")


# ---------------------------------------------------------------- (c) v3-B
def rel_class(wind_deg, heading_deg):
    rel = (wind_deg - heading_deg) % 360.0
    lab = {0: "tail", 45: "tail-cross", 90: "cross", 135: "head-cross", 180: "head", 225: "head-cross", 270: "cross", 315: "tail-cross"}
    return lab.get(int(round(rel / 45.0)) % 8 * 45, "oblique")


def dummies(labels, base):
    levels = sorted(set(labels) - {base})
    return {f"[{lv}]": np.array([1.0 if x == lv else 0.0 for x in labels]) for lv in levels}


def sec_c():
    print("=" * 100); print("(c) v3-B on the primary footing (campaign_v3 analysis_cruise; HC3 rank-aware; six primary covariates + AF FE)"); print("=" * 100)
    b1 = load("results/campaign_v3/analysis_cruise.csv", block="b1")
    rel = [rel_class(num(r, "wind_direction_deg"), num(r, "course_heading_deg")) for r in b1]
    ab = ["abs%d" % int(num(r, "wind_direction_deg")) for r in b1]
    hd = ["h%03d" % int(num(r, "course_heading_deg")) for r in b1]
    R, A, H = dummies(rel, "tail"), dummies(ab, "abs0"), dummies(hd, "h045")
    print(f"   b1 heading block n={len(b1)}; relative-class counts {dict(Counter(rel))}")
    f0 = fit(b1); fR = fit(b1, R); fA = fit(b1, A); fRA = fit(b1, {**R, **A}); fRH = fit(b1, {**R, **H}); fRAH = fit(b1, {**R, **A, **H})
    print(f"   R2: covariates+AF {f0['r2']:.4f} | + relative {fR['r2']:.4f} | + absolute {fA['r2']:.4f} | + both {fRA['r2']:.4f} | +rel+heading {fRH['r2']:.4f}")
    def tst(lbl, big, block, small):
        ch, dfv, p, F = wald_block(big, list(block))
        print(f"   {lbl:<34} dR2={big['r2']-small['r2']:+.4f}  HC3 Wald chi2={ch:.2f} df={dfv} p={p:.3g}")
    tst("cov -> + RELATIVE", fR, R, f0)
    tst("cov -> + ABSOLUTE", fA, A, f0)
    tst("+REL -> + ABSOLUTE", fRA, A, fR)
    tst("+ABS -> + RELATIVE", fRA, R, fA)
    tst("+REL -> + HEADING", fRH, H, fR)
    tst("+REL+ABS -> + HEADING", fRAH, H, fRA)
    print("   relative-class coefficients (model + RELATIVE; base tail):"); show(fR, only=list(R), label="REL")
    print("   covariate coefficients in that model:"); show(fR, only=[n for n in fR["names"] if n in FAC], label="REL covariates")
    # paired transitions
    v2c = {r["run_id"]: r for r in csv.DictReader(open("results/campaign_v2/combined_cruise.csv", encoding="utf-8"))}
    tr = defaultdict(list)
    for r in b1:
        c = v2c.get(r.get("paired_with_run_id") or "")
        if c is None or not num(c, M) > 0:
            continue
        tr[(rel_class(num(c, "wind_direction_deg"), 0.0), rel_class(num(r, "wind_direction_deg"), num(r, "course_heading_deg")))].append(
            math.log(num(r, M)) - math.log(num(c, M)))
    print("   paired deltas by relative-class transition (v2 control -> v3):   n   mean   SE")
    for k in sorted(tr):
        v = tr[k]; print(f"      {k[0]}->{k[1]:<11}{len(v):>6}{statistics.mean(v):>+9.4f}{statistics.stdev(v)/math.sqrt(len(v)):>8.4f}")
    # b2
    b2 = load("results/campaign_v3/analysis_cruise.csv", block="b2")
    d2 = []
    keep = []
    for r in b2:
        c = v2c.get(r.get("paired_with_run_id") or "")
        if c is not None and num(c, M) > 0:
            keep.append(r); d2.append(math.log(num(r, M)) - math.log(num(c, M)))
    d2 = np.array(d2)
    plan = np.array([r["plan"] for r in keep])
    print(f"\n   b2 maneuvers n={len(keep)} (paired deltas, HC3)")
    print("   mean paired change by plan (intercept-only HC3 SE):")
    for pl in sorted(set(plan)):
        m = plan == pl
        X1 = np.ones((int(m.sum()), 1)); ff = ols_hc3_rank(X1, d2[m], ["mean"])
        print(f"      {pl:<11} n={int(m.sum()):>5} mean {ff['beta'][0]:+.4f} se {ff['se'][0]:.4f} t {ff['t'][0]:+.1f} p {ff['p'][0]:.2g}")
    for fam, sel in (("turns", [p.startswith("turn") for p in plan]), ("climbs", [p.startswith("climb") for p in plan])):
        m = np.array(sel); sub = [r for r, k in zip(keep, m) if k]
        ex = dummies([r["plan"] for r in sub], sorted({r["plan"] for r in sub})[0])
        ff = fit(sub, ex, y=d2[m])
        mean_ff = ols_hc3_rank(np.ones((len(sub), 1)), d2[m], ["mean"])
        print(f"   {fam}: pooled mean {mean_ff['beta'][0]:+.4f} (HC3 se {mean_ff['se'][0]:.4f}, n={len(sub)}); paired-delta regression (covariates + plan + AF FE):")
        show(ff, only=FAC + list(ex), label=fam)
    ff = fit(keep, dummies(list(plan), "turn_l100"))
    print("   b2 LEVEL fit with plan dummies (base turn_l100):"); show(ff, only=FAC + [n for n in ff["names"] if n.startswith("[")], label="b2 level")


# ---------------------------------------------------------------- (d) v3-C
def sec_d():
    print("=" * 100); print("(d) v3-C curvature on the primary footing (campaign_v3_full block c1; HC3; delta vs the config's v2 flight)"); print("=" * 100)
    import curviness_analysis as ca
    rows = load("results/campaign_v3_full/analysis_cruise.csv", block="c1")
    v2c = {r["run_id"]: r for r in csv.DictReader(open("results/campaign_v2/combined_cruise.csv", encoding="utf-8"))}
    geo, chord_m = {}, {}
    for pl in sorted({r["plan"] for r in rows}):
        path = Path("generated/sobol_v2/plans") / (pl + ".plan")
        geo[pl] = ca.plan_geometry(path)
        _, wps = ca.plan_waypoints(path); p = ca.enu(wps)
        chord_m[pl] = float(np.linalg.norm((p[-1] - p[0])[:2]))  # start-to-end horizontal chord (arc_len_m is the polyline length)
    keep, dl, kap, chord = [], [], [], []
    for r in rows:
        c = v2c.get(r.get("paired_with_run_id") or "")
        if c is None or not num(c, M) > 0 or r["plan"] not in geo:
            continue
        keep.append(r); dl.append(math.log(num(r, M)) - math.log(num(c, M)))
        kap.append(geo[r["plan"]]["kappa_per_m"]); chord.append(chord_m[r["plan"]])
    dl = np.array(dl); kap = np.array(kap); chord = np.array(chord)
    print(f"   n={len(keep)} arcs; start-to-end chord {chord.min():.1f}-{chord.max():.1f} m; corr(kappa, chord) = {np.corrcoef(kap, chord)[0,1]:+.4f}")
    f_raw = ols_hc3_rank(np.column_stack([np.ones(len(kap)), kap]), dl, ["intercept", "kappa"])
    print("   delta ~ kappa only:"); show(f_raw, only=["kappa"])
    f_cov = fit(keep, {"kappa": kap}, af=False, y=dl); print("   delta ~ kappa + six primary covariates (no AF FE):"); show(f_cov, only=["kappa"] + FAC)
    f_fe = fit(keep, {"kappa": kap}, y=dl); print("   delta ~ kappa + six primary covariates + AF FE (HEADLINE):"); show(f_fe, only=["kappa"] + FAC)
    # Functional form. For constant-length arcs the start-to-end chord is a fixed function of the bend
    # (chord/length = sin(theta/2)/(theta/2)), i.e. another measure of curvature, so adding it -- or kappa^2 -- tests
    # whether the effect is linear in kappa rather than separating a second cause.
    f_q = fit(keep, {"kappa": kap, "kappa_sq": kap ** 2}, y=dl); print("   functional form: + kappa^2:"); show(f_q, only=["kappa", "kappa_sq"])
    f_ch = fit(keep, {"kappa": kap, "chord_m": chord}, y=dl); print("   functional form: + start-to-end chord (equivalent):"); show(f_ch, only=["kappa", "chord_m"])
    X, names = build(keep, {"kappa": kap}); nb = 1000; rng = np.random.default_rng(20261012)
    ik = names.index("kappa"); bs = np.empty(nb)
    for i in range(nb):
        s = rng.integers(0, len(dl), len(dl)); b, *_ = np.linalg.lstsq(X[s], dl[s], rcond=None); bs[i] = b[ik]
    print(f"   pairs bootstrap (1000) of headline kappa slope: se {bs.std(ddof=1):.2f}  95% CI [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
    print("   per-airframe kappa slope (delta ~ kappa + six covariates within airframe, HC3):")
    af = np.array([r["airframe_id"] for r in keep])
    for a in sorted(set(af)):
        m = af == a; sub = [r for r, k in zip(keep, m) if k]
        ff = fit(sub, {"kappa": kap[m]}, af=False, y=dl[m]); b, s, t, p = coef(ff, "kappa")
        print(f"      {a:<10} n={int(m.sum()):>5} kappa {b:+8.2f} (se {s:.2f}, t {t:+.1f})")


# ---------------------------------------------------------------- (e) v3-D
def sec_e():
    print("=" * 100); print("(e) v3-D on the primary footing (paired Dryden - white)"); print("=" * 100)
    rows = load("results/campaign_v3d/analysis_cruise.csv")
    v2c = {r["run_id"]: r for r in csv.DictReader(open("results/campaign_v2/combined_cruise.csv", encoding="utf-8"))}
    keep, dl = [], []
    for r in rows:
        c = v2c.get(r.get("paired_with_run_id") or "")
        if c is not None and num(c, M) > 0:
            keep.append(r); dl.append(math.log(num(r, M)) - math.log(num(c, M)))
    dl = np.array(dl)
    ff = ols_hc3_rank(np.ones((len(dl), 1)), dl, ["mean"])
    print(f"   n={len(dl)} mean paired change {ff['beta'][0]:+.4f}  HC3 se {ff['se'][0]:.4f}  (plain SE {dl.std(ddof=1)/math.sqrt(len(dl)):.4f}); ratio exp = {math.exp(ff['beta'][0]):.3f}; median {np.median(dl):+.4f}")
    f2 = fit(keep, y=dl); print("   paired delta ~ six primary covariates + AF FE (HC3):"); show(f2, only=FAC)
    f3 = fit(rows); print("   v3-D level fit (Dryden) six covariates + AF FE:"); show(f3, only=FAC)
    v2rows = [v2c[r["paired_with_run_id"]] for r in keep]
    f4 = fit(v2rows); print("   v2 paired subset (white) level fit six covariates + AF FE:"); show(f4, only=FAC)


# ---------------------------------------------------------------- (f) exponents
def sec_f():
    print("=" * 100); print("(f) log-log exponents on the canonical v2 analysis set (HC3; airframe dummies)"); print("=" * 100)
    rows = load("results/campaign_v2/analysis_cruise.csv")
    for nm, col in (("turbulence sigma_v", "turbulence_rms_m_s"), ("mean wind U", "mean_wind_m_s")):
        sub = [r for r in rows if num(r, col) > 0.2]
        lx = np.log(np.array([num(r, col) for r in sub]))
        f1 = ols_hc3_rank(np.column_stack([np.ones(len(sub)), lx] + list(afdummies(sub).values())),
                          np.log(np.array([num(r, M) for r in sub])), ["intercept", "logx"] + list(afdummies(sub)))
        b, s, t, p = coef(f1, "logx")
        # with the other primary covariates (log x replaces the linear term of that factor)
        fac = [k for k in FAC if not ((col == "turbulence_rms_m_s" and k == "turb") or (col == "mean_wind_m_s" and k == "wind"))]
        f2 = fit(sub, {"logx": lx}, fac=fac)
        b2, s2, t2, p2 = coef(f2, "logx")
        print(f"   {nm:<20} n={len(sub)}: log-log + AF dummies only: p={b:.4f} HC3 se {s:.4f} 95% CI [{b-Z*s:.4f}, {b+Z*s:.4f}] | with other primary covariates: p={b2:.4f} (se {s2:.4f}) CI [{b2-Z*s2:.4f}, {b2+Z*s2:.4f}]")
    lm = np.array([math.log(max(num(r, "mass_kg"), 1e-9)) for r in rows])
    f1 = ols_hc3_rank(np.column_stack([np.ones(len(rows)), lm] + list(afdummies(rows).values())),
                      np.log(np.array([num(r, M) for r in rows])), ["intercept", "logm"] + list(afdummies(rows)))
    b, s, t, p = coef(f1, "logm")
    print(f"   mass (within airframe) n={len(rows)}: p={b:.4f} HC3 se {s:.4f} CI [{b-Z*s:.4f}, {b+Z*s:.4f}]")


# ---------------------------------------------------------------- (g) metric table
def sec_g():
    print("=" * 100); print("(g) metric-robustness table on the primary footing (v2 analysis set; six factors + AF FE; HC3)"); print("=" * 100)
    rows = [r for r in csv.DictReader(open("results/campaign_v2/analysis_cruise.csv", encoding="utf-8")) if num(r, M) > 0]
    def metric(r, m):
        if m == "horizontal_rmse":
            a, b = num(r, "rms_error"), num(r, "altitude_rmse")
            return math.sqrt(max(a * a - b * b, 0.0)) if a == a and b == b else float("nan")
        return num(r, m)
    base = None
    print(f"   {'metric':<20}{'n':>8}{'R2':>8}{'sign agree':>12}{'signif agree':>14}")
    for m in ("cross_track_rmse", "rms_error", "horizontal_rmse", "max_error", "cross_track_max", "altitude_rmse"):
        sub = [r for r in rows if metric(r, m) > 0]
        y = np.log(np.array([metric(r, m) for r in sub]))
        f = fit(sub, y=y)
        bs = [coef(f, k) for k in FAC]
        if base is None:
            base = {k: coef(f, k) for k in FAC}
            print(f"   {m:<20}{f['n']:>8}{f['r2']:>8.3f}{'n/a':>12}{'n/a':>14}")
            continue
        sign = sum(1 for k, c in zip(FAC, bs) if np.sign(c[0]) == np.sign(base[k][0]))
        sig = sum(1 for k, c in zip(FAC, bs) if (abs(c[2]) > 1.96) == (abs(base[k][2]) > 1.96) and np.sign(c[0]) == np.sign(base[k][0]))
        print(f"   {m:<20}{f['n']:>8}{f['r2']:>8.3f}{str(sign)+'/6':>12}{str(sig)+'/6':>14}")


if __name__ == "__main__":
    secs = sys.argv[1:] or list("abcdefg")
    for s in secs:
        globals()["sec_" + s]()
        print()
