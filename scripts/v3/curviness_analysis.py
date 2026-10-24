"""Does commanded-path CURVATURE predict path-tracking error?

Idea (user, 2026-08-11): quantify how "curvy" each commanded path is on a single physical
axis and correlate it with the SOB2 deviation, instead of treating plan as an opaque
categorical. This turns the v3-B turn/climb families into a continuous dose-response and
tests whether a scaling law (error vs curvature, or vs commanded normal acceleration) holds.

Geometry: every v3-B plan is a 3-waypoint dogleg (takeoff wp + 2 nav wps = 2 segments, 1
vertex). Curvature is concentrated at that vertex, so we characterise each plan by:
  * turn_angle_3d  -- angle between the two 3-D segment directions at the vertex (rad).
                      Unifies turns (horizontal bend) and climbs (vertical bend) on one axis.
  * kappa          -- turn_angle_3d / arc_length  (mean curvature, 1/m; = 1/R for an arc).
  * a_normal       -- cruise_speed^2 * kappa  (commanded centripetal accel, m/s^2): the
                      acceleration the airframe must produce to negotiate the vertex at speed.
  * horiz_turn_deg, climb_angle_deg -- the horizontal- and vertical-plane components.

The scored cruise segment (missionItemIndex 2) sits on the vertex, so this curvature is the
curvature of the scored leg. straight_hXXX plans are straight (kappa=0) regardless of heading.

Usage: curviness_analysis.py <combined_v3_cruise.csv> <combined_v2_cruise.csv> <plans_dir> <label>
"""
import collections
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from robust_stats import ols_hc3_rank, wald_block  # noqa: E402

METRIC = os.environ.get("TDR_METRIC", "sob2")  # response column (sob2 | cross_track_rmse | ...)
R_EARTH = 6378137.0


def plan_waypoints(path):
    d = json.loads(Path(path).read_text())
    mission = d["mission"]
    pts = []
    for it in mission["items"]:
        p = it.get("params", [])
        if len(p) >= 7 and p[4] is not None and p[5] is not None:
            pts.append((float(p[4]), float(p[5]), float(p[6]) if p[6] is not None else 0.0))
    return mission.get("cruiseSpeed", 5.0), pts


def enu(pts):
    lat0, lon0 = pts[0][0], pts[0][1]
    out = []
    for la, lo, al in pts:
        x = math.radians(lo - lon0) * math.cos(math.radians(lat0)) * R_EARTH
        y = math.radians(la - lat0) * R_EARTH
        out.append(np.array([x, y, al]))
    return out


def plan_geometry(path):
    cruise, wps = plan_waypoints(path)
    p = enu(wps)
    # segments between consecutive waypoints; vertex angles between consecutive segments
    segs = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    arc_len = sum(float(np.linalg.norm(s)) for s in segs)
    total_turn = 0.0
    horiz_turn = 0.0
    climb_angle = 0.0
    for a, b in zip(segs, segs[1:]):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        total_turn += math.acos(cosang)
        # horizontal-plane bend
        ah, bh = a[:2], b[:2]
        if np.linalg.norm(ah) > 1e-6 and np.linalg.norm(bh) > 1e-6:
            ch = float(np.clip(np.dot(ah, bh) / (np.linalg.norm(ah) * np.linalg.norm(bh)), -1.0, 1.0))
            horiz_turn += math.acos(ch)
        # climb angle change (vertical bend)
        fpa_a = math.atan2(a[2], np.linalg.norm(a[:2]))
        fpa_b = math.atan2(b[2], np.linalg.norm(b[:2]))
        climb_angle = max(climb_angle, abs(fpa_b - fpa_a))
    kappa = total_turn / arc_len if arc_len > 0 else 0.0
    return {
        "cruise": float(cruise or 5.0),
        "arc_len_m": arc_len,
        "turn_angle_deg": math.degrees(total_turn),
        "horiz_turn_deg": math.degrees(horiz_turn),
        "climb_angle_deg": math.degrees(climb_angle),
        "kappa_per_m": kappa,
        "a_normal_m_s2": (float(cruise or 5.0) ** 2) * kappa,
    }


def num(r, k):
    try:
        v = float(r.get(k))
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    rows, seen = [], set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("from_failed_attempt") == "1":
            continue
        if (r.get("takeoff_sim_speed_factor") or "").startswith("2"):
            continue
        rid = r.get("run_id") or ""
        if not rid or rid in seen:
            continue
        if not (num(r, METRIC) > 0):
            continue
        seen.add(rid)
        rows.append(r)
    return rows


def ols(y, X, names, quiet=False, show=None):
    """Rank-aware HC3 OLS (audit pass 3 #8/#11). Returns R2; fit dict in ols.last."""
    f = ols_hc3_rank(X, y, names)
    ols.last = f
    if not quiet:
        print("   [HC3; rank %d/%d, cond(X)=%.1f%s]" % (
            f["rank_full"], f["k_full"], f["cond"],
            "; ALIASED dropped: %s" % f["aliased"] if f["aliased"] else ""))
        for nm, b, s, t in zip(f["names"], f["beta"], f["se"], f["t"]):
            if show is not None and nm not in show:
                continue
            star = " ***" if abs(t) > 3.29 else (" **" if abs(t) > 2.58 else (" *" if abs(t) > 1.96 else ""))
            print("   %-24s %+10.4f (HC3 se %.4f, t %+6.2f)%s" % (nm, b, s, t, star))
        print("   n=%d  R2=%.4f" % (f["n"], f["r2"]))
    return f["r2"]


def coef_of(f, nm):
    i = f["names"].index(nm)
    return f["beta"][i], f["se"][i], f["t"][i]


def covariates_fe(rows, extra):
    """Within-airframe covariates (audit pass 3 #1 convention): log1p(payload fraction),
    |com_y|/wheelbase, plus airframe fixed effects. No raw log mass (airframe FE absorb it)."""
    cols, names = [], []

    def add(nm, v):
        cols.append(np.nan_to_num(np.array(v, float), nan=0.0)); names.append(nm)

    add("mean_wind", [num(r, "mean_wind_m_s") for r in rows])
    add("turb_rms", [num(r, "turbulence_rms_m_s") for r in rows])
    add("log_payload", [math.log1p(num(r, "payload_mass_pct") / 100.0) for r in rows])
    add("thrust_to_weight", [num(r, "thrust_to_weight") for r in rows])
    add("inertia_scale", [num(r, "inertia_scale") for r in rows])
    add("abs_com_y_frac", [abs(num(r, "com_y_m")) / max(num(r, "wheelbase_m"), 1e-9) for r in rows])
    for nm, v in extra:
        add(nm, v)
    afs = sorted({r.get("airframe_id") for r in rows})
    for a in afs[1:]:
        add("AF[%s]" % a, [1.0 if r.get("airframe_id") == a else 0.0 for r in rows])
    return np.column_stack([np.ones(len(rows))] + cols), ["intercept"] + names


def cluster_boot_ci(X, y, clusters, name_idx, n_boot=1000, seed=12345):
    """Cluster (pairs) bootstrap of one coefficient: returns (se, lo95, hi95, n_clusters)."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    idx_by = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    kept, _ = None, None
    from robust_stats import independent_columns
    kept, _drop = independent_columns(X)
    Xk = X[:, kept]
    j = kept.index(name_idx)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by[c] for c in pick])
        beta, *_ = np.linalg.lstsq(Xk[rows], y[rows], rcond=None)
        vals[b] = beta[j]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(vals.std(ddof=1)), float(lo), float(hi), len(uniq)


def covariates(rows, extra):
    cols, names = [], []

    def add(nm, v):
        cols.append(np.nan_to_num(np.array(v, float), nan=0.0)); names.append(nm)

    add("mean_wind", [num(r, "mean_wind_m_s") for r in rows])
    add("turb_rms", [num(r, "turbulence_rms_m_s") for r in rows])
    add("log_mass", [math.log(max(num(r, "mass_kg"), 1e-9)) for r in rows])
    add("thrust_to_weight", [num(r, "thrust_to_weight") for r in rows])
    add("inertia_scale", [num(r, "inertia_scale") for r in rows])
    for nm, v in extra:
        add(nm, v)
    return np.column_stack([np.ones(len(rows))] + cols), ["intercept"] + names


def main():
    v3_path, v2_path, plans_dir, label = sys.argv[1:5]
    plans_dir = Path(plans_dir)

    # geometry per plan referenced in the data
    v3 = load(v3_path)
    plan_names = sorted({r.get("plan") for r in v3 if r.get("plan")})
    geo = {}
    for pl in plan_names:
        f = plans_dir / (pl + ".plan")
        if f.exists():
            geo[pl] = plan_geometry(f)

    print("=" * 84)
    print("PATH CURVATURE vs SOB2 -- %s segment" % label)
    print("=" * 84)
    print("\nPer-plan commanded geometry (scored leg sits on the single vertex):")
    print("   %-16s %8s %9s %9s %9s %10s" % ("plan", "arc_m", "turn deg", "horiz d", "climb d", "a_norm"))
    for pl in plan_names:
        g = geo.get(pl)
        if g:
            print("   %-16s %8.0f %9.1f %9.1f %9.1f %10.3f" % (
                pl, g["arc_len_m"], g["turn_angle_deg"], g["horiz_turn_deg"],
                g["climb_angle_deg"], g["a_normal_m_s2"]))

    # pair against v2 controls
    v2 = {r["run_id"]: r for r in load(v2_path)}
    pairs = []
    for r in v3:
        c = v2.get(r.get("paired_with_run_id") or "")
        g = geo.get(r.get("plan"))
        if c is not None and g is not None:
            d = math.log(num(r, METRIC)) - math.log(num(c, METRIC))
            pairs.append((r, g, d))
    print("\npaired rows with geometry: %d" % len(pairs))

    # --- core: mean sob2 and mean paired-delta by plan, next to its curvature ---
    print("\n1. Mean deviation by plan vs its curvature (the raw correlation):")
    print("   %-16s %6s %9s %9s %10s %10s" % ("plan", "n", "kappa", "a_norm", "mean log_sob2", "mean delta"))
    by_plan = collections.defaultdict(list)
    for r, g, d in pairs:
        by_plan[r.get("plan")].append((math.log(num(r, METRIC)), d, g))
    rowstats = []
    for pl in plan_names:
        v = by_plan.get(pl)
        if not v:
            continue
        g = v[0][2]
        ls = np.array([a for a, _b, _g in v]); dl = np.array([b for _a, b, _g in v])
        print("   %-16s %6d %9.5f %10.3f %10.4f %10.4f" % (
            pl, len(v), g["kappa_per_m"], g["a_normal_m_s2"], ls.mean(), dl.mean()))
        rowstats.append((pl, g, ls.mean(), dl.mean(), len(v)))

    # --- does curvature predict the paired delta across ALL maneuver plans? ---
    # Audit pass 3 #8: the v3-C arcs are constant 1000 m BY DESIGN (make_arc_plans.py), so the
    # chord-polyline "arc_len_m" is NOT an independent covariate -- 12 chords of a circular arc
    # are shorter than the arc by a deterministic function of curvature. The headline fits
    # therefore DROP it; it is reported only as a collinearity diagnostic (2c).
    man = [(r, g, d) for r, g, d in pairs if r.get("plan") not in
           (p for p in plan_names if p.startswith("straight"))]
    arcs = [(r, g, d) for r, g, d in man if (r.get("plan") or "").startswith("arc_")]
    for subset_name, sub in (("v3-C ARC plans only (constant 1000 m by design)", arcs),
                             ("ALL maneuver plans (arcs + v3-B doglegs/climbs; length varies in v3-B)", man)):
        if not sub or (subset_name.startswith("ALL") and len(sub) == len(arcs)):
            continue   # identical to the arc-only block (no v3-B plan geometry supplied)
        rows = [r for r, _g, _d in sub]
        d = np.array([dd for _r, _g, dd in sub])
        kappa = np.array([g["kappa_per_m"] for _r, g, _d in sub])
        anorm = [g["a_normal_m_s2"] for _r, g, _d in sub]
        arclen = np.array([g["arc_len_m"] for _r, g, _d in sub])
        print("\n" + "=" * 84)
        print("2. Paired maneuver-added error ~ curvature -- %s, n=%d" % (subset_name, len(sub)))
        print("=" * 84)

        # --- why the chord-length covariate was dropped ---
        print("\n 2.0 kappa vs chord-length covariate (documents why length is NOT a covariate):")
        print("     arc_len_m (12-chord polyline): min=%.2f max=%.2f  sd=%.3f" % (
            arclen.min(), arclen.max(), arclen.std()))
        if kappa.std() > 0 and arclen.std() > 0:
            print("     corr(kappa, arc_len_m) = %+.4f" % np.corrcoef(kappa, arclen)[0, 1])
        Xc, nc = covariates(rows, [("kappa_per_m", kappa), ("arc_len_m", arclen)])
        fz = ols_hc3_rank(Xc, d, nc)
        for nm in ("kappa_per_m", "arc_len_m"):
            if nm in nc:
                jx = nc.index(nm)
                oth = np.delete(Xc, jx, axis=1)
                bb_, *_ = np.linalg.lstsq(oth, Xc[:, jx], rcond=None)
                rs = Xc[:, jx] - oth @ bb_
                r2x = 1.0 - float(rs @ rs) / float(((Xc[:, jx] - Xc[:, jx].mean()) ** 2).sum())
                print("     VIF(%s | covariates + other) = %s" % (nm, "inf" if r2x > 1 - 1e-12 else "%.1f" % (1 / (1 - r2x))))
        print("     cond(X) of [1, covars, kappa, arc_len] = %.1f ; rank %d/%d%s" % (
            fz["cond"], fz["rank_full"], fz["k_full"],
            "; aliased: %s" % fz["aliased"] if fz["aliased"] else ""))

        print("\n 2a. HEADLINE  delta ~ kappa + covariates (chord length DROPPED), HC3:")
        X, names = covariates(rows, [("kappa_per_m", kappa)])
        ols(d, X, names, show={"intercept", "kappa_per_m"})
        k_plain = coef_of(ols.last, "kappa_per_m")
        print(" 2a'. same, raw kappa only (delta ~ 1 + kappa), HC3:")
        ols(d, np.column_stack([np.ones(len(d)), kappa]), ["intercept", "kappa_per_m"])
        print(" 2a-FE. delta ~ kappa + within-airframe covariates + AIRFRAME FE, HC3:")
        Xf, nf = covariates_fe(rows, [("kappa_per_m", kappa)])
        ols(d, Xf, nf, show={"intercept", "kappa_per_m", "log_payload", "mean_wind", "turb_rms"})
        k_fe = coef_of(ols.last, "kappa_per_m")
        clus = [r.get("model_name") or r.get("paired_with_run_id") or r.get("run_id") for r in rows]
        se_b, lo, hi, ncl = cluster_boot_ci(Xf, d, clus, nf.index("kappa_per_m"))
        print(" 2a-BOOT. cluster bootstrap (cluster=base v2 config/model_name, %d clusters, 1000 reps) "
              "of kappa slope in the FE model:" % ncl)
        print("     kappa slope = %+.2f /m  boot-se=%.2f  95%% percentile CI [%+.2f, %+.2f]" % (k_fe[0], se_b, lo, hi))
        print(" 2a-FE-HC3 summary: kappa slope %+.2f /m (HC3 se %.2f, t %+.1f); no-FE covariate model "
              "%+.2f (se %.2f, t %+.1f)" % (k_fe[0], k_fe[1], k_fe[2], k_plain[0], k_plain[1], k_plain[2]))
        print(" 2a-AF. per-airframe kappa slope (delta ~ kappa + covariates within airframe, HC3):")
        for a in sorted({r.get("airframe_id") for r in rows}):
            mk = np.array([r.get("airframe_id") == a for r in rows])
            sub_rows = [r for r, k in zip(rows, mk) if k]
            Xa, na = covariates(sub_rows, [("kappa_per_m", kappa[mk])])
            ols(d[mk], Xa, na, quiet=True)
            ka = coef_of(ols.last, "kappa_per_m")
            print("     %-10s n=%5d  kappa slope %+8.2f (HC3 se %.2f, t %+.1f)" % (a, int(mk.sum()), ka[0], ka[1], ka[2]))
        print(" NOTE: each v3-C arc row has its OWN base v2 config (19,941 rows = 19,941 distinct base configs/controls),")
        print("       so the 'cluster' bootstrap here is a pairs bootstrap (cluster size 1); there are no repeated")
        print("       base configs to cluster on. Airframe is only 5 clusters (too few for a cluster bootstrap).")
        print(" 2b. delta ~ a_normal (commanded centripetal accel), covariates, HC3:")
        X, names = covariates(rows, [("a_normal_m_s2", anorm)])
        ols(d, X, names, show={"intercept", "a_normal_m_s2"})
        print(" 2c. DIAGNOSTIC ONLY (not headline): delta ~ kappa + arc_len_m (collinear by construction), HC3:")
        X, names = covariates(rows, [("kappa_per_m", kappa), ("arc_len_m", arclen)])
        ols(d, X, names, show={"kappa_per_m", "arc_len_m"})

    print("\n NOTE (audit #8, setpoint curvature): per-run commanded-setpoint curvature (PX4's")
    print(" internally smoothed trajectory) is NOT available in the combined CSVs (only nominal plan")
    print(" geometry + scalar error metrics); it would require parsing ~20k ULogs. Skipped; kappa here")
    print(" is NOMINAL-plan curvature.")

    # --- turn family dose-response: is it monotonic in curvature? ---
    for fam, prefix in (("TURNS", "turn_"), ("CLIMBS", "climb_")):
        sub = [(pl, g, ls, dl, n) for pl, g, ls, dl, n in rowstats if pl.startswith(prefix)]
        if len(sub) >= 2:
            print("\n3.%s dose-response (mean paired delta vs curvature):" % fam)
            for pl, g, _ls, dl, n in sorted(sub, key=lambda z: z[1]["kappa_per_m"]):
                print("   %-14s kappa=%.5f a_norm=%.3f  mean_delta=%+.4f (n=%d)" % (
                    pl, g["kappa_per_m"], g["a_normal_m_s2"], dl, n))

    # --- level model: curvature vs plan dummies (parsimony) ---
    rows = [r for r, _g, _d in pairs]
    y = np.array([math.log(num(r, METRIC)) for r in rows])
    kappa = [geo[r["plan"]]["kappa_per_m"] for r in rows]
    print("\n4. Level fit: does 1 curvature term rival the plan dummies?")
    print(" 4a. log_sob2 ~ covariates + kappa:")
    X, names = covariates(rows, [("kappa_per_m", kappa)])
    r2_k = ols(y, X, names)
    plans_here = sorted({r["plan"] for r in rows})
    base = plans_here[0]
    dums = [("[%s]" % pl, [1.0 if r["plan"] == pl else 0.0 for r in rows]) for pl in plans_here[1:]]
    print(" 4b. log_sob2 ~ covariates + plan dummies (base=%s):" % base)
    X, names = covariates(rows, dums)
    r2_d = ols(y, X, names)
    print("\n   kappa-only R2=%.4f  vs  full plan-dummy R2=%.4f  (gap = non-curvature plan structure)"
          % (r2_k, r2_d))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main()
