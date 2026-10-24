"""Tier-2 validation of the COMPILED Dryden gz plugin.

Reads the plugin's own <log_file> CSV (t,entity,u,v,w,V,h_ft,Lu,Lv,Lw,windx,windy,windz), i.e.
the gust the C++ actually generated inside gz-sim, and compares per entity/axis against

  (a) the analytic MIL-STD-1797A Dryden spec (target sigma, ACF, UN-rescaled one-sided PSD)
  (b) the Python mirror in validate_dryden.py run with the same parameters and record length.

Metrics per entity/axis: achieved RMS / target sigma (with expected sampling error), integral
correlation time of the ACF over 3 tau vs the analytic value over the same window, Welch PSD
(long segments) / analytic PSD in three bands (below corner, corner..10x, above 10x), total
integrated PSD / sigma^2.  Plus cross-correlation between entities (links) and between axes.

Case spec:  --case label:path.csv:fixed:su,sv,sw     (sigma given in the world)
            --case label:path.csv:w20:7.0            (MIL altitude formula, sigma from logged h)
Usage: validate_dryden_plugin.py --case ... [--case ...] [--json out.json] [--txt out.txt] [--no-mirror]
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_dryden as vd  # analytic functions + python mirror

TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def load(path):
    try:
        import pandas as pd
        return {k: v.to_numpy() for k, v in pd.read_csv(path).items()}
    except ImportError:
        d = np.genfromtxt(path, delimiter=",", names=True)
        return {k: d[k] for k in d.dtype.names}


def psd_one_sided(f, sigma, tau, kind):
    w = 2 * np.pi * f
    if kind == "u":
        return 4 * sigma ** 2 * tau / (1 + (tau * w) ** 2)
    return 2 * sigma ** 2 * tau * (1 + 3 * (tau * w) ** 2) / (1 + (tau * w) ** 2) ** 2


def welch_long(x, dt, nseg):
    x = x - x.mean()
    win = np.hanning(nseg)
    norm = (win ** 2).sum() / dt
    acc, k = 0, 0
    for s in range(0, len(x) - nseg + 1, nseg // 2):
        X = np.fft.rfft(x[s:s + nseg] * win)
        acc = acc + np.abs(X) ** 2 / norm
        k += 1
    P = acc / k
    P[1:-1] *= 2.0
    return np.fft.rfftfreq(nseg, dt), P, k


def stats(x, dt, sigma, tau, kind, nseg):
    """Statistics of one gust series against analytic (sigma, tau)."""
    n = len(x)
    T = n * dt
    tau_int = tau if kind == "u" else 0.5 * tau
    rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    n_eff = T / (2 * tau_int)
    exp_err = 1 / math.sqrt(2 * n_eff)
    # integral correlation time over 3 tau
    fl = int(round(3 * tau / dt))
    emp = vd.empirical_acf(x, fl)
    lags = np.arange(fl + 1) * dt
    ana = vd.acf_u(lags, tau) if kind == "u" else vd.acf_vw(lags, tau)
    tau_hat = float(TRAPZ(emp, dx=dt))
    tau_ana_win = float(TRAPZ(ana, dx=dt))
    nl = int(round(tau / dt))
    near = float(np.max(np.abs(emp[:nl + 1] - ana[:nl + 1])))
    # PSD (absolute, no rescaling)
    f, P, nsegs = welch_long(x, dt, nseg)
    S = psd_one_sided(f, sigma, tau, kind)
    fc = 1 / (2 * math.pi * tau)
    flo = 2.0 / (nseg * dt)
    bands = {"lo(<fc)": (flo, fc), "mid(fc-10fc)": (fc, 10 * fc), "hi(>10fc,<10Hz)": (10 * fc, 10.0)}
    ratios = {}
    for name, (a, b) in bands.items():
        m = (f >= a) & (f < b)
        ratios[name] = float(TRAPZ(P[m], f[m]) / TRAPZ(S[m], f[m])) if m.sum() > 2 else None
    tot = float(TRAPZ(P, f) / sigma ** 2)
    hf = (f > 0.5) & (f < 20)
    slope = float(np.polyfit(np.log10(f[hf]), np.log10(P[hf]), 1)[0])
    return {"tau_s": tau, "tau_int_analytic_s": tau_int, "target_sigma": sigma, "rms": rms,
            "rms_over_sigma": rms / sigma, "expected_rms_rel_err": exp_err,
            "rms_err_in_expected_sd": (rms / sigma - 1) / exp_err,
            "tau_hat_3tau_s": tau_hat, "tau_analytic_3tau_window_s": tau_ana_win,
            "tau_hat_over_analytic_window": tau_hat / tau_ana_win,
            "near_acf_maxdev_1tau": near, "welch_segments": nsegs,
            "psd_ratio_bands": ratios, "psd_integral_over_sigma2": tot, "hf_slope": slope}


def mirror_stats(sigma, L, V, dt, n, kind, nseg, rng):
    runner = vd.run_u_ou if kind == "u" else vd.run_vw_2state
    x, tau = runner(sigma, L, V, dt, n, rng)
    return stats(x, dt, sigma, tau, kind, nseg)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--nseg", type=int, default=1 << 17)
    ap.add_argument("--json", default=None)
    ap.add_argument("--txt", default=None)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()

    out, lines = {"cases": {}}, []
    P = lambda s="": (print(s), lines.append(s))
    series = {}  # (label, entity, axis) -> array, for inter-run correlation
    rng = np.random.default_rng(a.seed)

    for spec in a.case:
        label, path, mode, val = spec.split(":", 3)
        d = load(path)
        t, ent = d["t"], d["entity"].astype(int)
        ents = sorted(set(ent.tolist()))
        case = {"mode": mode, "entities": {}, "cross": {}}
        P("=" * 100)
        P(f"CASE {label}  ({mode} {val})  entities={ents}  rows={len(t):,}")
        gust = {}
        for e in ents:
            m = ent == e
            te = t[m]
            dt = float(np.median(np.diff(te)))
            V = d["V"][m]
            hft = d["h_ft"][m]
            info = {"n": int(m.sum()), "dt": dt, "T_s": float(te[-1] - te[0]),
                    "V_mean": float(V.mean()), "V_min": float(V.min()), "V_max": float(V.max()),
                    "frac_V_above_floor_1pct": float(np.mean(V > V.min() * 1.01)),
                    "h_ft_mean": float(hft.mean()), "h_ft_range": [float(hft.min()), float(hft.max())]}
            Ls = {"u": float(d["Lu"][m].mean()), "v": float(d["Lv"][m].mean()), "w": float(d["Lw"][m].mean())}
            if mode == "fixed":
                sg = dict(zip("uvw", map(float, val.split(","))))
            else:
                (_, _, _), (su, sv, sw) = vd.mil_scales_and_sigma(float(hft.mean()), float(val))
                sg = {"u": su, "v": sv, "w": sw}
            info["L_m"], info["sigma_target"] = Ls, sg
            P(f"\n -- entity {e}: dt={dt*1e3:.3f} ms T={info['T_s']:.0f}s V mean/min/max={info['V_mean']:.3f}/{info['V_min']:.3f}/{info['V_max']:.3f} "
              f"(frac V>floor {info['frac_V_above_floor_1pct']:.3g}) h_ft={info['h_ft_mean']:.2f} "
              f"L={Ls['u']:.1f}/{Ls['v']:.1f}/{Ls['w']:.1f} m")
            P(f"    {'ax':<3}{'src':<7}{'sig':>7}{'rms/sig':>9}{'expErr':>8}{'tauHat/ana':>11}{'nearACF':>9}"
              f"{'PSDlo':>8}{'PSDmid':>8}{'PSDhi':>8}{'int/s2':>8}{'HFslope':>8}")
            info["axes"] = {}
            for ax in "uvw":
                x = d[ax][m]
                gust[(e, ax)] = x
                series[(label, e, ax)] = x
                kind = "u" if ax == "u" else "vw"
                tau = Ls[ax] / info["V_mean"]
                s = stats(x, dt, sg[ax], tau, kind, a.nseg)
                s["PSD_note"] = "analytic PSD NOT rescaled; ratios = Welch/analytic band integrals"
                rows = [("plugin", s)]
                if not a.no_mirror:
                    mir = mirror_stats(sg[ax], Ls[ax], info["V_mean"], dt, len(x), kind, a.nseg, rng)
                    rows.append(("mirror", mir))
                    s["mirror"] = mir
                for src, r in rows:
                    b = r["psd_ratio_bands"]
                    fm = lambda v: f"{v:8.3f}" if v is not None else "     n/a"
                    P(f"    {ax:<3}{src:<7}{sg[ax]:>7.3f}{r['rms_over_sigma']:>9.4f}{r['expected_rms_rel_err']:>8.4f}"
                      f"{r['tau_hat_over_analytic_window']:>11.3f}{r['near_acf_maxdev_1tau']:>9.4f}"
                      f"{fm(b['lo(<fc)'])}{fm(b['mid(fc-10fc)'])}{fm(b['hi(>10fc,<10Hz)'])}"
                      f"{r['psd_integral_over_sigma2']:>8.3f}{r['hf_slope']:>8.2f}")
                info["axes"][ax] = s
            case["entities"][str(e)] = info

        # cross-correlation between links (same axis) and between axes (same link)
        N = min(len(v) for v in gust.values())
        link_pairs, axis_pairs = {}, {}
        for i, e1 in enumerate(ents):
            for e2 in ents[i + 1:]:
                for ax in "uvw":
                    link_pairs[f"{e1}-{e2}:{ax}"] = corr(gust[(e1, ax)][:N], gust[(e2, ax)][:N])
        for e in ents:
            for p, q in (("u", "v"), ("u", "w"), ("v", "w")):
                axis_pairs[f"{e}:{p}{q}"] = corr(gust[(e, p)][:N], gust[(e, q)][:N])
        # sampling sd of a null correlation: 1/sqrt(N_eff), N_eff from slowest axis tau_int
        dt0 = case["entities"][str(ents[0])]["dt"]
        tau_slow = max(case["entities"][str(ents[0])]["axes"][ax]["tau_int_analytic_s"] for ax in "uvw")
        null_sd = 1 / math.sqrt(N * dt0 / (2 * tau_slow))
        case["cross"] = {"link_pairs": link_pairs, "axis_pairs_within_link": axis_pairs,
                         "null_sd_conservative": null_sd,
                         "max_abs_link_r": max(abs(v) for v in link_pairs.values()) if link_pairs else None,
                         "max_abs_axis_r": max(abs(v) for v in axis_pairs.values())}
        P(f"\n  cross-correlation (null sd ~ {null_sd:.4f}, conservative): max|r| between links = "
          f"{case['cross']['max_abs_link_r']}, max|r| between axes within link = {case['cross']['max_abs_axis_r']:.4f}")
        for k, v in link_pairs.items():
            P(f"    link {k}: r={v:+.4f}")
        out["cases"][label] = case

    # between-run correlation (different seeds, same entity ids)
    labels = [s.split(":")[0] for s in a.case]
    inter = {}
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            for (lab, e, ax), x in list(series.items()):
                if lab != l1 or (l2, e, ax) not in series:
                    continue
                y = series[(l2, e, ax)]
                n = min(len(x), len(y))
                inter[f"{l1}~{l2}:ent{e}:{ax}"] = corr(x[:n], y[:n])
    out["inter_run_correlation"] = inter
    if inter:
        P("\nBetween-run (different world/seed) correlation, same entity id & axis: "
          f"max|r|={max(abs(v) for v in inter.values()):.4f} over {len(inter)} pairs")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=2)
    if a.txt:
        open(a.txt, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
