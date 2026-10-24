"""Extended numerical validation of the Dryden-form gust generator (audit pass 3, item 5).

Strengthens scripts/analysis/validate_dryden.py (Python mirror of DrydenWind.cc) by:
  * LONG Welch segments (default 2**17 samples = ~524 s at 4 ms; resolution ~0.002 Hz) so the
    low-frequency Dryden corner and the integral scale are resolved for most axes;
  * comparing the ABSOLUTE analytic Dryden PSD (no rescaling to the empirical energy), in three
    bands relative to the corner frequency f_c = 1/(2 pi tau): below f_c (low-frequency / integral
    scale), f_c..10 f_c, and above 10 f_c to Nyquist/8 (high-frequency tail);
  * LOW-altitude cases (campaign phases fly well below the earlier 60/100 m points);
  * MANY independent seeds per case, reporting mean and spread (not a single realization).

The generator is the same exact discretization as the plugin (OU for u; 2-state Euler-Maruyama
Dryden filter for v, w), but vectorized with scipy.signal.lfilter (the Euler recursion is linear,
so it is exactly the same recursion as the loop in validate_dryden.py; tested against it for a
short record with --selftest).

Analytic one-sided PSDs (Hz), variance sigma^2, tau = L/V, w = 2 pi f:
    u (first order):   S(f) = 4 tau sigma^2 / (1 + (w tau)^2)
    v, w (second):     S(f) = 2 tau sigma^2 (1 + 3 (w tau)^2) / (1 + (w tau)^2)^2

Usage: python3 scripts/analysis/validate_dryden_extended.py [--seeds 5] [--steps 4000000]
                                                            [--json out.json]
"""
import argparse
import json
import math
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from validate_dryden import (mil_scales_and_sigma, run_u_ou, run_vw_2state,  # noqa: E402
                             empirical_acf)


def gen_u(sigma, L, V, dt, n, rng):
    tau = L / V
    phi = math.exp(-dt / tau)
    q = sigma * math.sqrt(max(0.0, 1.0 - phi * phi))
    return lfilter([q], [1.0, -phi], rng.standard_normal(n)), tau


def gen_vw(sigma, L, V, dt, n, rng):
    """Euler-Maruyama 2-state Dryden filter (nsub=1), vectorized via its z-domain recursion."""
    tau = L / V
    a, b = 1.0 / tau ** 2, 2.0 / tau
    g = sigma * math.sqrt(L / V) / tau ** 2
    if dt / tau > 0.05:
        raise ValueError("sub-stepping case (dt/tau > 0.05) not vectorized")
    h = dt
    # (z-1)(z-1+h b) x1 + h^2 a x1 = h sqrt(h) eta   =>  x1[n+2] = (2-hb) x1[n+1] - (1-hb+h^2 a) x1[n] + h^1.5 eta
    den = [1.0, -(2.0 - h * b), (1.0 - h * b + h * h * a)]
    x1 = lfilter([h ** 1.5], den, rng.standard_normal(n + 1))
    x2 = (x1[1:] - x1[:-1]) / h
    return g * x1[:-1] + g * math.sqrt(3.0) * tau * x2, tau


def psd_u(f, tau, sigma):
    w = 2 * np.pi * f
    return 4 * tau * sigma ** 2 / (1 + (w * tau) ** 2)


def psd_vw(f, tau, sigma):
    w = 2 * np.pi * f
    return 2 * tau * sigma ** 2 * (1 + 3 * (w * tau) ** 2) / (1 + (w * tau) ** 2) ** 2


def welch(x, dt, nseg):
    x = x - x.mean()
    win = np.hanning(nseg)
    wnorm = (win ** 2).sum() / dt
    acc = None
    cnt = 0
    for s in range(0, len(x) - nseg + 1, nseg // 2):
        X = np.fft.rfft(x[s:s + nseg] * win)
        p = (np.abs(X) ** 2) / wnorm
        acc = p if acc is None else acc + p
        cnt += 1
    P = acc / cnt
    P[1:-1] *= 2.0
    return np.fft.rfftfreq(nseg, dt), P, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--dt", type=float, default=0.004)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--nseg", type=int, default=2 ** 17)
    ap.add_argument("--seed0", type=int, default=20261011)
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true", help="compare vectorized vs loop generator (short)")
    args = ap.parse_args()

    if args.selftest:
        for gen, loop, nm, args_ in ((gen_u, run_u_ou, "u", (0.8, 30.0, 5.0)),
                                      (gen_vw, run_vw_2state, "vw", (0.8, 30.0, 5.0))):
            a, _ = gen(*args_, args.dt, 400_000, np.random.default_rng(1))
            b, _ = loop(*args_, args.dt, 400_000, np.random.default_rng(2))
            print(f"selftest {nm}: vec std={a.std():.4f} loop std={b.std():.4f} target={args_[0]}")
        return

    # (V m/s, altitude m, w20 m/s): earlier 4 points + low-altitude cases
    CASES = [(5.0, 60.0, 5.0), (10.0, 60.0, 7.0), (15.0, 100.0, 7.0), (25.0, 100.0, 9.0),
             (5.0, 5.0, 5.0), (5.0, 10.0, 5.0), (10.0, 20.0, 7.0), (15.0, 30.0, 7.0)]
    T = args.steps * args.dt
    f_res = 1.0 / (args.nseg * args.dt)
    print("=" * 110)
    print(f"EXTENDED Dryden validation  steps={args.steps:,} ({T:.0f} s)  seeds={args.seeds}  "
          f"Welch nseg={args.nseg} (f_res={f_res:.4f} Hz)  ABSOLUTE PSD, no rescaling")
    print("=" * 110)
    print(f"{'V':>4}{'h':>5}{'axis':>5}{'tau':>7}{'rms/tgt':>10}{'tauhat/expected':>16}"
          f"{'PSD dex  low':>14}{'mid':>8}{'high':>8}{'HFslope':>9}  resolved")
    results = []
    for ci, (V, h_m, w20) in enumerate(CASES):
        h_ft = max(3.0, h_m * 3.28084)
        (Lu, Lv, Lw), (su, sv, sw) = mil_scales_and_sigma(h_ft, w20)
        axes = [("u", su, Lu, gen_u, psd_u, 1.0), ("v", sv, Lv, gen_vw, psd_vw, 0.5),
                ("w", sw, Lw, gen_vw, psd_vw, 0.5)]
        case = {"V": V, "h_m": h_m, "w20": w20, "axes": {}}
        for ai, (nm, sig, L, gen, psdf, tif) in enumerate(axes):
            tau = L / V
            tau_int = tif * tau
            f_c = 1.0 / (2 * np.pi * tau)
            rows = []
            for s in range(args.seeds):
                rng = np.random.default_rng(args.seed0 + 1000 * ci + 100 * ai + s)
                x, _ = gen(sig, L, V, args.dt, args.steps, rng)
                rms_ratio = float(x.std() / sig)
                fl = int(round(min(3 * tau, 0.25 * T) / args.dt))
                emp = empirical_acf(x, fl)
                # the ACF integral is truncated at X=3 tau, so its expectation is NOT tau_int:
                #   u  : int_0^X e^-x dx = 1 - e^-X ;  v,w: 1/2 + e^-X (X-1)/2  (X = window/tau)
                X = fl * args.dt / tau
                trunc = (1 - math.exp(-X)) if nm == 'u' else (0.5 + math.exp(-X) * (X - 1) / 2) / 0.5
                tau_hat = float(np.trapezoid(emp, dx=args.dt)) / (tau_int * trunc)
                f, P, nseg_cnt = welch(x, args.dt, args.nseg)
                S = psdf(f, tau, sig)
                good = (f > 0) & (P > 0)
                bands = {"low": (3 * f_res, f_c), "mid": (f_c, 10 * f_c), "high": (10 * f_c, 1 / (args.dt * 8))}
                dex = {}
                for bn, (lo, hi) in bands.items():
                    m = good & (f >= lo) & (f < hi)
                    dex[bn] = float(np.mean(np.log10(P[m] / S[m]))) if m.sum() > 3 else float("nan")
                hf = good & (f > 0.5) & (f < 20)
                slope = float(np.polyfit(np.log10(f[hf]), np.log10(P[hf]), 1)[0])
                rows.append((rms_ratio, tau_hat, dex["low"], dex["mid"], dex["high"], slope))
            a = np.array(rows)
            mean, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0, ddof=1)
            resolved = f_c > 3 * f_res
            case["axes"][nm] = {"tau_s": tau, "tau_int_s": tau_int, "f_c_hz": f_c, "low_band_resolved": bool(resolved),
                                "rms_over_target": [float(mean[0]), float(sd[0])],
                                "tau_hat_over_tau_int": [float(mean[1]), float(sd[1])],
                                "psd_log10_ratio_low": [float(mean[2]), float(sd[2])],
                                "psd_log10_ratio_mid": [float(mean[3]), float(sd[3])],
                                "psd_log10_ratio_high": [float(mean[4]), float(sd[4])],
                                "hf_slope": [float(mean[5]), float(sd[5])]}
            print(f"{V:>4.0f}{h_m:>5.0f}{nm:>5}{tau:>7.2f}"
                  f"{mean[0]:>7.3f}±{sd[0]:.3f}{mean[1]:>9.3f}±{sd[1]:.3f}"
                  f"{mean[2]:>9.3f}±{sd[2]:.3f}{mean[3]:>8.3f}{mean[4]:>8.3f}{mean[5]:>9.2f}  "
                  f"{'yes' if resolved else 'NO (f_c < 3 f_res)'}")
        results.append(case)
    print("\nPSD dex = mean log10(empirical/analytic) per band (0 = exact; the analytic curve is NOT rescaled).")
    print("low = 3 f_res..f_c (needs f_c > 3 f_res to exist), mid = f_c..10 f_c, high = 10 f_c..Nyquist/8.")
    if args.json:
        json.dump({"params": vars(args), "cases": results}, open(args.json, "w"), indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
