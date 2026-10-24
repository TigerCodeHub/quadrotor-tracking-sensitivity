"""Generate paper figures from the committed re-analysis outputs.

Reads the refit_robust JSONs + the metric-tagged Saltelli complete-block CSVs and writes PNGs
to figures/ (override with TDR_FIG_OUT). Run on a host with matplotlib (py312) AND with the cross-track analysis
outputs present.

IMPORTANT (audit): this script FAILS LOUDLY if a required input for the requested metric is
missing. It never silently skips a campaign, zero-fills a missing factor, or falls back to a
different-metric CSV -- those behaviours previously let a SOB2 artifact masquerade as the
cross-track figure. Regenerate the inputs first:

    # per campaign: refit_robust -> refit_ct_<seg>.json  (responses keyed by metric)
    TDR_METRIC=cross_track_rmse python3 scripts/analysis/refit_robust.py \
        results/campaign_v2/combined_cruise.csv v2 --json results/campaign_v2/refit_ct_cruise.json
    # Saltelli: metric-tagged complete-block indices
    TDR_METRIC=cross_track_rmse python3 scripts/saltelli/saltelli_completeblock.py \
        generated/designs/saltelli results/campaign_saltelli_merged --segments cruise
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200, "savefig.bbox": "tight"})
# Okabe-Ito colorblind-safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "grey": "#999999"}
OUT = os.environ.get("TDR_FIG_OUT", "figures")
os.makedirs(OUT, exist_ok=True)
LAB = {"turbulence_rms": "turbulence", "mean_wind": "mean wind", "thrust_to_weight": "T/W",
       "log_payload": "log(1+payload)", "abs_com_y_frac": "|CoM$_y$|/wheelbase", "inertia_scale": "inertia"}
FACS = ["turbulence_rms", "mean_wind", "thrust_to_weight", "log_payload", "abs_com_y_frac", "inertia_scale"]

# The column historically named `cross_track_rmse` is the paper's 3-D setpoint-path RMSE
# (3-D SPE): 3-D distance to the commanded setpoint polyline. See DATA_MANIFEST.md.
PRIMARY = "cross_track_rmse"
METRIC_AXIS = "log 3-D SPE"
METRIC_NAME = "3-D setpoint-path RMSE"


def die(msg):
    sys.exit(f"[make_paper_figures] {msg}")


def require(path, hint=""):
    if not os.path.exists(path):
        die(f"MISSING required input: {path}\n  {hint}")
    return path


def load_refit(p):
    j = json.load(open(p))
    resp = j.get("responses", {})
    if PRIMARY not in resp:
        die(f"{p} has no '{PRIMARY}' response (found: {sorted(resp)}). "
            f"Re-run refit_robust with --responses including {PRIMARY}.")
    return {f["factor"]: f for f in resp[PRIMARY]["factors"]}, resp[PRIMARY]


def getf(d, f, where):
    if f not in d:
        die(f"factor '{f}' absent from {where}; refusing to plot a zero. "
            f"Re-run the refit so every factor is present.")
    return d[f]


CAMPS = [("v2 (white, straight)", "results/campaign_v2/refit_ct_cruise.json", C["grey"]),
         ("v3-D (Dryden, straight)", "results/campaign_v3d/refit_ct_cruise.json", C["blue"]),
         ("v4 (Dryden, routes)", "results/campaign_v4/refit_ct_cruise.json", C["orange"])]
refit = {}
for name, p, col in CAMPS:
    require(p, "run refit_robust.py for this campaign (see module docstring).")
    refit[name] = (load_refit(p)[0], col)

# --- Fig 1: IQR effect per factor across campaigns (importance, not raw |b|) ---
present = [(n, d, c) for n, (d, c) in refit.items()]
x = np.arange(len(FACS)); w = 0.8 / max(len(present), 1)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for i, (name, d, col) in enumerate(present):
    vals = [abs(getf(d, f, name)["iqr_effect"]) for f in FACS]
    ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=col)
ax.set_xticks(x); ax.set_xticklabels([LAB[f] for f in FACS], rotation=20, ha="right")
ax.set_ylabel(f"|effect across IQR|  ({METRIC_AXIS})"); ax.legend(fontsize=8, frameon=False)
ax.set_title(f"Factor importance by interquartile effect (cruise, {METRIC_NAME})")
fig.savefig(f"{OUT}/fig_importance.png"); plt.close(fig)

# --- Fig 2: drop-one delta-R^2 per factor (v4) ---
d = refit["v4 (Dryden, routes)"][0]
vals = [getf(d, f, "v4")["drop_one_dr2"] for f in FACS]
fig, ax = plt.subplots(figsize=(5.5, 3.0))
ax.bar([LAB[f] for f in FACS], vals, color=C["green"])
ax.set_ylabel(r"drop-one $\Delta R^2$"); ax.set_title(f"Variance explained per factor (v4 cruise, {METRIC_NAME})")
ax.tick_params(axis="x", rotation=20)
fig.savefig(f"{OUT}/fig_drop_one_dr2.png"); plt.close(fig)

# --- Fig 3: |IQR effect| per factor across experiments (log axis, HC3 95% CIs); full-width single panel.
exps = [(n.split(" (")[0], d, col) for n, (d, col) in refit.items()]
Z = 1.96
FLOOR = 3e-4  # bottom of the log axis; CIs whose lower bound is <= 0 are drawn down to it with a marker
fig, axa = plt.subplots(figsize=(7.0, 3.0))
xa = np.arange(len(FACS)); wa = 0.8 / len(exps)
numbers = ["|IQR effect| (log units); HC3 95% CI = value +- 1.96*|iqr/t|; * = CI includes 0 (down-triangle; lower whisker drawn to axis floor)"]
for i, (nm, d, col) in enumerate(exps):
    for k, f in enumerate(FACS):
        e = getf(d, f, nm)
        v = abs(e["iqr_effect"]); h = 1.96 * abs(e["iqr_effect"] / e["t"]) if e["t"] else 0.0
        lo, hi = v - h, v + h
        xc = xa[k] + i * wa - 0.4 + wa / 2
        axa.bar(xc, v - FLOOR, wa, bottom=FLOOR, color=col, label=nm if k == 0 else None)
        axa.plot([xc, xc], [max(lo, FLOOR), hi], color="k", lw=0.8)
        axa.plot([xc - wa * 0.2, xc + wa * 0.2], [hi, hi], color="k", lw=0.8)
        if lo > FLOOR:
            axa.plot([xc - wa * 0.2, xc + wa * 0.2], [lo, lo], color="k", lw=0.8)
        elif lo <= 0:  # CI genuinely includes zero -> down-triangle at the axis floor
            axa.plot([xc], [FLOOR * 1.12], marker="v", color="k", ms=2.8, ls="none")
        # else: 0 < lo <= FLOOR: lower bound positive but below the axis floor; whisker simply clipped at the floor
        numbers.append(f"  {nm:<5} {f:<18} |IQR|={v:.4f}  CI=[{max(lo,0):.4f}, {hi:.4f}]{'  *' if lo <= 0 else ''}")
axa.set_yscale("log"); axa.set_ylim(FLOOR, 3.5)
axa.set_xticks(xa); axa.set_xticklabels([LAB[f] for f in FACS], fontsize=7.3)
axa.set_ylabel(f"|effect across IQR| ({METRIC_AXIS}), log scale", fontsize=8)
axa.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_wind_transfer.png"); plt.close(fig)
numbers.append("airframe-FE coefficient beta (log 3-D SPE per unit) with HC3 95% CI = beta +- 1.96*|beta/t| (not plotted)")
for f in FACS:
    for nm, d, col in exps:
        e = getf(d, f, nm)
        h = 1.96 * abs(e["beta"] / e["t"]) if e["t"] else 0.0
        numbers.append(f"  {f:<18} {nm:<5} beta={e['beta']:+.4f}  95% CI=[{e['beta']-h:+.4f}, {e['beta']+h:+.4f}]")
open("results/fig_wind_transfer_numbers.txt", "w").write("\n".join(numbers) + "\n")

# --- Fig 4: metric invariance (coef across response metrics, v2) ---
jp = require("results/campaign_v2/refit_ct_cruise.json")
j = json.load(open(jp)); resp = j["responses"]
mets = [m for m in ("cross_track_rmse", "rms_error", "max_error") if m in resp]
show = ["turbulence_rms", "mean_wind", "log_payload"]
x = np.arange(len(show)); w = 0.8 / len(mets)
fig, ax = plt.subplots(figsize=(5.5, 3.0))
for i, m in enumerate(mets):
    fd = {f["factor"]: f for f in resp[m]["factors"]}
    ax.bar(x + i * w - 0.4 + w / 2, [getf(fd, f, f"{m} in v2")["beta"] for f in show], w, label=m)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x)
ax.set_xticklabels([LAB[f] for f in show]); ax.set_ylabel("coefficient (log-metric)")
ax.legend(fontsize=8, frameon=False); ax.set_title(f"Metric robustness (v2 cruise, {METRIC_NAME})")
fig.savefig(f"{OUT}/fig_metric_invariance.png"); plt.close(fig)

# --- Fig 5: Sobol indices for v5 (Saltelli), cruise, primary metric.
# S1 = given-data first-order index (equal-count binning, K=50, bias-corrected, ALL valid runs) with percentile bootstrap CIs over runs;
# ST = pairwise-complete Jansen total effect with percentile bootstrap CIs over base blocks. Produced by
# scripts/saltelli/saltelli_crosscheck_given_data.py. Whiskers are drawn for S1 only where S1 >= 0.005 (the near-zero factors are
# estimated as ~0 and the bootstrap interval of a plug-in binned estimator near zero is not meaningful).
sp = os.environ.get("TDR_V5_INDICES", "results/campaign_saltelli_merged/saltelli_v5_given_data_indices_final.csv")
if not os.path.exists(sp):
    die(f"missing {sp}; run scripts/saltelli/saltelli_crosscheck_given_data.py first")
rows = list(csv.DictReader(open(sp)))
SOBOL_LAB = {"mean_wind": "mean\nwind", "turbulence_rms": "turbulence", "seed_unit": "gust\nseed", "thrust_to_weight": "T/W",
             "wind_direction_deg": "wind\ndirection", "com_y_frac": "CoM\noffset", "payload_fraction": "payload\nfraction",
             "inertia_scale": "inertia"}
rows.sort(key=lambda r: -float(r["ST_pairwise_jansen"]))
nm = [r["factor"] for r in rows]
s1 = [float(r["S1_biascorrected_K50"]) for r in rows]
st = [float(r["ST_pairwise_jansen"]) for r in rows]
s1lo = [float(r["S1_lo95"]) for r in rows]; s1hi = [float(r["S1_hi95"]) for r in rows]
stlo = [float(r["ST_lo95"]) for r in rows]; sthi = [float(r["ST_hi95"]) for r in rows]
x = np.arange(len(nm))
fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.bar(x - 0.2, s1, 0.4, label="$S_1$ (first-order)", color=C["blue"])
ax.bar(x + 0.2, st, 0.4, label="$S_T$ (total)", color=C["orange"])
for k in range(len(nm)):
    ax.errorbar([x[k] + 0.2], [st[k]], yerr=[[st[k] - stlo[k]], [sthi[k] - st[k]]], color="k", lw=1.0, capsize=2, fmt="none")
    if s1[k] >= 0.005:
        ax.errorbar([x[k] - 0.2], [s1[k]], yerr=[[max(s1[k] - s1lo[k], 0)], [max(s1hi[k] - s1[k], 0)]], color="k", lw=1.0, capsize=2, fmt="none")
ax.set_xticks(x); ax.set_xticklabels([SOBOL_LAB.get(n, n.replace("_", " ")) for n in nm], fontsize=8)
floor = min(st)  # common floor = S_T of the weakest factor (inertia), thin dashed line spanning the plot
if os.environ.get("TDR_SOBOL_FLOOR", "0") != "0":  # paper figure: no floor line
    ax.axhline(floor, color="k", lw=0.7, ls="--", label="common floor (inertia $S_T$)", zorder=1)
ax.set_ylabel("Sobol index"); ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.axhline(0.0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_sobol.png"); plt.close(fig)

print("wrote figures:", sorted(os.listdir(OUT)))
