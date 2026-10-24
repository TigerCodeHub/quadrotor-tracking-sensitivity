"""Compute Sobol S1 / S_T (and second-order S_ij) indices from a flown Saltelli sweep.

Pairs with scripts/saltelli/make_saltelli_design.py. Reads the pinned problem sidecar
(saltelli_problem.json) so the factor order, N and second-order flag can never drift from
the sampling step, maps each flown run back to its saltelli_index, rebuilds the output
vector Y in EXACT sample order, and runs SALib.analyze.sobol.analyze.

Output metric: log(SOB2) by default (the campaigns model log(SOB2); indices are then
directly comparable to the v2/v4 regression coefficients). --transform raw uses SOB2.

GAPS: the Saltelli estimator needs a COMPLETE Y in order -- one missing row corrupts the
whole A/B/AB_i bookkeeping. Real flights carry a ~1-3% unflyable tail, so this script:
  * reports completeness per segment;
  * with --require-complete, aborts if anything is missing (use after a gap-fill pass);
  * otherwise imputes missing Y (median, or a cheap OLS surrogate on the physical factors
    via --impute surrogate) and LOUDLY flags it -- imputation biases indices toward 0, so
    treat imputed results as provisional and gap-fill to 100% before quoting numbers.

Usage:
  python3 scripts/saltelli/saltelli_analyze.py <design_dir> <results_dir> \
      [--segments cruise armed] [--transform log|raw] \
      [--impute median|surrogate] [--require-complete] [--out <dir>]

  <design_dir>   holds saltelli_problem.json + saltelli_all.csv (from the design script)
  <results_dir>  holds combined_<segment>.csv (from scripts/infra/tdr.sh combine)
"""
import argparse
import csv
import json
import math
from pathlib import Path

import os

import numpy as np

METRIC = os.environ.get("TDR_METRIC", "sob2")  # response column (sob2 | cross_track_rmse | ...)


def load_problem(design_dir):
    meta = json.loads((Path(design_dir) / "saltelli_problem.json").read_text(encoding="utf-8"))
    return meta


def load_index_map(design_dir):
    """run_id -> saltelli_index, from the design."""
    mapping = {}
    with (Path(design_dir) / "saltelli_all.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[row["run_id"]] = int(row["saltelli_index"])
    return mapping


def load_results(path):
    """run_id -> sob2 (first valid, positive). combined_<seg>.csv from tdr.sh combine."""
    out = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rid = row.get("run_id")
            if not rid or rid in out:
                continue
            try:
                sob2 = float(row.get(METRIC) or "nan")
            except ValueError:
                continue
            if math.isfinite(sob2) and sob2 > 0:
                out[rid] = sob2
    return out


def build_Y(index_map, results, num_runs, transform):
    """Y in saltelli_index order; NaN where a run is missing."""
    y = np.full(num_runs, np.nan)
    for rid, idx in index_map.items():
        if rid in results:
            v = results[rid]
            y[idx] = math.log(v) if transform == "log" else v
    return y


def impute(y, samples, reported_k, mode):
    missing = np.isnan(y)
    n_missing = int(missing.sum())
    if n_missing == 0:
        return y, 0
    if mode == "surrogate":
        # cheap OLS surrogate on the physical factors (first reported_k columns) + squares,
        # fit on the observed rows, used only to fill gaps.
        obs = ~missing
        X = samples[:, :reported_k]
        design = np.column_stack([np.ones(len(y)), X, X ** 2])
        coef, *_ = np.linalg.lstsq(design[obs], y[obs], rcond=None)
        y = y.copy()
        y[missing] = design[missing] @ coef
    else:  # median
        y = y.copy()
        y[missing] = np.nanmedian(y)
    return y, n_missing


def analyze_segment(meta, samples, index_map, results_path, transform, impute_mode,
                    require_complete, out_dir, segment, seed=20261005, num_resamples=1000):
    problem = meta["problem"]
    num_runs = meta["num_runs"]
    second_order = meta["calc_second_order"]
    names = problem["names"]
    reported = meta["reported_factors"]
    reported_k = len(reported)

    results = load_results(results_path)
    y = build_Y(index_map, results, num_runs, transform)
    n_have = int(np.isfinite(y).sum())
    frac = n_have / num_runs
    print(f"\n[{segment}] {n_have}/{num_runs} runs present ({100 * frac:.2f}%)")

    n_missing = num_runs - n_have
    if n_missing:
        if require_complete:
            raise SystemExit(
                f"[{segment}] {n_missing} runs missing and --require-complete set; "
                f"gap-fill then re-run."
            )
        y, filled = impute(y, samples, reported_k, impute_mode)
        print(f"  !! IMPUTED {filled} missing runs via {impute_mode}; indices are "
              f"PROVISIONAL (imputation biases S_i/S_T toward 0). Gap-fill to 100% "
              f"before quoting.")

    from SALib.analyze import sobol
    # seed + more bootstrap resamples so the confidence intervals are REPRODUCIBLE and
    # tighter (audit #20: SALib defaults to 100 resamples and no seed -> CIs drift run
    # to run). conf_level stays 95%.
    si = sobol.analyze(problem, y, calc_second_order=second_order,
                       num_resamples=num_resamples, conf_level=0.95, seed=seed,
                       print_to_console=False)

    # first / total order table (physical factors + the seed noise-floor diagnostic)
    idx_out = out_dir / f"saltelli_indices_{segment}.csv"
    with idx_out.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["factor", "S1", "S1_conf", "ST", "ST_conf", "interaction_ST_minus_S1"])
        for i, name in enumerate(names):
            w.writerow([name, f"{si['S1'][i]:.6f}", f"{si['S1_conf'][i]:.6f}",
                        f"{si['ST'][i]:.6f}", f"{si['ST_conf'][i]:.6f}",
                        f"{si['ST'][i] - si['S1'][i]:.6f}"])

    order = sorted(range(len(names)), key=lambda i: si["ST"][i], reverse=True)
    print("  factor              S1      ST     ST-S1")
    for i in order:
        tag = "  (seed: noise floor)" if names[i] not in reported else ""
        print(f"    {names[i]:<16} {si['S1'][i]:+.3f}  {si['ST'][i]:+.3f}  "
              f"{si['ST'][i] - si['S1'][i]:+.3f}{tag}")

    # second-order pairs
    top_pairs = []
    if second_order:
        s2 = si["S2"]
        s2c = si["S2_conf"]
        s2_out = out_dir / f"saltelli_s2_{segment}.csv"
        with s2_out.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["factor_i", "factor_j", "S2", "S2_conf"])
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    val = s2[i, j]
                    if not math.isnan(val):
                        w.writerow([names[i], names[j], f"{val:.6f}", f"{s2c[i, j]:.6f}"])
                        top_pairs.append((val, names[i], names[j]))
        top_pairs.sort(reverse=True)
        print("  top second-order pairs (S_ij):")
        for val, a, b in top_pairs[:6]:
            print(f"    {a} x {b}: {val:+.3f}")

    return {
        "segment": segment,
        "completeness": frac,
        "n_runs": num_runs,
        "S1": {names[i]: si["S1"][i] for i in range(len(names))},
        "ST": {names[i]: si["ST"][i] for i in range(len(names))},
        "top_pairs": [(a, b, v) for v, a, b in top_pairs[:6]],
        "seed_noise_floor_ST": si["ST"][names.index("seed_unit")]
        if "seed_unit" in names else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design_dir")
    ap.add_argument("results_dir")
    ap.add_argument("--segments", nargs="+", default=["cruise", "armed"])
    ap.add_argument("--transform", choices=["log", "raw"], default="log")
    ap.add_argument("--impute", choices=["median", "surrogate"], default="median")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--seed", type=int, default=20261005,
                    help="bootstrap seed for reproducible CIs (audit #20)")
    ap.add_argument("--num-resamples", type=int, default=1000,
                    help="SALib bootstrap resamples (default 1000; SALib default is 100)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = load_problem(args.design_dir)
    index_map = load_index_map(args.design_dir)

    # re-draw the exact sample matrix (only needed for the surrogate imputation features);
    # deterministic from the pinned seed in the sidecar.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from make_saltelli_design import _saltelli_sample  # type: ignore
    samples = _saltelli_sample(meta["problem"], meta["N"], meta["calc_second_order"])
    assert len(samples) == meta["num_runs"], "sample matrix size drift vs sidecar"

    out_dir = Path(args.out) if args.out else Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for seg in args.segments:
        results_path = Path(args.results_dir) / f"combined_{seg}.csv"
        if not results_path.exists():
            print(f"[{seg}] no combined_{seg}.csv in {args.results_dir}; skipping")
            continue
        summary.append(analyze_segment(
            meta, np.asarray(samples), index_map, results_path, args.transform,
            args.impute, args.require_complete, out_dir, seg,
            seed=args.seed, num_resamples=args.num_resamples))

    (out_dir / "saltelli_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote indices + summary to {out_dir}")


if __name__ == "__main__":
    main()
