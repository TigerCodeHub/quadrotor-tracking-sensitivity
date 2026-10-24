"""Sobol-style indices from COMPLETE 18-row Saltelli blocks only (no imputation).

With ~5% of rows missing at random, ~60% of base-sample blocks lose >=1 row. Median/
surrogate imputation of those rows contaminates a variance estimator (it spikes the seed
noise floor and inflates every ST). Instead we keep only base samples whose entire
(2D+2)-row block [A, AB_1..D, BA_1..D, B] is present, flatten those in order, and hand the
reduced but internally-complete Saltelli sequence to SALib. Each kept block is a valid
Saltelli unit, but because retention depends on every run in the block succeeding, the
retained blocks are NOT samples from the specified independent input distribution (selection
changes the input distribution, induces factor dependence, and partly selects on outcomes). The
result is a selected-complete-block sensitivity DIAGNOSTIC, not an unbiased estimate of the original
design's Sobol indices.

Usage: python saltelli_completeblock.py <design_dir> <results_dir> [--segments cruise armed]
"""
import argparse, csv, datetime, hashlib, json, math, os, sys
from pathlib import Path
import numpy as np

METRIC = os.environ.get("TDR_METRIC", "sob2")  # response column (sob2 | cross_track_rmse | ...)


def load(design_dir):
    m = json.loads((Path(design_dir) / "saltelli_problem.json").read_text())
    idx = {}
    with (Path(design_dir) / "saltelli_all.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            idx[r["run_id"]] = int(r["saltelli_index"])
    return m, idx


def load_results(path):
    out = {}
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            rid = r.get("run_id")
            if not rid or rid in out:
                continue
            try:
                v = float(r.get(METRIC) or "nan")
            except ValueError:
                continue
            if math.isfinite(v) and v > 0:
                out[rid] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir")
    ap.add_argument("results_dir")
    ap.add_argument("--segments", nargs="+", default=["cruise", "armed"])
    a = ap.parse_args()

    meta, index_map = load(a.design_dir)
    problem = meta["problem"]
    D = problem["num_vars"]
    names = problem["names"]
    reported = set(meta["reported_factors"])
    num_runs = meta["num_runs"]
    block = 2 * D + 2
    assert num_runs % block == 0, (num_runs, block)
    N = num_runs // block

    from SALib.analyze import sobol
    summary = []
    for seg in a.segments:
        # prefer the canonical analysis set (unique/valid/not-cooked) if materialized; else combined
        rp = Path(a.results_dir) / f"analysis_{seg}.csv"
        if not rp.exists():
            rp = Path(a.results_dir) / f"combined_{seg}.csv"
        if not rp.exists():
            print(f"[{seg}] no analysis_/combined_ {seg}.csv; skip"); continue
        print(f"[{seg}] input: {rp.name}")
        res = load_results(rp)
        y = np.full(num_runs, np.nan)
        for rid, i in index_map.items():
            if rid in res:
                y[i] = math.log(res[rid])
        blocks = y.reshape(N, block)
        complete = ~np.isnan(blocks).any(axis=1)
        nkeep = int(complete.sum())
        print(f"\n[{seg}] {int(np.isfinite(y).sum())}/{num_runs} rows present; "
              f"COMPLETE blocks {nkeep}/{N} ({100*nkeep/N:.1f}%) -> N'={nkeep}")
        if nkeep < 32:
            print("  too few complete blocks; skip"); continue
        yk = blocks[complete].reshape(-1)
        # seed + 1000 resamples -> reproducible CIs (audit #19/#20)
        si = sobol.analyze(problem, yk, calc_second_order=True, num_resamples=1000,
                           conf_level=0.95, seed=20261005, print_to_console=False)
        order = sorted(range(len(names)), key=lambda i: si["ST"][i], reverse=True)
        print("  factor              S1     S1c     ST     STc    ST-S1")
        for i in order:
            tag = "  (seed: stochastic input, not a noise floor)" if names[i] not in reported else ""
            print(f"    {names[i]:<16} {si['S1'][i]:+.3f}  {si['S1_conf'][i]:.3f}  "
                  f"{si['ST'][i]:+.3f}  {si['ST_conf'][i]:.3f}  {si['ST'][i]-si['S1'][i]:+.3f}{tag}")
        pairs = []
        s2, s2c = si["S2"], si["S2_conf"]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not math.isnan(s2[i, j]):
                    pairs.append((s2[i, j], s2c[i, j], names[i], names[j]))
        pairs.sort(reverse=True)
        print("  top second-order pairs (S_ij +/- conf):")
        for v, c, x, z in pairs[:6]:
            print(f"    {x} x {z}: {v:+.3f} +/- {c:.3f}")
        # write csv -- filename AND a column encode the metric so a cross_track_rmse run can
        # never silently overwrite or be mistaken for a sob2 run (audit: artifacts must encode
        # the metric/sample count/provenance). A sidecar .meta.json carries full provenance.
        out = Path(a.results_dir) / f"saltelli_cb_indices_{seg}_{METRIC}.csv"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["factor", "S1", "S1_conf", "ST", "ST_conf", "metric", "n_blocks"])
            for i in range(len(names)):
                w.writerow([names[i], f"{si['S1'][i]:.6f}", f"{si['S1_conf'][i]:.6f}",
                            f"{si['ST'][i]:.6f}", f"{si['ST_conf'][i]:.6f}", METRIC, nkeep])
        meta_out = out.with_suffix(".meta.json")
        try:
            src_sha = hashlib.sha256(rp.read_bytes()).hexdigest()
        except OSError:
            src_sha = None
        meta_out.write_text(json.dumps({
            "metric": METRIC, "segment": seg,
            "N_complete_blocks": nkeep, "N_total_blocks": N,
            "input_csv": str(rp), "input_sha256": src_sha,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, indent=2))
        summary.append({"segment": seg, "metric": METRIC,
                        "N_complete_blocks": nkeep, "N_total_blocks": N,
                        "S1": {names[i]: si["S1"][i] for i in range(len(names))},
                        "ST": {names[i]: si["ST"][i] for i in range(len(names))},
                        "seed_ST": si["ST"][names.index("seed_unit")] if "seed_unit" in names else None,
                        "top_pairs": [(x, z, round(float(v), 4)) for v, c, x, z in pairs[:6]]})
    (Path(a.results_dir) / f"saltelli_cb_summary_{METRIC}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote complete-block indices to {a.results_dir}")


if __name__ == "__main__":
    main()
