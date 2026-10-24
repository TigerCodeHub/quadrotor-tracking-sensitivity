"""Build the v3-D design: a Dryden-turbulence arm paired against already-flown v2 runs.

EXTENSIONS item 8 (turbulence realism). Every v3-D row is a verbatim copy of a v2 design
row with exactly ONE field changed -- turb_model: white -> dryden -- keeping the SAME model
and the SAME wind_seed, plan (straight), controller (fixed), mean wind, direction and
turbulence_rms. So each dryden run pairs with its v2 white control on the identical config,
and the within-pair delta  d = ln(sob2_dryden) - ln(sob2_white)  isolates the effect of the
turbulence SPECTRUM alone (white-noise WindEffects vs a physically-correct Dryden spectrum),
matched on intensity, axes and mean. The control arm is already flown, so it costs nothing.

The runner (sim.py _case_world) routes turb_model==dryden to: WindEffects for the mean +
deterministic sinusoid (all noise stddev 0) PLUS the DrydenWind system plugin for the colored
gusts at the white arm's per-axis intensities (sigma_u=rms, sigma_v=0, sigma_w=0.25 rms) in a
fixed frame (u along wind heading, w along world up). Requires TDR_GZ_PLUGIN_PATH to point at
the DrydenWind build dir so the .so is on GZ_SIM_SYSTEM_PLUGIN_PATH.

Same strict control eligibility as make_v3_design.py (winning v2 flight used cruise 6x AND
takeoff 1x), and the same systematic every-k-th sample_index selection to preserve Sobol
low-discrepancy coverage.

Usage:
  python scripts/v3/make_v3d_design.py <n_total> <out_dir> [--canary N]
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
V2_DESIGN = REPO / "generated/designs/sobol_v2_2e14.csv"
V2_CONTROL = REPO / "results/campaign_v2/combined_cruise.csv"


def eligible_models():
    """v2 models whose winning flight used the canonical speed profile."""
    ok = set()
    with V2_CONTROL.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("from_failed_attempt") == "1":
                continue
            if (row.get("takeoff_sim_speed_factor") or "").startswith("2"):
                continue
            cruise = (row.get("cruise_sim_speed_factor") or "").strip()
            if cruise and not cruise.startswith("6"):
                continue
            try:
                if float(row.get("sob2") or 0) <= 0:
                    continue
            except ValueError:
                continue
            name = row.get("model_name")
            if name:
                ok.add(name)
    return ok


def main():
    n_total = int(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = eligible_models()
    print("eligible v2 control models: %d" % len(ok))

    with V2_DESIGN.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r["model_name"] in ok and (r.get("repeat") or "1") == "1"]
    print("v2 design rows with an eligible control: %d" % len(rows))

    by_airframe = defaultdict(list)
    for row in rows:
        by_airframe[row["airframe_id"]].append(row)
    for airframe in by_airframe:
        by_airframe[airframe].sort(key=lambda r: int(r["sample_index"]))

    per_airframe = n_total // len(by_airframe)
    chosen = []
    for airframe, group in sorted(by_airframe.items()):
        step = max(1, len(group) // per_airframe)
        picked = group[::step][:per_airframe]
        chosen.extend(picked)
        print("  %-12s pool=%-6d step=%-3d picked=%d" % (airframe, len(group), step, len(picked)))
    print("selected configs: %d" % len(chosen))

    extra = ["turb_model", "paired_with_run_id"]
    out_fields = fields + [c for c in extra if c not in fields]

    def write(path, source_rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=out_fields)
            writer.writeheader()
            for source in source_rows:
                row = dict(source)
                row["paired_with_run_id"] = source["run_id"]
                row["turb_model"] = "dryden"
                # New run_id (runner skips a row whose ULog already exists); wind_seed is
                # copied verbatim, which is what makes the white/dryden pair valid.
                row["run_id"] = "%s_dw" % source["model_name"]
                writer.writerow({k: row.get(k, "") for k in out_fields})
        print("wrote %s (%d rows)" % (path, len(source_rows)))

    write(out_dir / "v3_d.csv", chosen)

    # run_id uniqueness guard
    ids = ["%s_dw" % s["model_name"] for s in chosen]
    assert len(ids) == len(set(ids)), "duplicate run_ids -- model_name not unique in selection"

    if "--canary" in sys.argv:
        n = int(sys.argv[sys.argv.index("--canary") + 1])
        # spread across airframes
        step = max(1, len(chosen) // n)
        write(out_dir / "v3_d_canary.csv", chosen[::step][:n])


if __name__ == "__main__":
    main()
