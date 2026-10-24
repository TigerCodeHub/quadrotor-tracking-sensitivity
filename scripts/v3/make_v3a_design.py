"""Build the v3-A autotune design: does per-config retuning remove the mass effect?

EXTENSIONS item 1. The original a1/a2 arms (formula gain-scaling: controller_mode retuned/
matched) were abandoned -- they detuned the vertical loop and caused 40-60% takeoff/route
failures. This replaces them with PX4's in-flight autotune: each config is first tuned in
calm air (a separate tune flight, see autotune_batch.sh, which writes {model}_autotune.json),
then flies the SAME scored straight mission as its v2 control but with controller_mode=autotune
(loads that JSON at sim.py:784). The within-config paired delta d = log(sob2_autotune) -
log(sob2_v2_fixed) answers: does retuning the attitude/rate loops reduce tracking error, and
does it remove the log_mass dependence? (Caveat: autotune tunes attitude/rate loops only, not
the MPC position loop, so a surviving mass effect is itself a valid result.)

Same straight plan + wind_seed as the v2 control, so ONLY the controller differs -> clean pair.

  usage: make_v3a_design.py <n_total> <out_csv>
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_DESIGN = REPO / "generated/designs/sobol_v2_2e14.csv"
V2_CONTROL = REPO / "results/campaign_v2/combined_cruise.csv"


def eligible_models():
    ok = set()
    for row in csv.DictReader(V2_CONTROL.open(newline="", encoding="utf-8")):
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
        if row.get("model_name"):
            ok.add(row["model_name"])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_total", type=int)
    ap.add_argument("out_csv")
    args = ap.parse_args()

    ok = eligible_models()
    print("eligible v2 control models: %d" % len(ok))
    with V2_DESIGN.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r["model_name"] in ok and (r.get("repeat") or "1") == "1"]

    by_af = defaultdict(list)
    for r in rows:
        by_af[r["airframe_id"]].append(r)
    for af in by_af:
        by_af[af].sort(key=lambda r: int(r["sample_index"]))

    per_af = args.n_total // len(by_af)
    chosen = []
    for af, group in sorted(by_af.items()):
        step = max(1, len(group) // per_af)
        picked = group[::step][:per_af]
        chosen.extend(picked)
        print("  %-12s pool=%-6d step=%-3d picked=%d" % (af, len(group), step, len(picked)))
    print("selected configs: %d" % len(chosen))

    extra = ["course_heading_deg", "maneuver", "v3_block", "paired_with_run_id"]
    out_fields = fields + [c for c in extra if c not in fields]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=out_fields)
        w.writeheader()
        for source in chosen:
            row = dict(source)
            row["paired_with_run_id"] = source["run_id"]
            row["v3_block"] = "a"
            row["course_heading_deg"] = 0.0
            row["maneuver"] = "straight"
            row["controller_mode"] = "autotune"   # loads {model}_autotune.json (from tune flight)
            row["run_id"] = "%s_at" % source["model_name"]
            w.writerow({k: row.get(k, "") for k in out_fields})
    print("wrote %s (%d rows, controller_mode=autotune)" % (out_path, len(chosen)))
    print("NOTE: each row needs generated/sobol_v2/controllers/<model>_autotune.json first")
    print("      (produce with autotune_batch.sh over THIS csv), then fly the scored mission.")


if __name__ == "__main__":
    main()
