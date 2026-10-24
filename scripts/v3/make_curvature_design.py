"""Build the v3-C constant-curvature design: a clean curvature sweep at fixed path length.

Every row copies a v2 config verbatim (same model, same wind_seed) and sets its plan to a
constant-length arc (arc_aXXX from make_arc_plans.py). The v2 control it pairs with flew the
1000 m STRAIGHT route, so the within-pair delta d = log(sob2_arc) - log(sob2_straight) is a
length-matched curvature contrast -- exactly what v3-B could not deliver (its turns changed
length and curvature together, and SOB2 normalises per step).

Curvature is assigned per config by a golden-ratio additive-recurrence sequence over the
selected-config index -- a 1-D low-discrepancy ("quasi-random") sweep. Because the selection
interleaves airframes, this spreads curvature evenly across airframes and wind seeds and
decorrelates it from every design covariate, so the curvature effect is not confounded.
Assignment is deterministic (no RNG; Date/random are unavailable on the fleet anyway).

  usage: make_curvature_design.py <n_total> <out_csv> [--step-deg 3] [--max-deg 180]
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_DESIGN = REPO / "generated/designs/sobol_v2_2e14.csv"
V2_CONTROL = REPO / "results/campaign_v2/combined_cruise.csv"
PHI = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio fractional part, the low-discrepancy step


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
    ap.add_argument("--step-deg", type=int, default=3)
    ap.add_argument("--max-deg", type=int, default=180)
    args = ap.parse_args()

    levels = list(range(0, args.max_deg + 1, args.step_deg))  # arc_aXXX turn-angle grid

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
        chosen.extend(group[::step][:per_af])
        print("  %-12s pool=%-6d step=%-3d picked=%d" % (af, len(group), step, min(per_af, len(group[::step]))))
    print("selected configs: %d" % len(chosen))

    extra = ["course_heading_deg", "maneuver", "v3_block", "paired_with_run_id",
             "arc_turn_deg", "arc_kappa_per_m", "arc_a_normal_m_s2"]
    out_fields = fields + [c for c in extra if c not in fields]

    counts = defaultdict(int)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=out_fields)
        w.writeheader()
        for i, source in enumerate(chosen):
            # low-discrepancy curvature draw in [0,1) -> nearest turn-angle level
            t = (0.5 + (i + 1) * PHI) % 1.0
            deg = min(levels, key=lambda d: abs(d - t * args.max_deg))
            kappa = math.radians(deg) / 1000.0            # constant arc length L=1000 m
            a_norm = (5.0 ** 2) * kappa                    # cruiseSpeed^2 * kappa
            row = dict(source)
            row["paired_with_run_id"] = source["run_id"]
            row["v3_block"] = "c1"
            row["course_heading_deg"] = 0.0
            row["maneuver"] = "arc"
            row["plan"] = "arc_a%03d" % deg
            row["arc_turn_deg"] = deg
            row["arc_kappa_per_m"] = "%.6f" % kappa
            row["arc_a_normal_m_s2"] = "%.5f" % a_norm
            row["run_id"] = "%s_c1" % source["model_name"]
            w.writerow({k: row.get(k, "") for k in out_fields})
            counts[deg] += 1
    print("wrote %s (%d rows, %d curvature levels)" % (out_path, len(chosen), len(counts)))
    lo, hi = min(counts.values()), max(counts.values())
    print("  per-level config counts: min=%d max=%d (balanced coverage)" % (lo, hi))


if __name__ == "__main__":
    main()
