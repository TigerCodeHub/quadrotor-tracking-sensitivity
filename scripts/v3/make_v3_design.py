"""Build the v3 design: four treatment arms paired against already-flown v2 runs.

EXTENSIONS items 1 and 2. Every arm is a verbatim copy of a v2 design row with exactly one
field changed, so each v3 run pairs with a v2 run on the SAME model and the SAME wind_seed --
i.e. the identical gust realisation. Paired within-config differences cancel airframe and
config effects entirely, and the control arm costs nothing because it is already flown.

  A1  controller_mode=retuned   existing mass-scheduled gains (+/-12%)
  A2  controller_mode=matched   bandwidth-matched gains, stronger and per-loop
  B1  plan=straight_hXXX        course rotated to one of 7 non-zero headings
  B2  plan=turn_lXXX|climb_sX   3 turn sharpnesses + 2 climb steepnesses

A1 and A2 on the same configs give a dose-response in retuning strength
(fixed -> gentle -> matched), which is far harder to explain away than a single A/B.

Control-arm eligibility is strict: a v2 row only qualifies if its winning flight used
cruise 6x AND takeoff 1x, so the ~2% of v2 rows rescued at 3x in recovery cannot enter a
pair. Config selection is systematic (every k-th sample_index per airframe) rather than
random, which preserves the Sobol sequence's low-discrepancy coverage instead of clumping.

Usage:
  python scripts/v3/make_v3_design.py <n_per_block> <out_dir> [--canary N]
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
V2_DESIGN = REPO / "generated/designs/sobol_v2_2e14.csv"
V2_CONTROL = REPO / "results/campaign_v2/combined_cruise.csv"

HEADINGS = [45, 90, 135, 180, 225, 270, 315]          # h000 is the v2 control itself
MANEUVERS = ["turn_l100", "turn_l250", "turn_l400", "climb_s1", "climb_s2"]


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
                continue          # recovered at 3x -> not comparable
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
    n_per_block = int(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = eligible_models()
    print("eligible v2 control models: %d" % len(ok))

    with V2_DESIGN.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        # repeat=1 only. Replicate 2 is a partial pass (8,190 of 81,920 configs) that every v2
        # fit auto-excludes as unbalanced, so pairing against it would mean pairing against
        # runs outside the analysed set. Restricting here also guarantees one row per model,
        # hence unique v3 run_ids.
        rows = [r for r in reader if r["model_name"] in ok and (r.get("repeat") or "1") == "1"]
    print("v2 design rows with an eligible control: %d" % len(rows))

    by_airframe = defaultdict(list)
    for row in rows:
        by_airframe[row["airframe_id"]].append(row)
    for airframe in by_airframe:
        by_airframe[airframe].sort(key=lambda r: int(r["sample_index"]))

    per_airframe = n_per_block // len(by_airframe)
    chosen = []
    for airframe, group in sorted(by_airframe.items()):
        step = max(1, len(group) // per_airframe)
        picked = group[::step][:per_airframe]
        chosen.extend(picked)
        print("  %-12s pool=%-6d step=%-3d picked=%d" % (airframe, len(group), step, len(picked)))
    print("selected configs: %d" % len(chosen))

    extra = ["course_heading_deg", "maneuver", "v3_block", "paired_with_run_id"]
    out_fields = fields + [c for c in extra if c not in fields]

    def emit(name, mutate):
        path = out_dir / ("v3_%s.csv" % name)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=out_fields)
            writer.writeheader()
            for index, source in enumerate(chosen):
                row = dict(source)
                row["paired_with_run_id"] = source["run_id"]
                row["v3_block"] = name
                row["course_heading_deg"] = 0.0
                row["maneuver"] = "straight"
                # run_id must be new (the runner skips a row whose ULog already exists) but
                # wind_seed is copied verbatim, which is what makes the pair valid.
                row["run_id"] = "%s_%s" % (source["model_name"], name)
                mutate(row, index)
                writer.writerow({k: row.get(k, "") for k in out_fields})
        print("wrote %s (%d rows)" % (path, len(chosen)))
        return path

    def a1(row, _index):
        row["controller_mode"] = "retuned"

    def a2(row, _index):
        row["controller_mode"] = "matched"

    def b1(row, index):
        heading = HEADINGS[index % len(HEADINGS)]
        row["plan"] = "straight_h%03d" % heading
        row["course_heading_deg"] = float(heading)

    def b2(row, index):
        maneuver = MANEUVERS[index % len(MANEUVERS)]
        row["plan"] = maneuver
        row["maneuver"] = maneuver

    paths = [emit("a1", a1), emit("a2", a2), emit("b1", b1), emit("b2", b2)]

    combined = out_dir / "v3_all.csv"
    with combined.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        total = 0
        for path in paths:
            with path.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    writer.writerow(row)
                    total += 1
    print("wrote %s (%d rows total)" % (combined, total))

    # Canary: a few rows from each block, so a broken block costs minutes not hours.
    if "--canary" in sys.argv:
        per = int(sys.argv[sys.argv.index("--canary") + 1]) // len(paths)
        canary = out_dir / "v3_canary.csv"
        with canary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=out_fields)
            writer.writeheader()
            count = 0
            for path in paths:
                with path.open(newline="", encoding="utf-8") as source:
                    picked = list(csv.DictReader(source))
                # spread across airframes and, for B, across headings/maneuvers
                step = max(1, len(picked) // per)
                for row in picked[::step][:per]:
                    writer.writerow(row)
                    count += 1
        print("wrote %s (%d rows)" % (canary, count))


if __name__ == "__main__":
    main()
