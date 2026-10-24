"""Build the v4 delivery-route design: four independently-analyzable ~20k chunks + canary.

Every row is a verbatim copy of an eligible v2 design row (same model/SDF, wind_seed, mean
wind, direction, turbulence_rms -- the Sobol draws are inherited) with the plan swapped to a
delivery route (dlv_{a,b,c}_{v0,v1,v2}, see scripts/v4/make_delivery_plans.py for the cited
geometry), turb_model=dryden, and per-archetype speed columns (applied by sim.py as
MPC_XY_CRUISE / MPC_XY_VEL_MAX / MPC_Z_VEL_MAX_UP / MPC_Z_VEL_MAX_DN):

  dlv_a  cruise 10 m/s (Matternet M2 10; Rodrigues 2022 base 12)
  dlv_b  cruise 15 m/s (DJI FlyCart 30 auto-route max 15; below Manna 17)
  dlv_c  cruise 12 m/s (Rodrigues 2022 base)
  all    climb 2.5 / descent 2.0 m/s (Rodrigues 2022, directly published)

Chunk layout (each chunk = balanced 5 airframes x 9 plans x full Sobol parameter range, so
each has standalone analysis value; the campaign flies c1 -> c4 in order):
  c1  the 20k v3-D config set (models with a flown <model>_dw dryden-straight control) ->
      paired delivery-vs-straight route contrast under IDENTICAL turbulence model, plus a
      standalone 20k sensitivity fit. paired_with_run_id = <model>_dw.
  c2-c4  the remaining eligible v2 configs, round-robin interleaved per airframe (each chunk
      a low-discrepancy subset). paired_with_run_id = the v2 white straight control.
The canary (2 configs per plan x airframe, drawn from c1) is a pipeline check flown before c1.

Plan assignment uses a golden-ratio low-discrepancy sweep on the within-airframe index
(decorrelated from the Sobol parameter values, as in make_curvature_design.py).

Usage:
  python3 scripts/v4/make_v4_design.py <out_dir> [--chunk-size 20000]
"""
import csv
import math
import sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parents[2]
V2_DESIGN = REPO / "generated/designs/sobol_v2_2e14.csv"
V2_CONTROL = REPO / "results/campaign_v2/combined_cruise.csv"
V3D_DESIGN = REPO / "generated/designs/v3_dryden/v3_d.csv"

PHI = (math.sqrt(5.0) - 1.0) / 2.0
PLANS = [f"dlv_{arch}_{var}" for arch in "abc" for var in ("v0", "v1", "v2")]
SPEEDS = {  # archetype -> (cruise, xy_max); climb/descent uniform
    "a": (10.0, 13.0),
    "b": (15.0, 18.0),
    "c": (12.0, 15.0),
}


def eligible_models():
    """v2 models whose winning flight used the canonical speed profile (as make_v3d_design)."""
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
            if row.get("model_name"):
                ok.add(row["model_name"])
    return ok


def golden_plan(index, exclude=None, multiplier=PHI):
    """Low-discrepancy plan pick; with exclude set, picks among the other 8."""
    pool = PLANS if exclude is None else [p for p in PLANS if p != exclude]
    return pool[int(((index * multiplier) % 1.0) * len(pool))]


def delivery_row(source, plan, chunk, out_fields):
    row = dict(source)
    row["plan"] = plan
    row["turb_model"] = "dryden"
    row["route_chunk"] = chunk
    cruise, xy_max = SPEEDS[plan.split("_")[1]]
    row["cruise_speed_m_s"] = cruise
    row["xy_vel_max_m_s"] = xy_max
    row["climb_rate_m_s"] = 2.5
    row["descent_rate_m_s"] = 2.0
    row["paired_with_run_id"] = (
        "%s_dw" % source["model_name"] if chunk == "c1" else source["run_id"]
    )
    row["run_id"] = "%s_v4%s" % (source["model_name"], chunk if chunk != "canary" else "cy")
    return {k: row.get(k, "") for k in out_fields}


def main():
    out_dir = Path(sys.argv[1])
    chunk_size = 20000
    if "--chunk-size" in sys.argv:
        chunk_size = int(sys.argv[sys.argv.index("--chunk-size") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = eligible_models()
    print("eligible v2 control models: %d" % len(ok))
    with V3D_DESIGN.open(newline="", encoding="utf-8-sig") as handle:
        dw_models = {row["model_name"] for row in csv.DictReader(handle)}
    print("v3-D (dryden straight control) models: %d" % len(dw_models))

    with V2_DESIGN.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r["model_name"] in ok and (r.get("repeat") or "1") == "1"]
    print("v2 design rows with an eligible control: %d" % len(rows))

    extra = [
        "turb_model", "paired_with_run_id", "route_chunk",
        "cruise_speed_m_s", "xy_vel_max_m_s", "climb_rate_m_s", "descent_rate_m_s",
    ]
    out_fields = fields + [c for c in extra if c not in fields]

    by_airframe = defaultdict(list)
    for row in rows:
        by_airframe[row["airframe_id"]].append(row)
    for group in by_airframe.values():
        group.sort(key=lambda r: int(r["sample_index"]))

    chunks = {name: [] for name in ("c1", "c2", "c3", "c4")}
    per_airframe_c1 = chunk_size // len(by_airframe)
    for airframe, group in sorted(by_airframe.items()):
        c1_group = [r for r in group if r["model_name"] in dw_models][:per_airframe_c1]
        rest = [r for r in group if r["model_name"] not in dw_models]
        for i, source in enumerate(c1_group):
            plan = golden_plan(i)
            chunks["c1"].append(delivery_row(source, plan, "c1", out_fields))
        for i, source in enumerate(rest):
            plan = golden_plan(i + 7919)  # offset so c2-c4 phase differs from c1
            chunk = "c%d" % (2 + i % 3)
            chunks[chunk].append(delivery_row(source, plan, chunk, out_fields))

    # canary: 2 configs per (plan x airframe) drawn from c1 = 90 rows
    canary = []
    seen = Counter()
    for row in chunks["c1"]:
        key = (row["plan"], row["airframe_id"])
        if seen[key] < 2:
            seen[key] += 1
            source = dict(row)
            source["run_id"] = source["run_id"].replace("_v4c1", "_v4cy")
            source["route_chunk"] = "canary"
            canary.append(source)

    all_ids = [r["run_id"] for c in chunks.values() for r in c] + [r["run_id"] for r in canary]
    assert len(all_ids) == len(set(all_ids)), "duplicate run_ids"

    def write(path, out_rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(out_rows)
        plans = Counter(r["plan"] for r in out_rows)
        airframes = Counter(r["airframe_id"] for r in out_rows)
        print("wrote %s: %d rows, %d plans (min %d max %d/plan), airframes %s"
              % (path.name, len(out_rows), len(plans), min(plans.values()),
                 max(plans.values()), dict(sorted(airframes.items()))))

    for name in ("c1", "c2", "c3", "c4"):
        write(out_dir / f"v4_{name}.csv", chunks[name])
    write(out_dir / "v4_canary.csv", canary)
    print("total campaign rows: %d" % sum(len(c) for c in chunks.values()))


if __name__ == "__main__":
    main()
