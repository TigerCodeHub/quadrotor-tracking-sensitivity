"""Per-leg SOB2 for multi-waypoint delivery missions (v4).

The whole-route SOB2 comes from the standard analyze (armed/cruise segments). This adds the
REQUIRED per-segment view: one output row per (run, leg), where legs are derived
GEOMETRICALLY from the flown .plan (the minimal logger profile has no mission topics, so
mission-item indices are not in the ULog).

Method: the commanded setpoint trace (vehicle_local_position_setpoint, NED) is projected
onto the plan's waypoint polyline (converted to the same NED-local frame). Each sample is
assigned to the leg (polyline segment) minimizing point-to-segment distance among the
current leg and a small lookahead window, with a MONOTONIC leg index (the commanded
trajectory visits waypoints in order; goto_location per item in sim.py guarantees this).
Hover items (command 19) are zero-length legs: samples map to them while the commanded
setpoint sits within HOVER_RADIUS_M of the hover waypoint between the surrounding legs.

Leg labels: leg00_takeoff (origin -> first route wp), then legNN_cruise / legNN_descend /
legNN_hover / legNN_climb by comparing consecutive waypoint altitudes/positions. Per-leg
metrics: sob2 (same formula as tdresearch.metrics), mean/max deviation, samples, sim
duration. Run on a host (needs pyulog from .venv):

  PYTHONPATH=$REPO $REPO/.venv/bin/python scripts/v4/analyze_v4_legs.py \
      --runs <host-local sobol_v4> --plans $REPO/generated/sobol_v2/plans \
      --output <results_dir>/legs/<host>.csv
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdresearch.metrics import Vec3, read_ulog, _sob2_vec  # noqa: E402

LAT0, LON0 = 47.3979711, 8.5461637
R_EARTH = 6378137.0
HOVER_RADIUS_M = 8.0
LOOKAHEAD = 2  # legs ahead the assigner may jump per sample


def latlon_to_ned(lat, lon, alt):
    """Plan waypoint -> PX4 local NED (x north, y east, z down; origin = home)."""
    north = math.radians(lat - LAT0) * R_EARTH
    east = math.radians(lon - LON0) * R_EARTH * math.cos(math.radians(LAT0))
    return Vec3(north, east, -alt)


def plan_waypoints(plan_path):
    """(position, is_hover) per item, takeoff first; positions in NED-local."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    waypoints = []
    for entry in plan["mission"]["items"]:
        params = entry["params"]
        position = latlon_to_ned(float(params[4]), float(params[5]), float(params[6]))
        waypoints.append((position, int(entry.get("command", 16)) == 19))
    return waypoints


def leg_label(index, previous, current, is_hover):
    if index == 0:
        kind = "takeoff"
    elif is_hover:
        kind = "hover"
    else:
        dz = current.z - previous.z  # NED: negative = up
        horizontal = math.hypot(current.x - previous.x, current.y - previous.y)
        if horizontal < 1.0 and dz > 1.0:
            kind = "descend"
        elif horizontal < 1.0 and dz < -1.0:
            kind = "climb"
        else:
            kind = "cruise"
    return f"leg{index:02d}_{kind}"


def build_legs(waypoints):
    """Legs = polyline segments between consecutive waypoints (+ zero-length hover legs)."""
    legs = []
    for index in range(1, len(waypoints)):
        previous, current = waypoints[index - 1][0], waypoints[index][0]
        is_hover = waypoints[index][1]
        legs.append(
            {
                "label": leg_label(index - 1, previous, current, is_hover),
                "start": previous,
                "end": current,
                "hover": is_hover,
            }
        )
    return legs


def _point_segment_distance(point, start, end):
    delta = end - start
    length2 = delta.dot(delta)
    if length2 < 1e-9:
        return (point - start).norm(), 0.0
    t = max(0.0, min(1.0, (point - start).dot(delta) / length2))
    closest = start + delta.scale(t)
    return (point - closest).norm(), t


def assign_legs(setpoints, legs):
    """Monotonic leg index per sample (commanded trajectory visits legs in order)."""
    assignment = []
    current = 0
    for point in setpoints:
        best_leg, best_distance = current, None
        for offset in range(0, LOOKAHEAD + 1):
            candidate = current + offset
            if candidate >= len(legs):
                break
            leg = legs[candidate]
            if leg["hover"]:
                distance = (point - leg["end"]).norm()
                distance = max(0.0, distance - HOVER_RADIUS_M)
            else:
                distance, _ = _point_segment_distance(point, leg["start"], leg["end"])
            if best_distance is None or distance < best_distance - 1e-9:
                best_leg, best_distance = candidate, distance
        current = best_leg
        assignment.append(current)
    return assignment


CARRY_COLUMNS = (
    "run_id", "experiment", "model_name", "airframe_id", "plan", "route_chunk",
    "controller_mode", "turb_model", "mass_kg", "mass_pct", "thrust_to_weight",
    "inertia_scale", "com_y_m", "mean_wind_m_s", "turbulence_rms_m_s",
    "wind_direction_deg", "wind_seed", "cruise_speed_m_s", "xy_vel_max_m_s",
    "climb_rate_m_s", "descent_rate_m_s", "paired_with_run_id",
)


def analyze_run(case_dir, plans_dir):
    metadata_path = case_dir / "metadata.json"
    ulogs = sorted(case_dir.glob("*.ulg")) + sorted(case_dir.glob("*.ulg.gz"))
    if not metadata_path.exists() or not ulogs:
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plan_name = metadata.get("plan") or "straight"
    legs = build_legs(plan_waypoints(plans_dir / f"{plan_name}.plan"))
    run = read_ulog(ulogs[0], flight_segment="armed")
    assignment = assign_legs(run.setpoint, legs)

    rows = []
    for leg_index, leg in enumerate(legs):
        indices = [i for i, a in enumerate(assignment) if a == leg_index]
        if len(indices) < 2:
            continue
        deviations = [run.actual[i] - run.setpoint[i] for i in indices]
        norms = [d.norm() for d in deviations]
        row = {key: metadata.get(key, "") for key in CARRY_COLUMNS}
        row.update(
            {
                "leg": leg["label"],
                "leg_index": leg_index,
                "leg_length_m": (leg["end"] - leg["start"]).norm(),
                "samples": len(indices),
                "duration_s": run.times[indices[-1]] - run.times[indices[0]],
                "sob2": _sob2_vec(deviations),
                "mean_deviation_m": sum(norms) / len(norms),
                "max_deviation_m": max(norms),
            }
        )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="host-local run dir (e.g. .../sobol_v4)")
    parser.add_argument("--plans", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match", default="_v4", help="only run_ids containing this")
    args = parser.parse_args()

    runs_dir, plans_dir = Path(args.runs), Path(args.plans)
    output_rows, failures = [], []
    case_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and args.match in d.name and (d / "metadata.json").exists()
    )
    for index, case_dir in enumerate(case_dirs, start=1):
        try:
            rows = analyze_run(case_dir, plans_dir)
            output_rows.extend(rows)
            print(f"[{index}/{len(case_dirs)}] {case_dir.name}: {len(rows)} legs", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep scanning over bad logs
            failures.append((case_dir.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{index}/{len(case_dirs)}] {case_dir.name} FAILED: {exc}", flush=True)

    if output_rows:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(output_rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"wrote {output_path} ({len(output_rows)} leg rows, {len(failures)} failures)")
    else:
        print(f"no leg rows produced ({len(failures)} failures)")


if __name__ == "__main__":
    main()
