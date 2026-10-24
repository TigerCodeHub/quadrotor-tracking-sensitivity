"""Build a purpose-built Saltelli sweep for variance-based (Sobol) sensitivity indices.

WHY THIS EXISTS: campaigns v1-v4 use
space-filling Sobol SAMPLING, which does NOT support the Saltelli first-order (S_i) /
total-effect (S_Ti) estimator -- that needs matched A / B / AB_i (and, for second order,
BA_i) evaluation matrices, not a plain low-discrepancy sequence. So the indices CANNOT be
computed from existing data; this generates the required structured design.

DESIGN (matches the established, hardware-anchored v2 ranges in tdresearch/campaign.py so
the indices are directly comparable to the v2/v4 regression coefficients):

  factor              unit / range                 source
  0 payload_fraction  [0.00, 0.30]  -> mass_kg = nominal*(1+pf)   campaign.py primary
  1 turbulence_rms    [0.0, 3.0]  m/s (Dryden sigma_u=sigma_v)    campaign.py primary
  2 mean_wind         [0.0, 6.0]  m/s                             campaign.py primary
  3 thrust_to_weight  [1.5, 3.0]                                  campaign.py
  4 inertia_scale     [0.75, 1.25]                                campaign.py
  5 com_y_frac        [-0.05, 0.05] -> com_y_m = frac*wheelbase   campaign.py
  6 wind_direction    [0.0, 360.0] deg  -- CONTINUOUS             REQUIRED for Saltelli
  7 seed_unit         [0.0, 1.0]  -> wind_seed int  -- SAMPLED    REQUIRED for Saltelli

Two deliberate departures from v2, both mandated by the variance-based estimator:
  * wind_direction is CONTINUOUS 0-360 (v2 used a 4-cardinal categorical). Sobol indices
    assume continuous, independent inputs. Its effect is course-relative (v2/v3-B), so the
    single fixed course here turns direction into a genuine continuous factor.
  * the turbulence/gust seed is an EXPLICIT sampled factor. Saltelli assumes f is
    deterministic given its inputs; the Dryden plugin RNG is keyed off `wind_seed`
    (tdresearch/sim.py), so promoting the seed to a factor makes f deterministic. Its own
    S_i then measures the irreducible stochastic-noise share (a built-in noise-floor /
    convergence check -- it should be small; a large S_i means N is too low or the metric
    is seed-dominated).

SCOPE: a SINGLE fixed base airframe (default px4_x500) on a SINGLE fixed path (default
`dlv_a_v0`, the shortest real-world delivery route -- 2 km urban out-and-back -- so the
indices apply to the delivery context v4 studies; `straight` runs the synthetic v2 baseline
instead), turb_model=dryden (the accepted MIL-STD-1797A model, as v3-D/v4). Delivery plans
carry per-archetype MPC speed columns; the straight baseline leaves them empty. Fixing the
airframe and path removes the two categoricals so all 8 factors are continuous and
independent -- the setup a valid Sobol decomposition requires. NOTE: on a dogleg/out-and-back
route the wind_direction index reflects the route's mixed leg bearings (report it
course-relative in analysis; v3-B established heading isotropy / course-relative wind).

ROW ORDER IS LOAD-BEARING: SALib.analyze.sobol.analyze consumes Y in the exact order
saltelli.sample emitted it. Every row carries a `saltelli_index` (0..M-1); NEVER shuffle,
and the analyzer re-sorts by it before reshaping. Unlike the v4 chunks, a chunk is NOT
independently analyzable -- indices need the COMPLETE set. Chunks exist only so the fleet
launcher can fly the sweep in ~20k-run pieces; the analyzer recombines them.

Usage:
  python3 scripts/saltelli/make_saltelli_design.py <out_dir> \
      [--N 4096] [--airframe px4_x500] [--plan straight] \
      [--catalog generated/airframes.csv] [--chunk-size 20000] [--no-second-order]

Cost: M = N*(2k+2) with second order (default), or N*(k+2) without; k=8.
  N=4096, second order -> 4096*18 = 73,728 runs (~3-4 fleet chunks).
"""
import argparse
import csv
import json
import math
from pathlib import Path

# Factor definition -- ORDER IS FIXED and shared with saltelli_analyze.py via the JSON
# sidecar this script writes. Do not reorder.
FACTOR_NAMES = [
    "payload_fraction",
    "turbulence_rms",
    "mean_wind",
    "thrust_to_weight",
    "inertia_scale",
    "com_y_frac",
    "wind_direction_deg",
    "seed_unit",
]
FACTOR_BOUNDS = [
    [0.0, 0.30],
    [0.0, 3.0],
    [0.0, 6.0],
    [1.5, 3.0],
    [0.75, 1.25],
    [-0.05, 0.05],
    [0.0, 360.0],
    [0.0, 1.0],
]
# The subset reported as physical sensitivities. seed_unit is kept as a factor (so f is
# deterministic) but its index is only a noise-floor diagnostic.
REPORTED_FACTORS = FACTOR_NAMES[:7]

GRAVITY = 9.80665
SEED_MAX = 2**31 - 2  # Gazebo's Ruby launcher casts --seed to a signed C int.
SAMPLE_SEED = 20260919  # fixed so the Saltelli draw is reproducible

# Per-archetype cruise/vel-max speeds for the delivery routes (as make_v4_design.py:
# sim.py applies these as MPC_XY_CRUISE / MPC_XY_VEL_MAX / MPC_Z_VEL_MAX_UP/DN after
# takeoff). Only set for dlv_* plans; empty for the synthetic `straight` baseline (empty
# -> sim.py leaves the MPC params at their v1/v2/v3 defaults, byte-identical).
ARCHETYPE_SPEEDS = {"a": (10.0, 13.0), "b": (15.0, 18.0), "c": (12.0, 15.0)}
CLIMB_RATE = 2.5
DESCENT_RATE = 2.0


def _speed_cols(plan):
    """Delivery routes carry per-archetype MPC speed columns; straight leaves them empty."""
    if not plan.startswith("dlv_"):
        return {"cruise_speed_m_s": "", "xy_vel_max_m_s": "",
                "climb_rate_m_s": "", "descent_rate_m_s": ""}
    archetype = plan.split("_")[1]  # dlv_<a|b|c>_v<n>
    cruise, xy_max = ARCHETYPE_SPEEDS[archetype]
    return {"cruise_speed_m_s": cruise, "xy_vel_max_m_s": xy_max,
            "climb_rate_m_s": CLIMB_RATE, "descent_rate_m_s": DESCENT_RATE}


def _saltelli_sample(problem, n, second_order):
    """SALib moved saltelli.sample -> sobol.sample around 1.4.6; support both."""
    try:  # newer SALib
        from SALib.sample import sobol as sobol_sample
        return sobol_sample.sample(
            problem, n, calc_second_order=second_order, seed=SAMPLE_SEED
        )
    except Exception:  # older SALib
        from SALib.sample import saltelli
        return saltelli.sample(problem, n, calc_second_order=second_order, seed=SAMPLE_SEED)


def read_airframe(catalog, airframe_id):
    with Path(catalog).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["airframe_id"] == airframe_id:
                return row
    raise SystemExit(f"airframe {airframe_id!r} not found in {catalog}")


def build_row(idx, factors, af, plan):
    """Construct one flyable design row from a sampled factor vector.

    Mirrors tdresearch/campaign.py's row schema exactly (generate.py + sim.py read these
    columns), then applies the two Saltelli-mandated overrides: continuous wind direction
    and the seed-derived wind_seed.
    """
    (payload_fraction, turbulence_rms, mean_wind, thrust_to_weight,
     inertia_scale, com_y_frac, wind_direction_deg, seed_unit) = factors

    nominal_mass = float(af["nominal_mass_kg"])
    wheelbase = float(af["wheelbase_m"])
    rotor_diameter = float(af["rotor_diameter_m"])
    motor_constant = float(af["motor_constant_n_per_rad_s2"])
    base_inertia = (
        float(af["inertia_xx_kg_m2"]),
        float(af["inertia_yy_kg_m2"]),
        float(af["inertia_zz_kg_m2"]),
    )
    rotor_area = 4.0 * math.pi * (rotor_diameter / 2.0) ** 2

    mass_kg = nominal_mass * (1.0 + payload_fraction)
    com_y = com_y_frac * wheelbase
    max_rot_velocity = math.sqrt(
        thrust_to_weight * mass_kg * GRAVITY / (4.0 * motor_constant)
    )
    wind_seed = int(round(seed_unit * SEED_MAX))
    wind_seed = min(max(wind_seed, 1), SEED_MAX)

    model_name = f"slt_{af['airframe_id']}_{idx:06d}"
    run_id = f"{model_name}_sl"
    row = {
        "experiment": "saltelli_sobol",
        "study_region": "primary",
        "saltelli_index": idx,
        "run_id": run_id,
        "sample_index": idx,
        "airframe_id": af["airframe_id"],
        "airframe_name": af["display_name"],
        "model_name": model_name,
        "nominal_mass_kg": round(nominal_mass, 8),
        "payload_mass_kg": round(mass_kg - nominal_mass, 8),
        "payload_mass_pct": round(payload_fraction * 100.0, 8),
        "mass_kg": round(mass_kg, 8),
        "mass_pct": round(payload_fraction * 100.0, 8),
        "wheelbase_m": round(wheelbase, 8),
        "rotor_diameter_m": round(rotor_diameter, 8),
        "motor_constant_n_per_rad_s2": motor_constant,
        "rotor_disk_area_m2": round(rotor_area, 8),
        "disk_loading_kg_m2": round(mass_kg / rotor_area, 8),
        "thrust_to_weight": round(thrust_to_weight, 8),
        "max_thrust_n": round(thrust_to_weight * mass_kg * GRAVITY, 8),
        "max_rot_velocity_rad_s": round(max_rot_velocity, 8),
        "inertia_scale": round(inertia_scale, 8),
        "inertia_xx_kg_m2": round(base_inertia[0] * inertia_scale, 10),
        "inertia_yy_kg_m2": round(base_inertia[1] * inertia_scale, 10),
        "inertia_zz_kg_m2": round(base_inertia[2] * inertia_scale, 10),
        "inertia_mode": "absolute",
        "com_x_m": 0.0,
        "com_y_m": round(com_y, 8),
        "com_z_m": 0.0,
        "payload_x": 0.0,
        "payload_y": round(com_y, 8),
        "payload_z": 0.0,
        "mean_wind_m_s": round(mean_wind, 8),
        "turbulence_rms_m_s": round(turbulence_rms, 8),
        "wind_direction_deg": round(wind_direction_deg, 8),  # CONTINUOUS override
        "turb_model": "dryden",
        "wind_seed": wind_seed,  # sampled factor -> Dryden RNG seed (sim.py)
        "wind_level": "calm",
        "rotor_pct": 0,
        "plan": plan,
        "controller_mode": "fixed",
        "repeat": 1,
        "design_seed": SAMPLE_SEED,
        "parameter_confidence": af["parameter_confidence"],
        "execution_order": idx + 1,
    }
    row.update(_speed_cols(plan))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--N", type=int, default=4096,
                    help="base sample size; runs = N*(2k+2) second order, N*(k+2) otherwise")
    ap.add_argument("--airframe", default="px4_x500")
    ap.add_argument("--plan", default="dlv_a_v0",
                    help="single fixed path; dlv_a_v0 = shortest real delivery route "
                         "(gets archetype speed cols), or `straight` for the synthetic baseline")
    ap.add_argument("--catalog", default=str(Path(__file__).resolve().parents[2] / "generated/airframes.csv"))
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--no-second-order", action="store_true",
                    help="skip second-order (S_ij) -> N*(k+2) runs instead of N*(2k+2)")
    args = ap.parse_args()

    second_order = not args.no_second_order
    k = len(FACTOR_NAMES)
    problem = {"num_vars": k, "names": FACTOR_NAMES, "bounds": FACTOR_BOUNDS}

    samples = _saltelli_sample(problem, args.N, second_order)
    m = len(samples)
    expected = args.N * ((2 * k + 2) if second_order else (k + 2))
    assert m == expected, f"SALib returned {m} rows, expected {expected}"

    af = read_airframe(args.catalog, args.airframe)
    rows = [build_row(i, samples[i], af, args.plan) for i in range(m)]
    assert len({r["run_id"] for r in rows}) == m, "duplicate run_id"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    def write(path, out_rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)

    write(out_dir / "saltelli_all.csv", rows)
    n_chunks = math.ceil(m / args.chunk_size)
    for c in range(n_chunks):
        chunk = rows[c * args.chunk_size:(c + 1) * args.chunk_size]
        write(out_dir / f"saltelli_c{c + 1}.csv", chunk)

    # Problem sidecar -- the analyzer reads this so the factor order, N and second-order
    # flag can never drift between sampling and analysis.
    meta = {
        "problem": problem,
        "N": args.N,
        "calc_second_order": second_order,
        "num_runs": m,
        "reported_factors": REPORTED_FACTORS,
        "airframe": args.airframe,
        "plan": args.plan,
        "turb_model": "dryden",
        "sample_seed": SAMPLE_SEED,
        "run_id_to_index": "saltelli_index column; sort by it before reshaping Y",
    }
    (out_dir / "saltelli_problem.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"airframe {args.airframe} ({af['nominal_mass_kg']} kg nominal), plan {args.plan}")
    print(f"k={k} factors, N={args.N}, second_order={second_order}")
    print(f"wrote {m} runs -> saltelli_all.csv + {n_chunks} chunk file(s) "
          f"(<= {args.chunk_size}/chunk)")
    print(f"wrote saltelli_problem.json (factor order + N pinned for the analyzer)")


if __name__ == "__main__":
    main()
