import csv
import hashlib
import math
import random
from pathlib import Path

from scipy.stats import qmc


AIRFRAME_FIELDS = (
    "airframe_id",
    "display_name",
    "nominal_mass_kg",
    "wheelbase_m",
    "rotor_diameter_m",
    "motor_constant_n_per_rad_s2",
    "upstream_max_rot_velocity_rad_s",
    "inertia_xx_kg_m2",
    "inertia_yy_kg_m2",
    "inertia_zz_kg_m2",
    "parameter_confidence",
    "source_url",
)

WIND_DIRECTIONS_DEG = (0.0, 90.0, 180.0, 270.0)
STUDY_REGIONS = {
    # Intended to yield mostly complete trajectories for tracking-error models.
    "primary": {
        "mean_wind_m_s": (0.0, 6.0),
        "turbulence_rms_m_s": (0.0, 3.0),
    },
    # Retains the original envelope for a smaller, failure-focused study.
    "boundary": {
        "mean_wind_m_s": (0.0, 10.0),
        "turbulence_rms_m_s": (0.0, 6.0),
    },
}


def read_airframes(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("airframe catalog is empty")
    missing = [field for field in AIRFRAME_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"airframe catalog is missing columns: {', '.join(missing)}")
    ids = [row["airframe_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("airframe_id values must be unique")
    for row in rows:
        for field in AIRFRAME_FIELDS[2:10]:
            if float(row[field]) <= 0:
                raise ValueError(f"{row['airframe_id']}: {field} must be positive")
    return rows


def _scale(low, high, value):
    return low + (high - low) * value


def _stable_seed(seed, airframe_index, sample_index, repeat):
    payload = f"{seed}:{airframe_index}:{sample_index}:{repeat}".encode("ascii")
    # Gazebo's Ruby launcher converts --seed to a signed C int.
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big") % (2**31 - 1)


def generate_sobol_campaign(
    airframes,
    samples_per_airframe=2048,
    repetitions=10,
    seed=20260717,
    plan="straight",
    region="primary",
):
    if samples_per_airframe <= 0:
        raise ValueError("samples_per_airframe must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if region not in STUDY_REGIONS:
        choices = ", ".join(sorted(STUDY_REGIONS))
        raise ValueError(f"unknown study region {region!r}; choose from {choices}")
    bounds = STUDY_REGIONS[region]

    sampler = qmc.Sobol(d=7, scramble=True, seed=seed)
    exponent = math.log2(samples_per_airframe)
    if exponent.is_integer():
        points = sampler.random_base2(int(exponent))
    else:
        points = sampler.random(samples_per_airframe)

    rows = []
    for airframe_index, airframe in enumerate(airframes):
        nominal_mass = float(airframe["nominal_mass_kg"])
        wheelbase = float(airframe["wheelbase_m"])
        rotor_diameter = float(airframe["rotor_diameter_m"])
        motor_constant = float(airframe["motor_constant_n_per_rad_s2"])
        rotor_area = 4.0 * math.pi * (rotor_diameter / 2.0) ** 2
        base_inertia = tuple(float(airframe[name]) for name in AIRFRAME_FIELDS[7:10])

        # A Cranley-Patterson shift gives each airframe distinct, reproducible
        # samples while preserving the low-discrepancy structure.
        shifted = (points + airframe_index / len(airframes)) % 1.0
        for sample_index, point in enumerate(shifted):
            payload_fraction = _scale(0.0, 0.30, point[0])
            mass_kg = nominal_mass * (1.0 + payload_fraction)
            turbulence_rms = _scale(*bounds["turbulence_rms_m_s"], point[1])
            mean_wind = _scale(*bounds["mean_wind_m_s"], point[2])
            thrust_to_weight = _scale(1.5, 3.0, point[3])
            inertia_scale = _scale(0.75, 1.25, point[4])
            com_x = 0.0
            com_y = _scale(-0.05, 0.05, point[5]) * wheelbase
            com_z = 0.0
            direction_index = min(int(point[6] * len(WIND_DIRECTIONS_DEG)), len(WIND_DIRECTIONS_DEG) - 1)
            wind_direction = WIND_DIRECTIONS_DEG[direction_index]
            model_name = f"sobol_{airframe['airframe_id']}_s{sample_index:04d}"
            max_rot_velocity = math.sqrt(
                thrust_to_weight * mass_kg * 9.80665 / (4.0 * motor_constant)
            )

            for repeat in range(1, repetitions + 1):
                run_id = f"{model_name}_n{repeat:02d}"
                row = {
                    "experiment": f"multi_airframe_{region}",
                    "study_region": region,
                    "run_id": run_id,
                    "sample_index": sample_index,
                    "airframe_id": airframe["airframe_id"],
                    "airframe_name": airframe["display_name"],
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
                    "max_thrust_n": round(thrust_to_weight * mass_kg * 9.80665, 8),
                    "max_rot_velocity_rad_s": round(max_rot_velocity, 8),
                    "inertia_scale": round(inertia_scale, 8),
                    "inertia_xx_kg_m2": round(base_inertia[0] * inertia_scale, 10),
                    "inertia_yy_kg_m2": round(base_inertia[1] * inertia_scale, 10),
                    "inertia_zz_kg_m2": round(base_inertia[2] * inertia_scale, 10),
                    "inertia_mode": "absolute",
                    "com_x_m": round(com_x, 8),
                    "com_y_m": round(com_y, 8),
                    "com_z_m": round(com_z, 8),
                    "payload_x": round(com_x, 8),
                    "payload_y": round(com_y, 8),
                    "payload_z": round(com_z, 8),
                    "mean_wind_m_s": round(mean_wind, 8),
                    "turbulence_rms_m_s": round(turbulence_rms, 8),
                    "wind_direction_deg": wind_direction,
                    "wind_seed": _stable_seed(seed, airframe_index, sample_index, repeat),
                    "wind_level": "calm",
                    "rotor_pct": 0,
                    "plan": plan,
                    "controller_mode": "fixed",
                    "repeat": repeat,
                    "design_seed": seed,
                    "parameter_confidence": airframe["parameter_confidence"],
                }
                rows.append(row)
    random.Random(seed).shuffle(rows)
    for execution_order, row in enumerate(rows, start=1):
        row["execution_order"] = execution_order
    return rows


def campaign_summary(rows):
    airframes = sorted({row["airframe_id"] for row in rows})
    samples = {(row["airframe_id"], int(row["sample_index"])) for row in rows}
    return {
        "airframes": airframes,
        "airframe_count": len(airframes),
        "configuration_count": len(samples),
        "run_count": len(rows),
    }


def generate_baseline_campaign(airframes, repetitions=1, plan="smoke", seed=20260717):
    rows = []
    for airframe_index, airframe in enumerate(airframes):
        mass = float(airframe["nominal_mass_kg"])
        motor_constant = float(airframe["motor_constant_n_per_rad_s2"])
        max_rot_velocity = float(airframe["upstream_max_rot_velocity_rad_s"])
        max_thrust = 4.0 * motor_constant * max_rot_velocity**2
        model_name = f"baseline_{airframe['airframe_id']}"
        for repeat in range(1, repetitions + 1):
            rows.append({
                "experiment": "multi_airframe_baseline",
                "run_id": f"{model_name}_n{repeat:02d}",
                "sample_index": 0,
                "airframe_id": airframe["airframe_id"],
                "airframe_name": airframe["display_name"],
                "model_name": model_name,
                "nominal_mass_kg": mass,
                "payload_mass_kg": 0.0,
                "payload_mass_pct": 0.0,
                "mass_kg": mass,
                "mass_pct": 0.0,
                "wheelbase_m": float(airframe["wheelbase_m"]),
                "rotor_diameter_m": float(airframe["rotor_diameter_m"]),
                "motor_constant_n_per_rad_s2": motor_constant,
                "thrust_to_weight": max_thrust / (mass * 9.80665),
                "max_thrust_n": max_thrust,
                "max_rot_velocity_rad_s": max_rot_velocity,
                "inertia_scale": 1.0,
                "inertia_xx_kg_m2": float(airframe["inertia_xx_kg_m2"]),
                "inertia_yy_kg_m2": float(airframe["inertia_yy_kg_m2"]),
                "inertia_zz_kg_m2": float(airframe["inertia_zz_kg_m2"]),
                "inertia_mode": "absolute",
                "com_x_m": 0.0, "com_y_m": 0.0, "com_z_m": 0.0,
                "mean_wind_m_s": 0.0, "turbulence_rms_m_s": 0.0,
                "wind_direction_deg": 0.0,
                "wind_seed": _stable_seed(seed, airframe_index, 0, repeat),
                "wind_level": "calm", "rotor_pct": 0, "plan": plan,
                "controller_mode": "fixed", "repeat": repeat,
                "design_seed": seed, "parameter_confidence": airframe["parameter_confidence"],
                "execution_order": len(rows) + 1,
            })
    return rows
