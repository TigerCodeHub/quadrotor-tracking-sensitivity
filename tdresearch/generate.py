import copy
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import jinja2


DEFAULT_WIND_LEVELS = {
    "calm": {
        "speed": 0.0,
        "max_speed": 0.0,
        "speed_stddev": 0.0,
        "direction_stddev": 0.0,
    },
    "weak": {
        "speed": 2.0,
        "max_speed": 4.0,
        "speed_stddev": math.sqrt(1.5),
        "direction_stddev": 0.5,
    },
    "moderate": {
        "speed": 4.0,
        "max_speed": 8.0,
        "speed_stddev": math.sqrt(6.0),
        "direction_stddev": 1.0,
    },
    "strong": {
        "speed": 6.0,
        "max_speed": 12.0,
        "speed_stddev": math.sqrt(10.0),
        "direction_stddev": 1.5,
    },
}

PAYLOAD_OFFSETS = {
    "payload_forward": (0.18, 0.0, 0.0),
    "payload_rear": (-0.18, 0.0, 0.0),
    "payload_left": (0.0, 0.18, 0.0),
    "payload_right": (0.0, -0.18, 0.0),
    "payload_high": (0.0, 0.0, 0.12),
    "payload_low": (0.0, 0.0, -0.08),
}

PX4_WORLD_SYSTEM_PLUGINS = (
    ("gz-sim-physics-system", "gz::sim::systems::Physics", None),
    ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands", None),
    ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster", None),
    ("gz-sim-contact-system", "gz::sim::systems::Contact", None),
    ("gz-sim-imu-system", "gz::sim::systems::Imu", None),
    ("gz-sim-air-pressure-system", "gz::sim::systems::AirPressure", None),
    ("gz-sim-air-speed-system", "gz::sim::systems::AirSpeed", None),
    ("gz-sim-apply-link-wrench-system", "gz::sim::systems::ApplyLinkWrench", None),
    ("gz-sim-navsat-system", "gz::sim::systems::NavSat", None),
    ("gz-sim-magnetometer-system", "gz::sim::systems::Magnetometer", None),
    ("gz-sim-sensors-system", "gz::sim::systems::Sensors", {"render_engine": "ogre2"}),
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_designs(output_dir, repetitions=7):
    output_dir = Path(output_dir)
    design_dir = output_dir / "designs"
    design_dir.mkdir(parents=True, exist_ok=True)

    mass_rotor = []
    for mass_pct in range(-15, 16):
        for rotor_pct in range(-10, 16):
            for repeat in range(1, repetitions + 1):
                mass_rotor.append(
                    {
                        "experiment": "mass_rotor_grid",
                        "run_id": f"mr_m{mass_pct:+03d}_r{rotor_pct:+03d}_n{repeat:02d}",
                        "model_name": f"x500_m{mass_pct:+03d}_r{rotor_pct:+03d}",
                        "mass_pct": mass_pct,
                        "rotor_pct": rotor_pct,
                        "inertia_mode": "coupled",
                        "wind_level": "moderate",
                        "plan": "straight",
                        "controller_mode": "fixed",
                        "repeat": repeat,
                    }
                )
    write_csv(design_dir / "mass_rotor_grid.csv", mass_rotor)

    mass_inertia_payload = []
    for mode in ("coupled", "mass_only", "inertia_only"):
        for mass_pct in range(-15, 16):
            for repeat in range(1, repetitions + 1):
                mass_inertia_payload.append(
                    {
                        "experiment": "mass_inertia",
                        "run_id": f"mi_{mode}_m{mass_pct:+03d}_n{repeat:02d}",
                        "model_name": f"x500_{mode}_m{mass_pct:+03d}",
                        "mass_pct": mass_pct,
                        "rotor_pct": 0,
                        "inertia_mode": mode,
                        "payload_mass_pct": 0,
                        "payload_x": 0,
                        "payload_y": 0,
                        "payload_z": 0,
                        "wind_level": "moderate",
                        "plan": "straight",
                        "controller_mode": "fixed",
                        "repeat": repeat,
                    }
                )
    for label, offset in PAYLOAD_OFFSETS.items():
        for repeat in range(1, repetitions + 1):
            mass_inertia_payload.append(
                {
                    "experiment": "payload_placement",
                    "run_id": f"{label}_n{repeat:02d}",
                    "model_name": f"x500_{label}",
                    "mass_pct": 0,
                    "rotor_pct": 0,
                    "inertia_mode": "payload",
                    "payload_mass_pct": 10,
                    "payload_x": offset[0],
                    "payload_y": offset[1],
                    "payload_z": offset[2],
                    "wind_level": "moderate",
                    "plan": "straight",
                    "controller_mode": "fixed",
                    "repeat": repeat,
                }
            )
    write_csv(design_dir / "mass_inertia_payload.csv", mass_inertia_payload)

    wind_sweep = []
    for wind_level in DEFAULT_WIND_LEVELS:
        for mass_pct in (-15, -10, -5, 0, 5, 10, 15):
            for rotor_pct in (-10, -5, 0, 5, 10, 15):
                for repeat in range(1, repetitions + 1):
                    wind_sweep.append(
                        {
                            "experiment": "wind_intensity",
                            "run_id": f"wind_{wind_level}_m{mass_pct:+03d}_r{rotor_pct:+03d}_n{repeat:02d}",
                            "model_name": f"x500_m{mass_pct:+03d}_r{rotor_pct:+03d}",
                            "mass_pct": mass_pct,
                            "rotor_pct": rotor_pct,
                            "inertia_mode": "coupled",
                            "wind_level": wind_level,
                            "plan": "straight",
                            "controller_mode": "fixed",
                            "repeat": repeat,
                        }
                    )
    write_csv(design_dir / "wind_intensity.csv", wind_sweep)

    path_controller = []
    for plan in ("straight", "turn", "climb", "descent", "hover", "headwind_leg", "crosswind_leg"):
        for mass_pct in (-15, -10, -5, 0, 5, 10, 15):
            for rotor_pct in (-10, 0, 10, 15):
                for controller_mode in ("fixed", "retuned"):
                    for repeat in range(1, repetitions + 1):
                        path_controller.append(
                            {
                                "experiment": "path_controller",
                                "run_id": f"pc_{plan}_m{mass_pct:+03d}_r{rotor_pct:+03d}_{controller_mode}_n{repeat:02d}",
                                    "model_name": f"x500_m{mass_pct:+03d}_r{rotor_pct:+03d}",
                                "mass_pct": mass_pct,
                                "rotor_pct": rotor_pct,
                                "inertia_mode": "coupled",
                                "wind_level": "moderate",
                                "plan": plan,
                                "controller_mode": controller_mode,
                                "repeat": repeat,
                            }
                        )
    write_csv(design_dir / "path_controller.csv", path_controller)
    write_csv(
        design_dir / "smoke.csv",
        [
            {
                "experiment": "smoke",
                "run_id": "smoke_x500_baseline",
                "model_name": "x500_m+00_r+00",
                "mass_pct": 0,
                "rotor_pct": 0,
                "inertia_mode": "coupled",
                "wind_level": "calm",
                "plan": "smoke",
                "controller_mode": "fixed",
                "repeat": 1,
            }
        ],
    )
    write_csv(
        design_dir / "all_experiments.csv",
        mass_rotor + mass_inertia_payload + wind_sweep + path_controller,
    )
    return design_dir


def _clamp(value, low, high):
    return max(low, min(high, value))


def generate_controller_scripts(output_dir, design_csv):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(design_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    unique = {}
    for row in rows:
        key = (row["model_name"], row.get("controller_mode", "fixed"), row.get("mass_pct", "0"))
        unique[key] = row

    baseline = {
        "MPC_XY_P": 0.95,
        "MPC_Z_P": 1.00,
        "MPC_XY_VEL_P_ACC": 1.80,
        "MPC_XY_VEL_I_ACC": 0.40,
        "MPC_Z_VEL_P_ACC": 4.00,
        "MPC_Z_VEL_I_ACC": 2.00,
        "MC_ROLLRATE_P": 0.15,
        "MC_PITCHRATE_P": 0.15,
        "MC_YAWRATE_P": 0.20,
    }

    for (model_name, controller_mode, mass_pct_text), design_row_for_mode in sorted(unique.items()):
        mass_factor = 1.0 + float(mass_pct_text or 0.0) / 100.0
        if controller_mode == "retuned":
            position_scale = max(0.7, min(1.3, 1.0 / math.sqrt(mass_factor)))
            vertical_scale = max(0.7, min(1.3, 1.0 / mass_factor))
            rate_scale = max(0.7, min(1.3, math.sqrt(mass_factor)))
        elif controller_mode == "matched":
            # v3 block A2: bandwidth-matched gains, one loop at a time.
            #
            # `retuned` above only moves gains by +/-12% over the 0-30% payload range (and
            # clamps at 1.3), which is too gentle to test whether the design-factor effects
            # are artefacts of holding gains fixed. This mode scales each loop by the physics
            # that loop actually sees:
            #   position/vertical -- PX4 converts acceleration setpoints to thrust with a
            #     hover-thrust estimate, so residual mass mis-compensation lands here; scale
            #     inversely with mass.
            #   rate -- angular acceleration = (T_max * b/2) * u / I, so holding rate-loop
            #     bandwidth needs u_gain proportional to inertia and inversely proportional to
            #     control authority (which grows with thrust-to-weight and mass).
            # Clamped to [0.5, 2.0]: wide enough to be a real treatment, tight enough that a
            # gain excursion cannot by itself destabilise the loop. T/W is normalised by 2.25,
            # the midpoint of the design's 1.5-3.0 range.
            inertia_scale = float(design_row_for_mode.get("inertia_scale") or 1.0)
            thrust_to_weight = float(design_row_for_mode.get("thrust_to_weight") or 2.25)
            position_scale = _clamp(1.0 / mass_factor, 0.5, 2.0)
            vertical_scale = _clamp(1.0 / mass_factor, 0.5, 2.0)
            rate_scale = _clamp(inertia_scale / ((thrust_to_weight / 2.25) * mass_factor), 0.5, 2.0)
        else:
            position_scale = vertical_scale = rate_scale = 1.0

        values = dict(baseline)
        for key in ("MPC_XY_P", "MPC_XY_VEL_P_ACC", "MPC_XY_VEL_I_ACC"):
            values[key] *= position_scale
        for key in ("MPC_Z_P", "MPC_Z_VEL_P_ACC", "MPC_Z_VEL_I_ACC"):
            values[key] *= vertical_scale
        for key in ("MC_ROLLRATE_P", "MC_PITCHRATE_P", "MC_YAWRATE_P"):
            values[key] *= rate_scale

        design_row = unique[(model_name, controller_mode, mass_pct_text)]
        rotor_pct = float(design_row.get("rotor_pct") or 0.0)
        requested_limit = design_row.get("max_rot_velocity_rad_s")
        motor_limit = int(round(float(requested_limit))) if requested_limit else int(
            round(1000.0 * (1.0 + rotor_pct / 100.0))
        )
        simulator_values = {f"SIM_GZ_EC_MAX{index}": motor_limit for index in range(1, 5)}
        config = {
            "model_name": model_name,
            "controller_mode": controller_mode,
            "parameters": {key: round(value, 8) for key, value in values.items()},
            "simulator_parameters": simulator_values,
        }
        path = output_dir / f"{model_name}_{controller_mode}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output_dir


def _find_first(root, path):
    value = root.find(path)
    if value is None:
        raise ValueError(f"missing XML path: {path}")
    return value


def _scale_inertia(inertia_elem, factor):
    for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        child = _find_first(inertia_elem, name)
        child.text = str(float(child.text) * factor)


def _add_payload_inertia(inertia_elem, payload_mass, x, y, z):
    updates = {
        "ixx": payload_mass * (y * y + z * z),
        "iyy": payload_mass * (x * x + z * z),
        "izz": payload_mass * (x * x + y * y),
        "ixy": -payload_mass * x * y,
        "ixz": -payload_mass * x * z,
        "iyz": -payload_mass * y * z,
    }
    for name, delta in updates.items():
        child = _find_first(inertia_elem, name)
        child.text = str(float(child.text) + delta)


def _write_model_config(path, model_name, description="PX4 Gazebo Harmonic research variant."):
    text = f"""<?xml version=\"1.0\"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version=\"1.9\">model.sdf</sdf>
  <author><name>TDResearch</name></author>
  <description>{description}</description>
</model>
"""
    path.write_text(text, encoding="utf-8")


MRS_AIRFRAME_TEMPLATES = {
    "mrs_f330": "f330.sdf.jinja",
    "mrs_f450": "f450.sdf.jinja",
    "mrs_t650": "t650.sdf.jinja",
    "mrs_m690": "m690.sdf.jinja",
}


def _normalize_mrs_motor_plugins(root):
    converted = 0
    for plugin in root.findall(".//plugin"):
        if plugin.attrib.get("filename") != "MrsGazeboCommonResources_MulticopterMotorModel":
            continue
        plugin.attrib.update(
            filename="gz-sim-multicopter-motor-model-system",
            name="gz::sim::systems::MulticopterMotorModel",
        )
        actuator = plugin.find("actuator_number")
        if actuator is None:
            raise ValueError("MRS motor plugin has no actuator_number")
        actuator.tag = "motorNumber"
        namespace = plugin.find("robotNamespace")
        if namespace is not None:
            plugin.remove(namespace)
        converted += 1
    if converted != 4:
        raise ValueError(f"expected four MRS motor plugins, found {converted}")


def _render_mrs_airframe(mrs_dir, airframe_id):
    models_dir = Path(mrs_dir) / "models"
    template = MRS_AIRFRAME_TEMPLATES[airframe_id]
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(models_dir)),
        autoescape=False,
    )
    environment.globals["math"] = math
    rendered = environment.get_template(
        f"mrs_robots_description/sdf/drones/{template}"
    ).render(spawner_args={"name": airframe_id})
    root = ET.fromstring(rendered)
    _normalize_mrs_motor_plugins(root)
    return ET.ElementTree(root)


def _copy_mrs_resources(mrs_dir, output_dir):
    if not mrs_dir:
        raise ValueError("--mrs-dir is required for MRS airframes")
    source = Path(mrs_dir) / "models" / "mrs_robots_description"
    if not source.is_dir():
        raise FileNotFoundError(f"missing MRS model resources: {source}")
    destination = Path(output_dir) / "mrs_robots_description"
    shutil.copytree(source, destination)
    license_path = Path(mrs_dir) / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"missing MRS license: {license_path}")
    shutil.copy2(license_path, destination / "LICENSE")
    apache_license = Path(__file__).resolve().parent.parent / "LICENSES" / "Apache-2.0.txt"
    if not apache_license.is_file():
        raise FileNotFoundError(f"missing bundled Apache-2.0 license: {apache_license}")
    shutil.copy2(apache_license, destination / "LICENSE.Apache-2.0")


def _base_tree_for_airframe(row, base_sdf, motor_sdf, mrs_dir):
    airframe_id = row.get("airframe_id") or "px4_x500"
    if airframe_id in MRS_AIRFRAME_TEMPLATES:
        if not mrs_dir:
            raise ValueError(f"--mrs-dir is required for {airframe_id}")
        return _render_mrs_airframe(mrs_dir, airframe_id), airframe_id
    if airframe_id not in ("px4_x500", "x500"):
        raise ValueError(f"unknown airframe_id: {airframe_id}")

    tree = ET.parse(base_sdf)
    model = _find_first(tree.getroot(), "model")
    motor_model = _find_first(ET.parse(motor_sdf).getroot(), "model")
    for plugin in motor_model.findall("plugin"):
        model.append(copy.deepcopy(plugin))
    return tree, "px4_x500"


def generate_models(
    design_csv,
    output_dir,
    base_sdf,
    base_config=None,
    motor_sdf=None,
    mrs_dir=None,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_sdf = Path(base_sdf)
    motor_sdf = Path(motor_sdf) if motor_sdf else base_sdf.parent.parent / "x500" / "model.sdf"
    with Path(design_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    unique = {}
    for row in rows:
        unique[row["model_name"]] = row

    if any((row.get("airframe_id") or "") in MRS_AIRFRAME_TEMPLATES for row in unique.values()):
        _copy_mrs_resources(mrs_dir, output_dir)

    for model_name, row in sorted(unique.items()):
        tree, airframe_id = _base_tree_for_airframe(row, base_sdf, motor_sdf, mrs_dir)
        root = tree.getroot()
        model = _find_first(root, "model")
        model.attrib["name"] = model_name

        inertial = _find_first(model, "link/inertial")
        mass_elem = _find_first(inertial, "mass")
        inertia_elem = _find_first(inertial, "inertia")
        base_mass = float(mass_elem.text)
        mass_pct = float(row.get("mass_pct") or 0.0)
        rotor_pct = float(row.get("rotor_pct") or 0.0)
        mode = row.get("inertia_mode") or "coupled"

        if mode == "absolute":
            mass_elem.text = str(float(row["mass_kg"]))
            for name, field in (
                ("ixx", "inertia_xx_kg_m2"),
                ("iyy", "inertia_yy_kg_m2"),
                ("izz", "inertia_zz_kg_m2"),
            ):
                _find_first(inertia_elem, name).text = str(float(row[field]))
            for name in ("ixy", "ixz", "iyz"):
                _find_first(inertia_elem, name).text = "0.0"
            pose = inertial.find("pose")
            if pose is None:
                pose = ET.SubElement(inertial, "pose")
            pose.text = "{} {} {} 0 0 0".format(
                float(row.get("com_x_m") or 0.0),
                float(row.get("com_y_m") or 0.0),
                float(row.get("com_z_m") or 0.0),
            )
        elif mode == "coupled":
            factor = 1.0 + mass_pct / 100.0
            mass_elem.text = str(base_mass * factor)
            _scale_inertia(inertia_elem, factor)
        elif mode == "mass_only":
            mass_elem.text = str(base_mass * (1.0 + mass_pct / 100.0))
        elif mode == "inertia_only":
            _scale_inertia(inertia_elem, 1.0 + mass_pct / 100.0)
        elif mode == "payload":
            payload_mass = base_mass * float(row.get("payload_mass_pct") or 0.0) / 100.0
            mass_elem.text = str(base_mass + payload_mass)
            pose = inertial.find("pose")
            if pose is None:
                pose = ET.SubElement(inertial, "pose")
            offset = (
                float(row.get("payload_x") or 0.0),
                float(row.get("payload_y") or 0.0),
                float(row.get("payload_z") or 0.0),
            )
            total_mass = base_mass + payload_mass
            center = tuple(payload_mass * value / total_mass for value in offset)
            pose.text = f"{center[0]} {center[1]} {center[2]} 0 0 0"
            _add_payload_inertia(
                inertia_elem,
                payload_mass,
                *offset,
            )
        else:
            raise ValueError(f"unknown inertia_mode: {mode}")

        for link in model.findall("link"):
            enable_wind = link.find("enable_wind")
            if enable_wind is None:
                enable_wind = ET.SubElement(link, "enable_wind")
            enable_wind.text = "true"

        requested_max_rot = row.get("max_rot_velocity_rad_s")
        for plugin in model.findall("plugin"):
            max_rot = plugin.find("maxRotVelocity")
            if max_rot is not None:
                if requested_max_rot:
                    max_rot.text = str(float(requested_max_rot))
                else:
                    max_rot.text = str(float(max_rot.text) * (1.0 + rotor_pct / 100.0))

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        ET.indent(tree, space="  ")
        tree.write(model_dir / "model.sdf", encoding="unicode", xml_declaration=True)
        _write_model_config(
            model_dir / "model.config",
            model_name,
            f"PX4 Gazebo Harmonic variant based on {airframe_id}.",
        )

    return output_dir


def _wind_plugin(settings):
    plugin = ET.Element(
        "plugin",
        filename="gz-sim-wind-effects-system",
        name="gz::sim::systems::WindEffects",
    )
    ET.SubElement(plugin, "force_approximation_scaling_factor").text = "1"
    horizontal = ET.SubElement(plugin, "horizontal")
    magnitude = ET.SubElement(horizontal, "magnitude")
    ET.SubElement(magnitude, "time_for_rise").text = "2"
    sin = ET.SubElement(magnitude, "sin")
    speed = settings["speed"]
    amplitude = 0.0 if speed == 0 else max(0.0, settings["max_speed"] / speed - 1.0)
    ET.SubElement(sin, "amplitude_percent").text = str(amplitude)
    ET.SubElement(sin, "period").text = "20"
    noise = ET.SubElement(magnitude, "noise", type="gaussian")
    ET.SubElement(noise, "mean").text = "0"
    ET.SubElement(noise, "stddev").text = str(settings["speed_stddev"])
    direction = ET.SubElement(horizontal, "direction")
    ET.SubElement(direction, "time_for_rise").text = "2"
    direction_noise = ET.SubElement(direction, "noise", type="gaussian")
    ET.SubElement(direction_noise, "mean").text = "0"
    ET.SubElement(direction_noise, "stddev").text = str(settings["direction_stddev"])
    vertical = ET.SubElement(plugin, "vertical")
    vertical_noise = ET.SubElement(vertical, "noise", type="gaussian")
    ET.SubElement(vertical_noise, "mean").text = "0"
    ET.SubElement(vertical_noise, "stddev").text = str(settings["speed_stddev"] * 0.25)
    return plugin


def _dryden_plugin(settings):
    """DrydenWind system-plugin element: accepted-model (MIL-STD-1797A) turbulence.

    One plugin carries the mean wind AND the anisotropic Dryden gusts in
    airspeed-relative axes (the plugin default) with altitude-dependent length
    scales. No deterministic sinusoid, no WindEffects. sigma is a per-axis
    (u, v, w) tuple. Requires the DrydenWind .so on GZ_SIM_SYSTEM_PLUGIN_PATH
    (set TDR_GZ_PLUGIN_PATH).
    """
    plugin = ET.Element("plugin", filename="DrydenWind", name="tdr::DrydenWind")
    mx, my, mz = settings.get("mean_wind", (0.0, 0.0, 0.0))
    ET.SubElement(plugin, "mean_wind").text = f"{mx} {my} {mz}"
    su, sv, sw = settings["sigma"]
    ET.SubElement(plugin, "sigma").text = f"{su} {sv} {sw}"
    ET.SubElement(plugin, "min_airspeed").text = str(settings.get("min_airspeed", 2.0))
    ET.SubElement(plugin, "seed").text = str(int(settings.get("seed", 0)))
    return plugin


def _ensure_px4_world_systems(world):
    existing = {plugin.attrib.get("name") for plugin in world.findall("plugin")}
    for filename, name, children in PX4_WORLD_SYSTEM_PLUGINS:
        if name in existing:
            continue
        plugin = ET.Element("plugin", filename=filename, name=name)
        for child_name, child_value in (children or {}).items():
            ET.SubElement(plugin, child_name).text = child_value
        world.insert(0, plugin)


def generate_worlds(output_dir, base_world):
    output_dir = Path(output_dir)
    base_world = Path(base_world)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, settings in DEFAULT_WIND_LEVELS.items():
        tree = ET.parse(base_world)
        root = tree.getroot()
        world = _find_first(root, "world")
        world.attrib["name"] = f"turbulence_{name}"
        for plugin in list(world.findall("plugin")):
            if "WindEffects" in plugin.attrib.get("name", ""):
                world.remove(plugin)
        _ensure_px4_world_systems(world)
        wind = world.find("wind")
        if wind is None:
            wind = ET.SubElement(world, "wind")
        velocity = wind.find("linear_velocity")
        if velocity is None:
            velocity = ET.SubElement(wind, "linear_velocity")
        velocity.text = f"{settings['speed']} 0 0"
        if settings["speed"] > 0:
            world.append(_wind_plugin(settings))
        ET.indent(tree, space="  ")
        tree.write(output_dir / f"turbulence_{name}.sdf", encoding="unicode", xml_declaration=True)
    return output_dir


def _offset_lat_lon(lat, lon, north_m, east_m):
    lat2 = lat + north_m / 111_320.0
    lon2 = lon + east_m / (111_320.0 * math.cos(math.radians(lat)))
    return lat2, lon2


def _simple_item(command, do_jump_id, lat, lon, alt, params_prefix=None):
    params = list(params_prefix or [0, 0, 0, None])
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": alt,
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": command,
        "doJumpId": do_jump_id,
        "frame": 3,
        "params": params + [lat, lon, alt],
        "type": "SimpleItem",
    }


def _plan_template():
    return {
        "fileType": "Plan",
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 5,
            "firmwareType": 12,
            "globalPlanAltitudeMode": 1,
            "hoverSpeed": 5,
            "items": [],
            "plannedHomePosition": [47.3979711, 8.5461637, 488.0],
            "vehicleType": 2,
            "version": 2,
        },
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }


def generate_plans(output_dir, base_plan=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = _plan_template()
    home = template["mission"]["plannedHomePosition"]
    lat0, lon0 = home[0], home[1]
    alt0 = 25.0
    short_leg = 500.0
    main_leg = 1000.0
    lateral = 250.0

    definitions = {
        "smoke": [(0, 0, 5.0, 22), (0, 10, 5.0, 16)],
        "straight": [(0, 0, alt0, 22), (0, short_leg, alt0, 16), (0, main_leg, alt0, 16)],
        "turn": [(0, 0, alt0, 22), (lateral, short_leg, alt0, 16), (0, main_leg, alt0, 16)],
        "climb": [(0, 0, alt0, 22), (0, short_leg, alt0 + 15, 16), (0, main_leg, alt0 + 30, 16)],
        "descent": [(0, 0, alt0 + 30, 22), (0, short_leg, alt0 + 15, 16), (0, main_leg, alt0, 16)],
        "headwind_leg": [(0, 0, alt0, 22), (0, main_leg, alt0, 16)],
        "crosswind_leg": [(0, 0, alt0, 22), (main_leg, 0, alt0, 16)],
    }

    # --- v3 block B1: the same 1 km course rotated through 8 headings ---------------
    # Every v2 flight cruised due east, so "crosswind" and "wind along world +y" are not
    # separable in that data -- and crosswind is the headline v2 effect. Rotating the COURSE
    # while wind direction keeps its own Sobol dimension makes the relative angle
    # (wind_direction - course_heading) identifiable apart from world-frame direction.
    # Heading is measured exactly like wind_direction_deg (ENU, 0 deg = +x = East), so
    # straight_h000 reproduces `straight` waypoint-for-waypoint.
    for heading in range(0, 360, 45):
        radians = math.radians(heading)
        rotated = [(0.0, 0.0, alt0, 22)]
        for distance in (short_leg, main_leg):
            rotated.append((distance * math.sin(radians), distance * math.cos(radians), alt0, 16))
        definitions["straight_h%03d" % heading] = rotated

    # --- v3 block B2: turns ---------------------------------------------------------
    # Dogleg with a parameterised lateral offset; a larger offset means a sharper heading
    # change at the midpoint. This is the demand that should exercise rotational inertia,
    # which came out null on the straight-line design and so cannot yet be called null.
    for lateral_offset in (100, 250, 400):
        definitions["turn_l%03d" % lateral_offset] = [
            (0.0, 0.0, alt0, 22),
            (float(lateral_offset), short_leg, alt0, 16),
            (0.0, main_leg, alt0, 16),
        ]

    # --- v3 block B2: climbs --------------------------------------------------------
    # Steep but LOW: the altitude gain is packed into a short horizontal distance so the
    # vertical rate is high (the demand that should exercise thrust-to-weight) while peak
    # altitude stays under 90 m. Nothing in this pipeline sets GF_* parameters, so the runs
    # must not depend on PX4's vertical-geofence defaults, which are outside our control.
    for tag, (climb_distance, climb_gain) in (("s1", (200.0, 40.0)), ("s2", (120.0, 60.0))):
        definitions["climb_%s" % tag] = [
            (0.0, 0.0, alt0, 22),
            (0.0, climb_distance, alt0 + climb_gain, 16),
            (0.0, main_leg, alt0 + climb_gain, 16),
        ]

    for name, points in definitions.items():
        plan = copy.deepcopy(template)
        items = []
        for index, (north, east, alt, command) in enumerate(points, start=1):
            lat, lon = _offset_lat_lon(lat0, lon0, north, east)
            items.append(_simple_item(command, index, lat, lon, alt))
        plan["mission"]["items"] = items
        plan["mission"]["plannedHomePosition"] = [lat0, lon0, home[2]]
        (output_dir / f"{name}.plan").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    hover = copy.deepcopy(template)
    lat1, lon1 = _offset_lat_lon(lat0, lon0, 0, short_leg)
    lat2, lon2 = _offset_lat_lon(lat0, lon0, 0, main_leg)
    hover["mission"]["items"] = [
        _simple_item(22, 1, lat0, lon0, alt0),
        _simple_item(16, 2, lat1, lon1, alt0),
        _simple_item(19, 3, lat1, lon1, alt0, params_prefix=[20, 0, 0, None]),
        _simple_item(16, 4, lat2, lon2, alt0),
    ]
    (output_dir / "hover.plan").write_text(json.dumps(hover, indent=2), encoding="utf-8")
    return output_dir


def generate_runner(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = """#!/usr/bin/env bash
set -euo pipefail
exec python3 -m tdresearch run-batch "$@"
"""
    path = output_dir / "run_px4_batch.sh"
    path.write_text(script, encoding="utf-8", newline="\n")
    return path


def generate_assets(output_dir, px4_dir="~/PX4-Autopilot"):
    output_dir = Path(output_dir)
    px4_dir = Path(px4_dir).expanduser()
    for name in ("designs", "worlds", "plans", "controllers", "scripts"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    design_dir = generate_designs(output_dir)
    generate_worlds(output_dir / "worlds", px4_dir / "Tools/simulation/gz/worlds/default.sdf")
    generate_plans(output_dir / "plans")
    generate_controller_scripts(output_dir / "controllers", design_dir / "all_experiments.csv")
    generate_runner(output_dir / "scripts")
    return design_dir
