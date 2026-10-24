import asyncio
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


MINIMAL_LOGGER_TOPICS = (
    ("vehicle_status", 0),
    ("vehicle_local_position", 0),
    ("vehicle_local_position_setpoint", 0),
    ("actuator_motors", 0),
)

RUN_STATUS_SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_run_status(case_dir, run_id):
    path = Path(case_dir) / "run_status.json"
    if path.exists():
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(status.get("attempts"), list):
                return status
        except (OSError, ValueError, TypeError):
            pass
    return {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "not_started",
        "attempt_count": 0,
        "failed_attempts": 0,
        "attempts": [],
    }


def _preliminary_failure_classification(phase, evidence):
    if phase in {
        "asset_validation",
        "gazebo_startup",
        "px4_startup",
        "connection",
        "preflight",
        "process_interrupted",
        "telemetry_collection",
    }:
        return "infrastructure"
    if phase in {"takeoff", "route", "landing"} and evidence.get("ulog_valid"):
        return "flight_mechanics_candidate"
    return "indeterminate"


def _record_attempt(case_dir, run_id, attempt_record):
    attempt_record.setdefault("run_id", run_id)
    status = _read_run_status(case_dir, run_id)
    attempts = [
        item for item in status["attempts"]
        if int(item.get("attempt", -1)) != int(attempt_record["attempt"])
    ]
    attempts.append(attempt_record)
    attempts.sort(key=lambda item: int(item["attempt"]))
    failures = [item for item in attempts if item.get("outcome") == "failed"]
    successes = [item for item in attempts if item.get("outcome") == "success"]
    running = [item for item in attempts if item.get("outcome") == "running"]
    status.update(
        {
            "schema_version": RUN_STATUS_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "completed" if successes else ("running" if running else "failed"),
            "attempt_count": len(attempts),
            "failed_attempts": len(failures),
            "succeeded_after_retry": bool(successes and failures),
            "attempts": attempts,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json_atomic(Path(case_dir) / "run_status.json", status)
    attempt_dir = attempt_record.get("attempt_dir")
    if attempt_dir:
        _write_json_atomic(Path(attempt_dir) / "attempt_status.json", attempt_record)
    return status


def _write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _refresh_failure_summary(output_dir, failure_path):
    rows = []
    for status_path in sorted(Path(output_dir).glob("*/run_status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if status.get("status") != "failed" or not status.get("attempts"):
            continue
        attempt = status["attempts"][-1]
        rows.append(
            {
                "run_id": status.get("run_id", status_path.parent.name),
                "attempt": attempt.get("attempt", ""),
                "preliminary_classification": attempt.get(
                    "preliminary_classification", "indeterminate"
                ),
                "failure_phase": attempt.get("failure_phase", ""),
                "error": attempt.get("error", ""),
                "run_status": str(status_path),
            }
        )
    _write_csv_atomic(
        failure_path,
        (
            "run_id", "attempt", "preliminary_classification",
            "failure_phase", "error", "run_status",
        ),
        rows,
    )


async def _first(stream):
    async for value in stream:
        return value
    raise RuntimeError("telemetry stream ended before returning a value")


async def _wait_for_connection(drone, timeout):
    async def watch():
        next_report = time.monotonic() + 5.0
        async for state in drone.core.connection_state():
            if state.is_connected:
                return
            if time.monotonic() >= next_report:
                print("  waiting for MAVSDK connection", flush=True)
                next_report = time.monotonic() + 5.0

    await asyncio.wait_for(watch(), timeout)


async def _wait_until_armable(drone, timeout):
    async def watch():
        next_report = time.monotonic() + 5.0
        async for health in drone.telemetry.health():
            if health.is_armable and health.is_global_position_ok and health.is_home_position_ok:
                return
            if time.monotonic() >= next_report:
                print(
                    "  waiting for armable state "
                    f"(armable={health.is_armable}, global_position={health.is_global_position_ok}, "
                    f"home_position={health.is_home_position_ok})",
                    flush=True,
                )
                next_report = time.monotonic() + 5.0

    await asyncio.wait_for(watch(), timeout)


async def _wait_for_altitude(drone, minimum_altitude, timeout):
    async def watch():
        next_report = time.monotonic() + 5.0
        async for position in drone.telemetry.position():
            if position.relative_altitude_m >= minimum_altitude:
                return
            if time.monotonic() >= next_report:
                print(
                    f"  climbing: altitude={position.relative_altitude_m:.1f} m "
                    f"target>={minimum_altitude:.1f} m",
                    flush=True,
                )
                next_report = time.monotonic() + 5.0

    await asyncio.wait_for(watch(), timeout)


async def _wait_until_landed(drone, timeout):
    from mavsdk.telemetry import LandedState

    async def watch():
        next_report = time.monotonic() + 5.0
        async for state in drone.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                return
            if time.monotonic() >= next_report:
                print(f"  waiting for landing: state={state}", flush=True)
                next_report = time.monotonic() + 5.0

    await asyncio.wait_for(watch(), timeout)


def _horizontal_distance_m(latitude_a, longitude_a, latitude_b, longitude_b):
    north = (latitude_b - latitude_a) * 111_320.0
    mean_latitude = math.radians((latitude_a + latitude_b) / 2.0)
    east = (longitude_b - longitude_a) * 111_320.0 * math.cos(mean_latitude)
    return math.hypot(north, east)


async def _wait_for_position(
    drone,
    latitude,
    longitude,
    relative_altitude,
    timeout,
    label=None,
    horizontal_acceptance_m=15.0,
):
    async def watch():
        next_report = time.monotonic() + 5.0
        async for position in drone.telemetry.position():
            horizontal = _horizontal_distance_m(
                position.latitude_deg,
                position.longitude_deg,
                latitude,
                longitude,
            )
            vertical = abs(position.relative_altitude_m - relative_altitude)
            if horizontal <= horizontal_acceptance_m and vertical <= 2.0:
                return
            if time.monotonic() >= next_report:
                name = f" {label}" if label else ""
                print(
                    f"  enroute{name}: horizontal={horizontal:.1f} m vertical={vertical:.1f} m",
                    flush=True,
                )
                next_report = time.monotonic() + 5.0

    await asyncio.wait_for(watch(), timeout)


async def _apply_parameters(drone, parameters):
    for name, value in parameters.items():
        if isinstance(value, int):
            await drone.param.set_param_int(name, value)
        else:
            await drone.param.set_param_float(name, float(value))


async def run_smoke_flight(
    system_address="udpin://0.0.0.0:14540",
    takeoff_altitude=5.0,
    north_m=20.0,
    east_m=10.0,
    hold_seconds=10.0,
    timeout=60.0,
):
    try:
        from mavsdk import System
    except ImportError as exc:
        raise RuntimeError("smoke-flight requires the mavsdk package") from exc

    drone = System()
    await drone.connect(system_address=system_address)
    await _wait_for_connection(drone, timeout)
    await _wait_until_armable(drone, timeout)
    home = await _first(drone.telemetry.position())
    await drone.action.set_takeoff_altitude(takeoff_altitude)
    await drone.action.arm()

    try:
        await drone.action.takeoff()
        await _wait_for_altitude(drone, takeoff_altitude * 0.8, timeout)
        latitude = home.latitude_deg + north_m / 111_320.0
        longitude = home.longitude_deg + east_m / (
            111_320.0 * math.cos(math.radians(home.latitude_deg))
        )
        await drone.action.goto_location(
            latitude,
            longitude,
            home.absolute_altitude_m + takeoff_altitude,
            0.0,
        )
        await _wait_for_position(drone, latitude, longitude, takeoff_altitude, timeout)
        await asyncio.sleep(hold_seconds)
        await drone.action.return_to_launch()
        await _wait_until_landed(drone, timeout)
    except Exception:
        try:
            await drone.action.land()
        except Exception:
            pass
        raise


def smoke_flight(**kwargs):
    asyncio.run(run_smoke_flight(**kwargs))


def _load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    mission = plan["mission"]
    items = mission["items"]
    if not items:
        raise ValueError(f"{path} contains no mission items")
    return mission["plannedHomePosition"], items


# Datalink-loss timeouts are evaluated on PX4's SIMULATED clock. Keep them large
# enough that wall-clock heartbeat jitter is harmless at every sim speed factor
# used here (takeoff 2x, cruise 10x).
_DATALINK_FAILSAFE_PARAMS = {
    "COM_DL_LOSS_T": 300,
    "COM_OBC_LOSS_T": 300.0,
}


async def _fly_plan(
    drone,
    plan_path,
    parameters,
    timeout,
    land_after_route=False,
    after_takeoff=None,
    waypoint_acceptance_m=15.0,
    takeoff_timeout=90.0,
    waypoint_timeout=None,
    phase_callback=None,
    hold_sim_speed_factor=1.0,
    route_parameters=None,
):
    def set_phase(value):
        if phase_callback is not None:
            phase_callback(value)

    set_phase("connection")

    print("  connecting to PX4 over MAVSDK", flush=True)
    await _wait_for_connection(drone, timeout)
    # PX4's rcS sets COM_DL_LOSS_T = PX4_SIM_SPEED_FACTOR * 10 at boot, using the
    # takeoff factor. We raise Gazebo to the cruise factor after takeoff without
    # PX4 re-scaling that timeout, so at 10x the tolerance collapses to ~2 s of
    # wall-clock while MAVSDK heartbeats arrive at 1 Hz wall-clock. Widen it here,
    # before the armable wait, so heartbeat jitter cannot raise a spurious
    # "No connection to the ground control station" preflight failure or a
    # mid-route datalink failsafe.
    await _apply_parameters(drone, _DATALINK_FAILSAFE_PARAMS)
    print("  connected; waiting until vehicle is armable", flush=True)
    set_phase("preflight")
    await _wait_until_armable(drone, timeout)
    print(f"  applying {len(parameters)} controller parameters", flush=True)
    await _apply_parameters(drone, parameters)
    planned_home, items = _load_plan(plan_path)
    home = await _first(drone.telemetry.position())
    takeoff_altitude = float(items[0]["params"][6])
    print(f"  arming and taking off to {takeoff_altitude:.1f} m", flush=True)
    await drone.action.set_takeoff_altitude(takeoff_altitude)
    # The MIS_TAKEOFF_ALT write can lose a PX4 startup race -- the vehicle then takes off to
    # PX4's 2.5 m default ("Using default takeoff altitude: 2.5 m") and _wait_for_altitude
    # times out in phase=takeoff. The race fires reliably when a large controller-param batch
    # precedes it: controller_mode=autotune applies 21 params and failed ~73% at takeoff,
    # while fixed applies 9 and flew (identical MPC vertical loop, so it is not the gains).
    # Read MIS_TAKEOFF_ALT back and re-send until it sticks (same remedy as autotune_probe.py).
    for _ in range(5):
        try:
            got = await drone.param.get_param_float("MIS_TAKEOFF_ALT")
        except Exception:
            break
        if abs(got - takeoff_altitude) < 0.5:
            break
        print(f"  MIS_TAKEOFF_ALT read back {got:.1f}, re-sending", flush=True)
        await _apply_parameters(drone, {"MIS_TAKEOFF_ALT": takeoff_altitude})
        await asyncio.sleep(0.5)  # let the write commit before re-reading (else the
        #                           readback races the same way the takeoff does)
    await drone.action.arm()

    try:
        set_phase("takeoff")
        await drone.action.takeoff()
        _takeoff_deadline = min(timeout, takeoff_timeout) if takeoff_timeout else timeout
        try:
            await _wait_for_altitude(drone, takeoff_altitude * 0.8, _takeoff_deadline)
        except asyncio.TimeoutError:
            # Takeoff stalled at PX4's 2.5 m default despite the MIS_TAKEOFF_ALT re-send
            # (the write can still lose the startup race). Deterministically climb the rest
            # with a plain goto at the current spot. Two bugs made the old single-shot
            # recovery fail (and leave the flight stuck in phase=takeoff until the overall
            # timeout, which is the bulk of the Saltelli unflyable tail):
            #   1. it reused `_takeoff_deadline`, already spent by the first wait, so the
            #      climb had little/no time budget left;
            #   2. it ran once with no retry and re-raised nothing cleanly.
            # Fix: give each climb attempt a FRESH budget and retry a few times; re-raise
            # only if the vehicle genuinely cannot reach altitude (e.g. a real high-wind
            # unflyable corner), so those still fail honestly instead of hanging.
            climb_budget = takeoff_timeout if takeoff_timeout else 90.0
            reached = False
            for attempt in range(3):
                pos = await _first(drone.telemetry.position())
                if pos.relative_altitude_m >= takeoff_altitude * 0.8:
                    reached = True
                    break
                target_amsl = pos.absolute_altitude_m + (takeoff_altitude - pos.relative_altitude_m)
                print(f"  takeoff stalled at {pos.relative_altitude_m:.1f} m "
                      f"(recovery {attempt + 1}/3); goto same spot at "
                      f"{takeoff_altitude:.0f} m rel", flush=True)
                await drone.action.goto_location(pos.latitude_deg, pos.longitude_deg,
                                                 target_amsl, float("nan"))
                try:
                    await _wait_for_altitude(drone, takeoff_altitude * 0.8, climb_budget)
                    reached = True
                    break
                except asyncio.TimeoutError:
                    continue
            if not reached:
                raise
        print("  takeoff altitude reached", flush=True)
        if after_takeoff is not None:
            after_takeoff()
        if route_parameters:
            print(f"  applying {len(route_parameters)} route parameters", flush=True)
            await _apply_parameters(drone, route_parameters)
        set_phase("route")
        for index, item in enumerate(items[1:], start=2):
            params = item["params"]
            latitude = home.latitude_deg + float(params[4]) - float(planned_home[0])
            longitude = home.longitude_deg + float(params[5]) - float(planned_home[1])
            relative_altitude = float(params[6])
            print(
                f"  waypoint {index}/{len(items)}: command={item.get('command', 16)} "
                f"altitude={relative_altitude:.1f} m",
                flush=True,
            )
            await drone.action.goto_location(
                latitude,
                longitude,
                home.absolute_altitude_m + relative_altitude,
                0.0,
            )
            await _wait_for_position(
                drone,
                latitude,
                longitude,
                relative_altitude,
                min(timeout, waypoint_timeout) if waypoint_timeout else timeout,
                label=f"to waypoint {index}/{len(items)}",
                horizontal_acceptance_m=waypoint_acceptance_m,
            )
            if int(item.get("command", 16)) == 19:
                # params[0] is SIM seconds; asyncio.sleep runs in wall time, so divide by
                # the world speed factor (route phase runs at the cruise factor).
                hold_s = float(params[0] or 0.0)
                print(f"  holding at waypoint {index} for {hold_s:.1f} sim s", flush=True)
                await asyncio.sleep(hold_s / max(hold_sim_speed_factor, 1e-6))
        if land_after_route:
            print("  route complete; returning to launch", flush=True)
            set_phase("landing")
            await drone.action.return_to_launch()
            await _wait_until_landed(drone, timeout)
            print("  landed", flush=True)
        else:
            print("  route complete; stopping before RTL/landing", flush=True)
        set_phase("route_complete")
    except Exception:
        try:
            print("  flight failed; commanding land", flush=True)
            await drone.action.land()
            await _wait_until_landed(drone, min(timeout, 30.0))
        except Exception:
            pass
        raise


def _terminate_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _wait_for_world(environment, world_name, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Gazebo exited before its world became available")
        try:
            result = subprocess.run(
                ["gz", "topic", "-l"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            time.sleep(0.5)
            continue
        if f"/world/{world_name}/clock" in result.stdout:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Gazebo world {world_name} did not start within {timeout} seconds")


def _set_world_speed(environment, world_name, speed_factor):
    if not speed_factor or speed_factor <= 0:
        return
    result = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{world_name}/set_physics",
            "--reqtype",
            "gz.msgs.Physics",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "5000",
            "--req",
            f"real_time_factor: {speed_factor}",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode == 0:
        print(f"  set Gazebo real_time_factor={speed_factor}", flush=True)
    else:
        print(
            f"  warning: failed to set Gazebo real_time_factor={speed_factor}: "
            f"{result.stderr.strip() or result.stdout.strip()}",
            flush=True,
        )


# Dryden low-altitude anisotropy: sigma_w/sigma_u = (0.177 + 0.000823*h_ft)^0.4,
# evaluated at the ~25 m (82 ft) mission cruise altitude (MIL-STD-1797A / MIL-F-8785C).
DRYDEN_SIGMA_W_RATIO = 0.57


def _case_world(base_world, destination, model_name, instance, design_row=None):
    tree = ET.parse(base_world)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"{base_world} does not contain a world")
    world_name = destination.stem.replace("+", "p").replace("-", "m")
    world.attrib["name"] = world_name
    design_row = design_row or {}
    if design_row.get("mean_wind_m_s") not in (None, ""):
        from .generate import _wind_plugin, _dryden_plugin

        mean_speed = float(design_row["mean_wind_m_s"])
        turbulence_rms = float(design_row.get("turbulence_rms_m_s") or 0.0)
        direction_deg = float(design_row.get("wind_direction_deg") or 0.0)
        direction = math.radians(direction_deg)
        turb_model = (design_row.get("turb_model") or "white").lower()
        for plugin in list(world.findall("plugin")):
            if "WindEffects" in plugin.attrib.get("name", ""):
                world.remove(plugin)
        wind = world.find("wind")
        if wind is None:
            wind = ET.SubElement(world, "wind")
        velocity = wind.find("linear_velocity")
        if velocity is None:
            velocity = ET.SubElement(wind, "linear_velocity")
        velocity.text = f"{mean_speed * math.cos(direction)} {mean_speed * math.sin(direction)} 0"
        seed = int(design_row.get("wind_seed") or 0)
        if mean_speed > 0.0 or turbulence_rms > 0.0:
            if turb_model == "dryden":
                # Accepted-model (MIL-STD-1797A) turbulence: a SINGLE DrydenWind plugin
                # carries the mean wind + anisotropic Dryden gusts (sigma_u=sigma_v=rms,
                # sigma_w = DRYDEN_SIGMA_W_RATIO * rms, the low-altitude ratio) in
                # airspeed-relative axes with altitude length scales. NO WindEffects and
                # NO deterministic sinusoid. Paired vs the v2 white control, the delta
                # reads as "crude v2 turbulence model -> accepted Dryden model".
                mean_vec = (
                    mean_speed * math.cos(direction),
                    mean_speed * math.sin(direction),
                    0.0,
                )
                world.append(
                    _dryden_plugin(
                        {
                            "mean_wind": mean_vec,
                            "sigma": (
                                turbulence_rms,
                                turbulence_rms,
                                DRYDEN_SIGMA_W_RATIO * turbulence_rms,
                            ),
                            "min_airspeed": 2.0,
                            "seed": seed,
                        }
                    )
                )
            else:
                world.append(
                    _wind_plugin(
                        {
                            "speed": mean_speed,
                            "max_speed": max(mean_speed, mean_speed + 3.0 * turbulence_rms),
                            "speed_stddev": turbulence_rms,
                            "direction_stddev": 0.0,
                            "seed": seed,
                        }
                    )
                )
    include = ET.SubElement(world, "include")
    ET.SubElement(include, "uri").text = f"model://{model_name}"
    ET.SubElement(include, "name").text = f"{model_name}_{instance}"
    ET.SubElement(include, "pose").text = "0 0 0 0 0 0"
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=True)
    return world_name


def _newest_ulog(log_root, known_logs, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = set(Path(log_root).rglob("*.ulg")) - known_logs
        if candidates:
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            previous_size = -1
            for _ in range(20):
                size = path.stat().st_size
                if size == previous_size:
                    return path
                previous_size = size
                time.sleep(0.25)
            return path
        time.sleep(0.5)
    raise FileNotFoundError("PX4 did not create a new ULog for the completed run")


def _copy_new_ulog(log_root, known_logs, destination, timeout=15, overwrite=False):
    if destination.exists() and not overwrite:
        return destination
    ulog = _newest_ulog(log_root, known_logs, timeout=timeout)
    shutil.copy2(ulog, destination)
    return destination


def _validate_gzip_copy(path, expected_size, expected_digest):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    if size != expected_size or digest.digest() != expected_digest:
        raise ValueError(f"gzip content does not match source ULog: {path}")


def _compress_ulog_safely(path):
    """Atomically gzip a ULog, preserving the source on every failure."""
    path = Path(path)
    destination = path.with_name(f"{path.name}.gz")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as source, gzip.GzipFile(
                filename=path.name,
                mode="wb",
                compresslevel=6,
                fileobj=output,
                mtime=0,
            ) as compressed:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                    compressed.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        expected_digest = digest.digest()
        _validate_gzip_copy(temporary, size, expected_digest)
        os.replace(temporary, destination)
        temporary = None
        _validate_gzip_copy(destination, size, expected_digest)
        path.unlink()
        return destination, None
    except Exception as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        warning = f"{type(exc).__name__}: {exc}"
        try:
            print(f"  compression warning for {path}: {warning}", flush=True)
        except OSError:
            pass
        return path, warning


def _ulog_complete(path):
    path = Path(path)
    if not path.exists():
        return False
    if path.name.lower().endswith(".ulg.gz"):
        try:
            size = 0
            with gzip.open(path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    size += len(chunk)
            return size > 1024
        except (OSError, EOFError):
            return False
    return path.stat().st_size > 1024


def _canonical_ulog_paths(path):
    path = Path(path)
    if path.name.lower().endswith(".ulg.gz"):
        return path.with_suffix(""), path
    return path, path.with_name(f"{path.name}.gz")


def _validate_ulog(path, min_samples=10, min_duration_s=1.0):
    from .metrics import read_telemetry

    run = read_telemetry(path)
    duration = run.times[-1] - run.times[0] if len(run.times) > 1 else 0.0
    if len(run.actual) < min_samples:
        raise ValueError(f"ULog has only {len(run.actual)} usable samples")
    if duration < min_duration_s:
        raise ValueError(f"ULog duration is only {duration:.1f}s")
    return {"samples": len(run.actual), "duration_s": duration}


def _acquire_case_lock(case_dir, stale_seconds):
    case_dir.mkdir(parents=True, exist_ok=True)
    lock_path = case_dir / "run.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags)
    except FileExistsError:
        if stale_seconds and stale_seconds > 0:
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_seconds:
                lock_path.unlink(missing_ok=True)
                fd = os.open(lock_path, flags)
            else:
                return None
        else:
            return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": time.time()}, handle)
        handle.write("\n")
    return lock_path


def _release_case_lock(lock_path):
    if lock_path:
        Path(lock_path).unlink(missing_ok=True)


def _write_logger_topics(path, topics):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for topic in topics:
        if isinstance(topic, str):
            lines.append(topic)
        else:
            name, interval_ms = topic
            lines.append(f"{name} {interval_ms}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _px4_runtime_root(px4_rootfs, runtime_dir, run_id, instance, minimal_logging=True):
    if runtime_dir is None:
        return px4_rootfs if instance == 0 else px4_rootfs / str(instance)
    runtime_root = Path(runtime_dir).expanduser().resolve() / f"{run_id}_i{instance}_{os.getpid()}"
    runtime_root.mkdir(parents=True, exist_ok=True)
    px4_build = px4_rootfs.parent
    etc_dir = runtime_root / "etc"
    etc_dir.mkdir(exist_ok=True)
    for name in ("extras", "init.d", "init.d-posix"):
        source = px4_build / "etc" / name
        target = etc_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
    test_data = runtime_root / "test_data"
    if test_data.exists() or test_data.is_symlink():
        test_data.unlink()
    test_data.symlink_to((px4_rootfs / "test_data").resolve())
    if minimal_logging:
        _write_logger_topics(etc_dir / "logging" / "logger_topics.txt", MINIMAL_LOGGER_TOPICS)
    return runtime_root


def _run_case(
    row,
    px4_dir,
    assets_dir,
    output_dir,
    instance,
    timeout,
    gui,
    land_after_route=False,
    validate_ulog=True,
    px4_runtime_dir="/tmp/px4",
    keep_px4_runtime=False,
    minimal_logging=True,
    sim_speed_factor=10.0,
    takeoff_sim_speed_factor=None,
    waypoint_acceptance_m=15.0,
    takeoff_timeout=90.0,
    waypoint_timeout=None,
    attempt=1,
    attempt_context=None,
):
    attempt_context = attempt_context if attempt_context is not None else {}
    run_id = row["run_id"]
    model_name = row["model_name"]
    wind_level = row.get("wind_level") or "moderate"
    plan_name = row.get("plan") or "straight"
    controller_mode = row.get("controller_mode") or "fixed"
    case_dir = output_dir / run_id
    case_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = case_dir / "attempts" / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_context.update(
        {
            "attempt": attempt,
            "attempt_dir": str(attempt_dir),
            "failure_phase": "asset_validation",
            "ulog_path": "",
            "ulog_valid": False,
            "ulog_samples": "",
            "ulog_duration_s": "",
        }
    )


    model_dir = assets_dir / "models" / model_name
    base_world = assets_dir / "worlds" / f"turbulence_{wind_level}.sdf"
    plan_path = assets_dir / "plans" / f"{plan_name}.plan"
    controller_path = assets_dir / "controllers" / f"{model_name}_{controller_mode}.json"
    for required in (model_dir / "model.sdf", base_world, plan_path, controller_path):
        if not required.exists():
            raise FileNotFoundError(f"missing generated asset: {required}")

    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    # Optional per-route speed columns (v4 delivery routes). Absent or empty columns leave
    # the controller untouched, so v1/v2/v3 designs fly byte-identically. Applied AFTER
    # takeoff, not preflight: growing the preflight batch 9->13 params made the
    # MIS_TAKEOFF_ALT startup race fire (v4 canary: "Using default takeoff altitude:
    # 2.5 m"), and these only matter once the route starts.
    route_parameters = {}
    for column, parameter in (
        ("cruise_speed_m_s", "MPC_XY_CRUISE"),
        ("xy_vel_max_m_s", "MPC_XY_VEL_MAX"),
        ("climb_rate_m_s", "MPC_Z_VEL_MAX_UP"),
        ("descent_rate_m_s", "MPC_Z_VEL_MAX_DN"),
    ):
        value = row.get(column)
        if value not in (None, ""):
            route_parameters[parameter] = float(value)
    metadata_path = case_dir / "metadata.json"
    _write_json_atomic(metadata_path, row)
    _write_json_atomic(attempt_dir / "design_row.json", row)
    world_path = attempt_dir / f"{run_id}.sdf"
    world_name = _case_world(base_world, world_path, model_name, instance, row)
    px4_build = px4_dir / "build" / "px4_sitl_default"
    px4_binary = px4_build / "bin" / "px4"
    px4_rootfs = px4_build / "rootfs"
    instance_rootfs = _px4_runtime_root(
        px4_rootfs,
        px4_runtime_dir,
        run_id,
        instance,
        minimal_logging=minimal_logging,
    )
    log_root = instance_rootfs
    if not px4_binary.exists():
        raise FileNotFoundError(f"PX4 SITL is not built: {px4_binary}")

    environment = os.environ.copy()
    resource_paths = [
        str(assets_dir / "models"),
        str(px4_dir / "Tools/simulation/gz/models"),
        str(assets_dir / "worlds"),
        str(px4_dir / "Tools/simulation/gz/worlds"),
    ]
    environment["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(resource_paths)
    plugin_paths = [str(px4_build / "src/modules/simulation/gz_plugins")]
    # Optional extra system-plugin dir (e.g. the DrydenWind .so on NFS, dlopen-able by
    # every host's identical px4-u24 container). Empty by default -> v1/v2/v3 unchanged.
    extra_plugin_path = os.environ.get("TDR_GZ_PLUGIN_PATH")
    if extra_plugin_path:
        plugin_paths.append(extra_plugin_path)
    environment["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(plugin_paths)
    environment["GZ_SIM_SERVER_CONFIG_PATH"] = str(px4_dir / "src/modules/simulation/gz_bridge/server.config")
    environment["GZ_PARTITION"] = f"tdresearch_{os.getpid()}_{instance}"
    environment.update(
        {
            "PX4_SYS_AUTOSTART": "4001",
            "PX4_GZ_STANDALONE": "1",
            "PX4_GZ_MODEL_NAME": f"{model_name}_{instance}",
            "PX4_GZ_WORLD": world_name,
            "PX4_PARAM_SDLOG_MODE": "0",
        }
    )
    startup_speed_factor = takeoff_sim_speed_factor or sim_speed_factor
    if startup_speed_factor and startup_speed_factor > 0:
        environment["PX4_SIM_SPEED_FACTOR"] = str(startup_speed_factor)
    for name, value in controller.get("simulator_parameters", {}).items():
        environment[f"PX4_PARAM_{name}"] = str(value)

    known_logs = set((log_root / "log").rglob("*.ulg")) if log_root.exists() else set()
    gazebo_log = (attempt_dir / "gazebo.log").open("w", encoding="utf-8")
    px4_log = (attempt_dir / "px4.log").open("w", encoding="utf-8")
    gazebo = None
    px4 = None
    flight_error = None
    try:
        gazebo_command = ["gz", "sim", "-r", "-s"]
        attempt_context["failure_phase"] = "gazebo_startup"
        if row.get("wind_seed") not in (None, ""):
            gazebo_command.extend(["--seed", str(int(row["wind_seed"]))])
        gazebo_command.append(str(world_path))
        gazebo = subprocess.Popen(
            gazebo_command,
            env=environment,
            stdout=gazebo_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_for_world(environment, world_name, gazebo, min(timeout, 30))
        initial_speed_factor = takeoff_sim_speed_factor or sim_speed_factor
        _set_world_speed(environment, world_name, initial_speed_factor)
        if gui:
            subprocess.Popen(["gz", "sim", "-g"], env=environment, start_new_session=True)
        px4_command = [str(px4_binary), "-d", "-i", str(instance)]
        attempt_context["failure_phase"] = "px4_startup"
        if px4_runtime_dir is not None:
            px4_command = [
                str(px4_binary),
                "-d",
                str(px4_build / "etc"),
                "-i",
                str(instance),
                "-w",
                str(instance_rootfs),
            ]
        px4 = subprocess.Popen(
            px4_command,
            cwd=instance_rootfs if px4_runtime_dir is not None else px4_rootfs,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=px4_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        drone_port = 14540 + instance
        mavsdk_server_port = 50050 + instance

        async def fly():
            from mavsdk import System

            drone = System(port=mavsdk_server_port)
            try:
                await asyncio.wait_for(
                    drone.connect(system_address=f"udpin://0.0.0.0:{drone_port}"),
                    timeout=min(timeout, 60.0),
                )
                def switch_to_cruise_speed():
                    if (
                        takeoff_sim_speed_factor
                        and sim_speed_factor
                        and abs(takeoff_sim_speed_factor - sim_speed_factor) > 1e-6
                    ):
                        _set_world_speed(environment, world_name, sim_speed_factor)

                await _fly_plan(
                    drone,
                    plan_path,
                    controller.get("parameters", {}),
                    timeout,
                    route_parameters=route_parameters,
                    hold_sim_speed_factor=sim_speed_factor or 1.0,
                    land_after_route=land_after_route,
                    after_takeoff=switch_to_cruise_speed,
                    waypoint_acceptance_m=waypoint_acceptance_m,
                    takeoff_timeout=takeoff_timeout,
                    waypoint_timeout=waypoint_timeout,
                    phase_callback=lambda value: attempt_context.update(
                        {"failure_phase": value}
                    ),
                )
            finally:
                server_process = getattr(drone, "_server_process", None)
                stop_server = getattr(drone, "_stop_mavsdk_server", None)
                if stop_server is not None:
                    stop_server()
                if server_process is not None:
                    try:
                        server_process.wait(timeout=5)
                    except Exception:
                        pass

        try:
            asyncio.run(fly())
        except Exception as exc:
            flight_error = exc
    finally:
        _terminate_process(px4)
        _terminate_process(gazebo)
        gazebo_log.close()
        attempt_context["px4_exit_code"] = px4.poll() if px4 is not None else ""
        attempt_context["gazebo_exit_code"] = gazebo.poll() if gazebo is not None else ""
        px4_log.close()
    try:
        copied_ulog = None
        canonical_ulog = None
        ulog_error = None
        try:
            if flight_error is None:
                attempt_context["failure_phase"] = "telemetry_collection"
            copied_ulog = _copy_new_ulog(
                log_root / "log",
                known_logs,
                attempt_dir / f"{run_id}.ulg",
                overwrite=True,
            )
            attempt_context["ulog_path"] = str(copied_ulog)
            if validate_ulog:
                validation = _validate_ulog(copied_ulog)
                attempt_context.update(
                    {
                        "ulog_valid": True,
                        "ulog_samples": validation["samples"],
                        "ulog_duration_s": validation["duration_s"],
                    }
                )
            else:
                attempt_context["ulog_valid"] = True
        except Exception as exc:
            ulog_error = exc
            attempt_context["ulog_error"] = f"{type(exc).__name__}: {exc}"
        if flight_error is None and ulog_error is None and copied_ulog is not None:
            canonical_ulog = case_dir / f"{run_id}.ulg"
            try:
                shutil.copy2(copied_ulog, canonical_ulog)
                attempt_context["failure_phase"] = "completed"
            except Exception as exc:
                ulog_error = exc
                canonical_ulog = None
                attempt_context["ulog_error"] = f"{type(exc).__name__}: {exc}"
        if copied_ulog is not None and copied_ulog.exists():
            copied_ulog, warning = _compress_ulog_safely(copied_ulog)
            attempt_context["ulog_path"] = str(copied_ulog)
            if warning:
                attempt_context.setdefault("compression_warnings", []).append(warning)
        if canonical_ulog is not None and canonical_ulog.exists():
            canonical_ulog, warning = _compress_ulog_safely(canonical_ulog)
            attempt_context["canonical_ulog_path"] = str(canonical_ulog)
            if warning:
                attempt_context.setdefault("compression_warnings", []).append(warning)
        if flight_error is not None:
            raise flight_error
        if ulog_error is not None:
            raise ulog_error
    finally:
        if px4_runtime_dir is not None and not keep_px4_runtime:
            shutil.rmtree(instance_rootfs, ignore_errors=True)


def _select_chunk(rows, chunk_index, chunk_count):
    base, remainder = divmod(len(rows), chunk_count)
    start = chunk_index * base + min(chunk_index, remainder)
    size = base + (chunk_index < remainder)
    return rows[start : start + size]


def _attempt_budget_used(status):
    attempts = status.get("attempts", [])
    recorded_count = int(status.get("attempt_count", 0) or 0)
    highest_attempt = max(
        (int(item.get("attempt", 0) or 0) for item in attempts),
        default=0,
    )
    return max(recorded_count, len(attempts), highest_attempt)


def run_batch(
    design_csv,
    px4_dir="~/PX4-Autopilot",
    assets_dir="generated",
    output_dir="runs",
    instance=1,
    limit=None,
    run_id=None,
    timeout=360.0,
    gui=False,
    failures_path=None,
    overwrite=False,
    land_after_route=False,
    retries=2,
    skip_existing=True,
    claim=True,
    lock_stale_seconds=3600.0,
    start_index=None,
    count=None,
    chunk_index=None,
    chunk_count=None,
    validate_ulog=True,
    px4_runtime_dir="/tmp/px4",
    keep_px4_runtime=False,
    minimal_logging=True,
    sim_speed_factor=10.0,
    takeoff_sim_speed_factor=2.0,
    fallback_sim_speed_factor=None,
    fallback_takeoff_sim_speed_factor=1.0,
    waypoint_acceptance_m=15.0,
    takeoff_timeout=90.0,
    waypoint_timeout=None,
):
    px4_dir = Path(px4_dir).expanduser().resolve()
    assets_dir = Path(assets_dir).resolve()
    output_dir = Path(output_dir).resolve()
    with Path(design_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if run_id:
        rows = [row for row in rows if row.get("run_id") == run_id]
        if not rows:
            raise ValueError(f"run id not found in design: {run_id}")
    if chunk_count is not None or chunk_index is not None:
        if chunk_count is None or chunk_index is None:
            raise ValueError("--chunk-index and --chunk-count must be used together")
        if chunk_count <= 0:
            raise ValueError("--chunk-count must be positive")
        if chunk_index < 0 or chunk_index >= chunk_count:
            raise ValueError("--chunk-index is zero-based and must be less than --chunk-count")
        rows = _select_chunk(rows, chunk_index, chunk_count)
    if start_index is not None:
        if start_index < 1:
            raise ValueError("--start-index is 1-based and must be positive")
        rows = rows[start_index - 1 :]
    if count is not None:
        if count < 0:
            raise ValueError("--count must be non-negative")
        rows = rows[:count]
    if limit is not None:
        rows = rows[:limit]
    if overwrite:
        for row in rows:
            row["_overwrite_log"] = "1"
    if not rows:
        raise ValueError("design contains no selected runs")

    failure_path = Path(failures_path).resolve() if failures_path else output_dir / "failures.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    _refresh_failure_summary(output_dir, failure_path)
    failures = []
    completed = 0
    skipped = 0
    skipped_terminal_failed = 0
    skipped_budget_exhausted = 0
    locked = 0
    max_attempts = retries + 1
    explicit_rerun = overwrite or not skip_existing
    for index, row in enumerate(rows, start=1):
        run_id = row["run_id"]
        case_dir = output_dir / run_id
        destination = case_dir / f"{run_id}.ulg"
        print(f"[{index}/{len(rows)}] {run_id}", flush=True)
        existing_destination = None
        existing_error = None
        if skip_existing and not overwrite:
            for candidate in _canonical_ulog_paths(destination):
                if not _ulog_complete(candidate):
                    continue
                try:
                    if validate_ulog:
                        _validate_ulog(candidate)
                    existing_destination = candidate
                    break
                except Exception as exc:
                    existing_error = exc
        if existing_destination is not None:
            try:
                skipped += 1
                print(f"  skipped: existing ULog {existing_destination}", flush=True)
                status = _read_run_status(case_dir, run_id)
                if not status["attempts"]:
                    status = _record_attempt(
                        case_dir,
                        run_id,
                        {
                            "attempt": 1,
                            "started_at_utc": "",
                            "finished_at_utc": _utc_now(),
                            "outcome": "success",
                            "failure_phase": "",
                            "preliminary_classification": "success",
                            "final_classification": "success",
                            "classification_notes": "Imported from an existing validated ULog.",
                            "error": "",
                            "ulog_path": str(existing_destination),
                            "canonical_ulog_path": str(existing_destination),
                            "ulog_valid": True,
                        },
                    )
                status["canonical_ulog_path"] = str(existing_destination)
                _write_json_atomic(case_dir / "run_status.json", status)
                metadata = dict(row)
                metadata.update({key: status[key] for key in (
                    "attempt_count", "failed_attempts", "succeeded_after_retry"
                )})
                metadata["canonical_ulog_path"] = str(existing_destination)
                _write_json_atomic(case_dir / "metadata.json", metadata)
                continue
            except Exception as exc:
                print(f"  existing ULog failed validation; rerunning: {type(exc).__name__}: {exc}", flush=True)
        elif existing_error is not None:
            print(
                "  existing ULog failed validation; rerunning: "
                f"{type(existing_error).__name__}: {existing_error}",
                flush=True,
            )

        prior_status = _read_run_status(case_dir, run_id)
        if (
            not explicit_rerun
            and prior_status.get("status") == "failed"
            and _attempt_budget_used(prior_status) >= max_attempts
        ):
            skipped_terminal_failed += 1
            print(
                "  skipped: terminal failure "
                f"(attempt_count={prior_status.get('attempt_count', 0)}, "
                f"max_attempts={max_attempts})",
                flush=True,
            )
            continue

        lock_path = None
        if claim:
            lock_path = _acquire_case_lock(case_dir, lock_stale_seconds)
            if lock_path is None:
                locked += 1
                print("  skipped: another worker holds the run lock", flush=True)
                continue

        try:
            last_error = None
            prior_status = _read_run_status(case_dir, run_id)
            for interrupted in list(prior_status["attempts"]):
                if interrupted.get("outcome") != "running":
                    continue
                interrupted = dict(interrupted)
                interrupted.update(
                    {
                        "finished_at_utc": _utc_now(),
                        "outcome": "failed",
                        "failure_phase": "process_interrupted",
                        "preliminary_classification": "infrastructure",
                        "final_classification": "unreviewed",
                        "classification_notes": "",
                        "error": "Process interrupted before a terminal attempt status was recorded.",
                    }
                )
                _record_attempt(case_dir, run_id, interrupted)
            prior_status = _read_run_status(case_dir, run_id)
            base_attempt = _attempt_budget_used(prior_status)
            if not explicit_rerun and base_attempt >= max_attempts:
                if prior_status.get("status") == "failed":
                    skipped_terminal_failed += 1
                    label = "terminal failure"
                else:
                    skipped_budget_exhausted += 1
                    label = f"attempt budget exhausted with status={prior_status.get('status')}"
                print(
                    f"  skipped: {label} "
                    f"(attempt_count={prior_status.get('attempt_count', 0)}, "
                    f"max_attempts={max_attempts})",
                    flush=True,
                )
                _refresh_failure_summary(output_dir, failure_path)
                continue
            attempts_to_launch = (
                max_attempts
                if explicit_rerun
                else max_attempts - base_attempt
            )
            last_attempt_number = base_attempt
            for attempt_index in range(1, attempts_to_launch + 1):
                attempt_number = base_attempt + attempt_index
                last_attempt_number = attempt_number
                attempt_row = dict(row)
                if explicit_rerun or attempt_number > 1:
                    attempt_row["_overwrite_log"] = "1"
                attempt_sim_speed_factor = sim_speed_factor
                attempt_takeoff_sim_speed_factor = takeoff_sim_speed_factor
                # Use the slow fallback profile on only the final attempt of the
                # budget. Earlier retries re-run at full speed (most transient
                # failures clear on a plain re-run), so the expensive realtime
                # attempt is a genuine last resort rather than paid every retry.
                if max_attempts >= 2 and attempt_number == max_attempts:
                    attempt_sim_speed_factor = fallback_sim_speed_factor or sim_speed_factor
                    attempt_takeoff_sim_speed_factor = (
                        fallback_takeoff_sim_speed_factor
                        if fallback_takeoff_sim_speed_factor is not None
                        else takeoff_sim_speed_factor
                    )
                evidence = {
                    "attempt": attempt_number,
                    "started_at_utc": _utc_now(),
                    "attempt_dir": str(
                        case_dir / "attempts" / f"attempt_{attempt_number:03d}"
                    ),
                    "outcome": "running",
                    "final_classification": "pending",
                    "takeoff_sim_speed_factor": (
                        attempt_takeoff_sim_speed_factor or attempt_sim_speed_factor
                    ),
                    "cruise_sim_speed_factor": attempt_sim_speed_factor,
                }
                _record_attempt(case_dir, run_id, evidence)
                try:
                    if attempt_number > 1:
                        if explicit_rerun and base_attempt >= max_attempts:
                            print(
                                f"  explicit rerun override: attempt {attempt_number}",
                                flush=True,
                            )
                        else:
                            print(
                                f"  retry {attempt_number - 1}/{retries}",
                                flush=True,
                            )
                    print(
                        "  speed profile: "
                        f"takeoff={attempt_takeoff_sim_speed_factor or attempt_sim_speed_factor}x "
                        f"cruise={attempt_sim_speed_factor}x",
                        flush=True,
                    )
                    _run_case(
                        attempt_row,
                        px4_dir,
                        assets_dir,
                        output_dir,
                        instance,
                        timeout,
                        gui,
                        land_after_route=land_after_route,
                        validate_ulog=validate_ulog,
                        px4_runtime_dir=px4_runtime_dir,
                        keep_px4_runtime=keep_px4_runtime,
                        minimal_logging=minimal_logging,
                        sim_speed_factor=attempt_sim_speed_factor,
                        takeoff_sim_speed_factor=attempt_takeoff_sim_speed_factor,
                        waypoint_acceptance_m=waypoint_acceptance_m,
                        takeoff_timeout=takeoff_timeout,
                        waypoint_timeout=waypoint_timeout,
                        attempt=attempt_number,
                        attempt_context=evidence,
                    )
                    evidence.update(
                        {
                            "finished_at_utc": _utc_now(),
                            "outcome": "success",
                            "preliminary_classification": "success",
                            "final_classification": "success",
                            "classification_notes": "",
                            "error": "",
                        }
                    )
                    status = _record_attempt(case_dir, run_id, evidence)
                    metadata = dict(row)
                    metadata.update({key: status[key] for key in (
                        "attempt_count", "failed_attempts", "succeeded_after_retry"
                    )})
                    _write_json_atomic(case_dir / "metadata.json", metadata)
                    _refresh_failure_summary(output_dir, failure_path)
                    completed += 1
                    last_error = None
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    evidence.update(
                        {
                            "finished_at_utc": _utc_now(),
                            "outcome": "failed",
                            "preliminary_classification": _preliminary_failure_classification(
                                evidence.get("failure_phase", "indeterminate"), evidence
                            ),
                            "final_classification": "unreviewed",
                            "classification_notes": "",
                            "error": last_error,
                        }
                    )
                    _record_attempt(case_dir, run_id, evidence)
                    _refresh_failure_summary(output_dir, failure_path)
                    print(f"  attempt {attempt_number} failed: {last_error}", flush=True)
                    time.sleep(min(10.0, 2.0 * attempt_index))
            if last_error is not None:
                failures.append(
                    {
                        "run_id": run_id,
                        "attempt": last_attempt_number,
                        "error": last_error,
                    }
                )
                print(f"failed {run_id}: {last_error}", flush=True)
        finally:
            _release_case_lock(lock_path)
    print(
        f"batch summary: completed={completed} skipped_existing={skipped} "
        f"skipped_terminal_failed={skipped_terminal_failed} "
        f"skipped_budget_exhausted={skipped_budget_exhausted} "
        f"skipped_locked={locked} failed={len(failures)} selected={len(rows)}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(rows)} runs failed; see {failure_path}")
