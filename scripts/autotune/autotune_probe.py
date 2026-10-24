"""Standalone prototype: does PX4 in-flight autotune work in our SITL, via MAVSDK?

De-risks EXTENSIONS item 1 before any integration. Launches ONE PX4+Gazebo SITL for a
given v2 config (reusing tdresearch.sim's launch helpers), takes off into calm air, triggers
the built-in multicopter autotuner (MC_AT_*), waits for it to converge, and reads back the
15 rate/attitude gains it writes. Prints baseline vs tuned so we can see the autotuner both
runs AND produces sensible, config-dependent gains.

This touches NO production code path and writes NO controller JSON -- it only proves the
mechanism. Run on a host NOT flying the campaign.

  usage: autotune_probe.py <model_name> <instance>
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tdresearch.sim import (
    _case_world,
    _px4_runtime_root,
    _wait_for_world,
    _set_world_speed,
    _load_plan,
    _wait_until_armable,
    _wait_for_altitude,
    _first,
    _apply_parameters,
    _terminate_process,
    _wait_until_landed,
    _DATALINK_FAILSAFE_PARAMS,
)

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "generated/sobol_v2"
PX4_DIR = Path(os.environ.get("PX4_DIR", str(Path.home() / "PX4-Autopilot")))

# The 15 params PX4 autotune writes (mc_autotune_attitude_control.cpp apply()).
GAINS = [
    "MC_ROLLRATE_K", "MC_ROLLRATE_P", "MC_ROLLRATE_I", "MC_ROLLRATE_D", "MC_ROLL_P",
    "MC_PITCHRATE_K", "MC_PITCHRATE_P", "MC_PITCHRATE_I", "MC_PITCHRATE_D", "MC_PITCH_P",
    "MC_YAWRATE_K", "MC_YAWRATE_P", "MC_YAWRATE_I", "MC_YAWRATE_D", "MC_YAW_P",
]


async def read_gains(drone):
    out = {}
    for name in GAINS:
        try:
            out[name] = await drone.param.get_param_float(name)
        except Exception as exc:
            out[name] = "ERR:%s" % exc
    return out


async def run(model_name, instance, sysid_amp=0.7, write_controller_dir=None, tune_rtf=1.0):
    controller = json.loads((ASSETS / "controllers" / f"{model_name}_fixed.json").read_text())
    base_world = ASSETS / "worlds" / "turbulence_calm.sdf"
    plan_path = ASSETS / "plans" / "straight.plan"
    runtime_dir = f"/tmp/px4-atprobe-{instance}"
    attempt_dir = Path(runtime_dir + "-work")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    world_path = attempt_dir / f"{model_name}.sdf"
    row = {"wind_seed": 1, "mean_wind_m_s": 0.0, "turbulence_rms_m_s": 0.0}
    world_name = _case_world(base_world, world_path, model_name, instance, row)

    px4_build = PX4_DIR / "build" / "px4_sitl_default"
    px4_binary = px4_build / "bin" / "px4"
    px4_rootfs = px4_build / "rootfs"
    # Full logging (not the minimal whitelist): the logger's default topic set includes
    # autotune_attitude_control_status @10 Hz, which is the ONLY place the tuner exposes
    # its state machine and identified gains -- MC_AT_START returning to 0 is ambiguous
    # (state::fail also resets it via stopAutotune()).
    instance_rootfs = _px4_runtime_root(px4_rootfs, runtime_dir, "atprobe", instance, minimal_logging=False)

    env = os.environ.copy()
    resource_paths = [
        str(ASSETS / "models"), str(PX4_DIR / "Tools/simulation/gz/models"),
        str(ASSETS / "worlds"), str(PX4_DIR / "Tools/simulation/gz/worlds"),
    ]
    env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(resource_paths)
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = str(px4_build / "src/modules/simulation/gz_plugins")
    env["GZ_SIM_SERVER_CONFIG_PATH"] = str(PX4_DIR / "src/modules/simulation/gz_bridge/server.config")
    env["GZ_PARTITION"] = f"atprobe_{os.getpid()}_{instance}"
    env.update({
        "PX4_SYS_AUTOSTART": "4001", "PX4_GZ_STANDALONE": "1",
        "PX4_GZ_MODEL_NAME": f"{model_name}_{instance}", "PX4_GZ_WORLD": world_name,
        "PX4_PARAM_SDLOG_MODE": "0", "PX4_SIM_SPEED_FACTOR": "1",
    })
    for name, value in controller.get("simulator_parameters", {}).items():
        env[f"PX4_PARAM_{name}"] = str(value)

    gz_log = (attempt_dir / "gazebo.log").open("w")
    px4_log = (attempt_dir / "px4.log").open("w")
    gazebo = px4 = None
    status_task = None
    result = {"model": model_name, "status": "unknown"}
    try:
        gazebo = subprocess.Popen(
            ["gz", "sim", "-r", "-s", "--seed", "1", str(world_path)],
            env=env, stdout=gz_log, stderr=subprocess.STDOUT, start_new_session=True)
        _wait_for_world(env, world_name, gazebo, 30)
        _set_world_speed(env, world_name, 1.0)
        px4 = subprocess.Popen(
            [str(px4_binary), "-d", str(px4_build / "etc"), "-i", str(instance),
             "-w", str(instance_rootfs)],
            cwd=instance_rootfs, env=env, stdin=subprocess.DEVNULL,
            stdout=px4_log, stderr=subprocess.STDOUT, start_new_session=True)
        drone_port = 14540 + instance
        mavsdk_port = 50050 + instance

        from mavsdk import System
        drone = System(port=mavsdk_port)
        await asyncio.wait_for(drone.connect(system_address=f"udpin://0.0.0.0:{drone_port}"), timeout=60)

        # Autotune reports success/failure and the reason only via MAVLink STATUSTEXT, not the
        # PX4 console log -- capture it so we can see WHY verification passes or fails.
        async def watch_status():
            try:
                async for st in drone.telemetry.status_text():
                    print(f"  [STATUS {st.type}] {st.text}", flush=True)
            except Exception:
                pass
        status_task = asyncio.ensure_future(watch_status())
        print("  connected; applying datalink failsafe params", flush=True)
        # Must precede the armable wait, exactly as _fly_plan does: without the extended
        # datalink-loss timeout the vehicle enters failsafe the instant it arms and never
        # climbs (observed: "Takeoff detected" but altitude pinned at ground).
        await _apply_parameters(drone, _DATALINK_FAILSAFE_PARAMS)
        print("  waiting armable", flush=True)
        await _wait_until_armable(drone, 120)
        await _apply_parameters(drone, controller.get("parameters", {}))
        planned_home, items = _load_plan(plan_path)
        takeoff_alt = float(items[0]["params"][6])
        # Set takeoff altitude via BOTH the param and the MAVSDK call: set_takeoff_altitude
        # alone flakily loses to PX4's 2.5 m default in a startup race (observed), and 2.5 m
        # is too low for the autotune oscillation maneuver. MC_AT_SYSID_AMP is PX4's own
        # documented remedy for non-convergence; at the 0.7 default the YAW phase converges
        # too slowly for the module's hard 20 s per-state timeout (variance 71.7 vs gate 50).
        await _apply_parameters(drone, {
            "MIS_TAKEOFF_ALT": takeoff_alt,
            "MC_AT_EN": 1,
            "MC_AT_APPLY": 1,
            "MC_AT_SYSID_AMP": sysid_amp,
        })
        await drone.action.set_takeoff_altitude(takeoff_alt)
        # The MIS_TAKEOFF_ALT set can lose a startup race and PX4 then takes off to its
        # 2.5 m default (observed: "Using default takeoff altitude: 2.5 m" despite the
        # param write). Read it back and re-send until it sticks.
        for _ in range(5):
            got = await drone.param.get_param_float("MIS_TAKEOFF_ALT")
            if abs(got - takeoff_alt) < 0.5:
                break
            print(f"  MIS_TAKEOFF_ALT read back {got:.1f}, re-sending", flush=True)
            await _apply_parameters(drone, {"MIS_TAKEOFF_ALT": takeoff_alt})
        print(f"  arming, taking off to {takeoff_alt:.0f} m", flush=True)
        await drone.action.arm()
        await drone.action.takeoff()
        try:
            await _wait_for_altitude(drone, takeoff_alt * 0.8, 90)
        except asyncio.TimeoutError:
            # Takeoff completed at the stale default altitude. Climb the rest of the way
            # with a plain goto at the current position -- Hold mode hover at altitude is
            # exactly what the autotuner wants anyway.
            pos = await _first(drone.telemetry.position())
            if pos.relative_altitude_m >= takeoff_alt * 0.8:
                pass
            else:
                target_amsl = pos.absolute_altitude_m + (takeoff_alt - pos.relative_altitude_m)
                print(f"  takeoff stalled at {pos.relative_altitude_m:.1f} m; "
                      f"goto same spot at {takeoff_alt:.0f} m rel", flush=True)
                await drone.action.goto_location(pos.latitude_deg, pos.longitude_deg,
                                                 target_amsl, float("nan"))
                await _wait_for_altitude(drone, takeoff_alt * 0.8, 180)
        # Speed up the physics for the autotune portion ONLY (takeoff stayed at 1x -- fast
        # takeoff is known to fail). The autotuner's timeouts and identification are all in
        # SIM time, so a higher real-time factor compresses wall-clock without changing the
        # identified gains (verified 1x vs 4x). Cuts each tune flight several-fold.
        if tune_rtf and abs(tune_rtf - 1.0) > 1e-6:
            _set_world_speed(env, world_name, tune_rtf)
            print(f"  altitude reached; sim speed -> {tune_rtf}x for autotune", flush=True)
        else:
            print("  altitude reached", flush=True)
        print("  settling hover 8 s", flush=True)
        await asyncio.sleep(8)

        base = await read_gains(drone)
        print("  starting autotune", flush=True)
        await drone.param.set_param_int("MC_AT_START", 1)

        # Identification finishes in ~35-60 s SIM time (x500 slowest, ~57 s), then the tuner
        # parks in state::wait_for_disarm with MC_AT_START STILL 1 -- it only clears on disarm.
        # So polling for START=0 wastes the whole deadline (the old 320 s poll was the tuner's
        # real bottleneck, NOT the physics). Instead wait a bounded sim-time that covers the
        # slowest airframe, then land -- landing commits the gains. During flight MC_AT_START=0
        # can ONLY mean the tune FAILED (complete clears it only after disarm), so break early
        # on 0 to skip the wait. The wait is sim-time / RTF in wall-clock.
        ident_wait_sim = 120.0
        deadline = time.monotonic() + ident_wait_sim / max(tune_rtf, 0.1) + 20
        started = time.monotonic()
        state = 1
        while time.monotonic() < deadline:
            await asyncio.sleep(3)
            try:
                state = await drone.param.get_param_int("MC_AT_START")
            except Exception as exc:
                print(f"  poll error: {exc}", flush=True)
                continue
            if state == 0:
                print(f"  t={time.monotonic()-started:5.0f}s  MC_AT_START=0 (tune FAILED "
                      f"in-flight -- landing, no gains)", flush=True)
                break
        # With MC_AT_APPLY=1 the identified gains are committed to params ONLY on disarm
        # (saveGainsToParams runs in state::wait_for_disarm when !_armed). So land, let it
        # disarm, then read -- reading while still armed returns the untouched baseline.
        print("  autotune done; landing to commit gains on disarm", flush=True)
        try:
            await drone.action.land()
            await _wait_until_landed(drone, 120)
        except Exception as exc:
            print(f"  land warning: {exc}", flush=True)
        await asyncio.sleep(4)
        tuned = await read_gains(drone)
        result["converged"] = (state == 0)
        result["elapsed_s"] = round(time.monotonic() - started, 1)
        result["baseline"] = base
        result["tuned"] = tuned
        result["status"] = "ok" if state == 0 else "timeout"

        print("\n  === GAINS: baseline -> tuned ===", flush=True)
        n_changed = 0
        for name in GAINS:
            b, t = base.get(name), tuned.get(name)
            try:
                ratio = "x%.2f" % (t / b) if b else "n/a"
                if isinstance(b, float) and isinstance(t, float) and abs(t - b) > 1e-6 * max(abs(b), 1e-6):
                    n_changed += 1
            except Exception:
                ratio = "n/a"
            print(f"    {name:16s} {b!s:>10.10} -> {t!s:>10.10}  {ratio}", flush=True)
        result["n_gains_changed"] = n_changed

        # Write the autotune controller JSON when the tune actually SUCCEEDED. The reliable
        # success signal is "gains changed from baseline": on COMPLETE the gains commit on
        # disarm and differ; on FAIL they never commit and equal baseline (this does NOT rely
        # on the ambiguous MC_AT_START/status). A COMPLETE tune changes ~10/15: the 9 rate
        # P/I/D plus MC_YAW_P. FIVE gains never move even on success -- the 3 MC_*RATE_K (PX4
        # sets them to 1.0, the baseline) and MC_ROLL_P/MC_PITCH_P (att_p clamped at its 6.5
        # max, already the baseline). So the gate is >=6 changed: well above a failure's 0,
        # safely below the ~10 a success delivers. Writes {model}_autotune.json = the fixed
        # controller with the autotuned rate/attitude gains swapped in (position-loop MPC_*
        # params and motor scaling kept), which is exactly what controller_mode=autotune loads.
        if write_controller_dir is not None and n_changed >= 6:
            out = {
                "model_name": model_name,
                "controller_mode": "autotune",
                "parameters": dict(controller.get("parameters", {})),
                "simulator_parameters": dict(controller.get("simulator_parameters", {})),
                "autotune_sysid_amp": sysid_amp,
            }
            for name in GAINS:
                if isinstance(tuned.get(name), float):
                    out["parameters"][name] = tuned[name]
            dest = Path(write_controller_dir) / f"{model_name}_autotune.json"
            tmp = dest.with_suffix(".json.partial")
            tmp.write_text(json.dumps(out, indent=2))
            tmp.replace(dest)
            result["controller_written"] = str(dest)
            print(f"  WROTE {dest} ({n_changed}/15 gains changed)", flush=True)
        elif write_controller_dir is not None:
            print(f"  NOT writing controller: only {n_changed}/15 gains changed (<6 -- tune "
                  f"did not commit; a success changes ~10)", flush=True)

        try:
            await drone.action.land()
        except Exception:
            pass
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {result['error']}", flush=True)
    finally:
        # The status_text() async generator keeps the MAVSDK gRPC stream (and thus the
        # event loop) alive forever; without cancelling it asyncio.run() never returns, so
        # the process hangs AFTER printing RESULT -- observed leaking a mavsdk_server per run
        # and blocking any chained probe. Cancel it and let it settle before teardown.
        if status_task is not None:
            status_task.cancel()
            try:
                await status_task
            except (asyncio.CancelledError, Exception):
                pass
        _terminate_process(px4)
        _terminate_process(gazebo)
        gz_log.close()
        px4_log.close()
        # /tmp is tmpfs; move the ULog somewhere persistent so the autotune status topic
        # can be inspected after the fact.
        keep_dir = Path(os.environ.get("TDR_ATPROBE_LOGS", str(REPO / "atprobe_logs")))
        keep_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for ulg in sorted(Path(instance_rootfs).glob("log/**/*.ulg")):
            dest = keep_dir / f"{model_name}_i{instance}_{ulg.name}"
            shutil.copy2(ulg, dest)
            result.setdefault("ulogs", []).append(str(dest))
            print(f"  ULog kept: {dest}", flush=True)
    print("\nRESULT " + json.dumps({k: v for k, v in result.items() if k not in ("baseline", "tuned")}))
    return result


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "sobol_mrs_f330_s0000"
    inst = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    amp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    # 4th arg (or $TDR_AUTOTUNE_OUT): dir to write {model}_autotune.json on success.
    write_dir = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("TDR_AUTOTUNE_OUT")
    rtf = float(os.environ.get("TDR_TUNE_RTF", "1"))   # sim speed for the autotune portion
    asyncio.run(run(model, inst, amp, write_dir, rtf))
    # Belt-and-suspenders: MAVSDK's atexit gRPC teardown has been seen to block even after
    # the loop closes. RESULT is already printed and flushed, so exit hard.
    sys.stdout.flush()
    os._exit(0)
