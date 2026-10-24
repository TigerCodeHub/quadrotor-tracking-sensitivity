import argparse
import concurrent.futures
import csv
import json
import time
from pathlib import Path

from .generate import generate_assets, generate_controller_scripts, generate_models, generate_plans, generate_worlds
from .metrics import compute_metrics, infer_metadata, iter_csv_files, read_csv_rows, read_telemetry, write_csv


DEFAULT_FACTORS = (
    "mass_pct",
    "rotor_pct",
    "payload_mass_pct",
    "payload_x",
    "payload_y",
    "payload_z",
)


def _read_metadata(path):
    if not path:
        return {}
    rows = read_csv_rows(path)
    by_key = {}
    for row in rows:
        for key in ("run_id", "run_name", "case", "model_name"):
            value = row.get(key)
            if value:
                by_key.setdefault(value, row)
    return by_key


def _metadata_for(log_file, root, metadata):
    inferred = infer_metadata(log_file, root)
    keys = [
        inferred.get("run_name"),
        inferred.get("case"),
        Path(log_file).parent.name,
        Path(log_file).stem,
    ]
    merged = dict(inferred)
    sidecar = Path(log_file).parent / "metadata.json"
    if sidecar.exists():
        merged.update(json.loads(sidecar.read_text(encoding="utf-8")))
    for name in ("design_row.json", "attempt_status.json"):
        attempt_sidecar = Path(log_file).parent / name
        if attempt_sidecar.exists():
            merged.update(
                json.loads(attempt_sidecar.read_text(encoding="utf-8"))
            )
    for key in keys:
        if key in metadata:
            merged.update(metadata[key])
            break
    return merged


def _analyze_one(task):
    index, total, log_file, log_root, metadata, mission_index, flight_segment, recovery_threshold, recovery_below = task
    item_start = time.monotonic()
    try:
        run = read_telemetry(log_file, mission_index=mission_index, flight_segment=flight_segment)
        row = _metadata_for(log_file, log_root, metadata)
        row.update(
            compute_metrics(
                run,
                recovery_threshold=recovery_threshold,
                recovery_below=recovery_below,
            )
        )
        return {
            "status": "ok",
            "index": index,
            "total": total,
            "log_file": str(log_file),
            "row": row,
            "samples": row.get("samples"),
            "elapsed": time.monotonic() - item_start,
        }
    except Exception as exc:  # noqa: BLE001 - CLI should continue over bad logs.
        return {
            "status": "failed",
            "index": index,
            "total": total,
            "log_file": str(log_file),
            "failure": {"log_file": str(log_file), "error": f"{type(exc).__name__}: {exc}"},
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed": time.monotonic() - item_start,
        }


def analyze(args):
    metadata = _read_metadata(args.metadata)
    rows_by_index = {}
    failures_by_index = {}
    seen = set()
    start = time.monotonic()
    output_path = Path(args.output)
    failure_path = Path(args.failures) if args.failures else None
    for path in (output_path, failure_path):
        if path and path.exists():
            path.unlink()
    jobs = max(1, int(args.jobs or 1))

    def write_partial_outputs():
        rows = [rows_by_index[index] for index in sorted(rows_by_index)]
        failures = [failures_by_index[index] for index in sorted(failures_by_index)]
        if rows:
            write_csv(output_path, rows)
        if failures and failure_path:
            write_csv(failure_path, failures)

    # Checkpoint interval for write_partial_outputs(). That function rewrites the WHOLE
    # CSV from scratch, so calling it once per result made analysis O(n^2): measured on
    # one host at 17k rows, the parent process pegged one core rewriting a 9 MB file per
    # log while its 24 workers idled at 11% CPU, and throughput decayed 4.6 -> 1.4 logs/s
    # over a single segment. The unconditional final write after the scan loop still
    # guarantees the output is complete.
    CHECKPOINT_EVERY = 500
    _recorded = [0]

    def record_result(result):
        print(f"[{result['index']}/{result['total']}] {result['log_file']}", flush=True)
        if result["status"] == "ok":
            rows_by_index[result["index"]] = result["row"]
            print(f"  ok: samples={result.get('samples')} elapsed={result['elapsed']:.1f}s", flush=True)
        else:
            failures_by_index[result["index"]] = result["failure"]
            print(f"  failed: {result['error']} elapsed={result['elapsed']:.1f}s", flush=True)
        _recorded[0] += 1
        if _recorded[0] % CHECKPOINT_EVERY == 0:
            write_partial_outputs()

    def run_batch(log_files, start_index, total):
        tasks = [
            (
                index,
                total,
                log_file,
                args.log_root,
                metadata,
                args.mission_index,
                args.flight_segment,
                args.recovery_threshold,
                args.recovery_below,
            )
            for index, log_file in enumerate(log_files, start=start_index)
        ]
        if jobs == 1:
            for task in tasks:
                record_result(_analyze_one(task))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_analyze_one, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    record_result(future.result())
        return start_index + len(tasks)

    next_index = 1
    idle_since = time.monotonic()
    while True:
        log_files = [path for path in iter_csv_files(args.log_root) if path not in seen]
        if log_files:
            for path in log_files:
                seen.add(path)
            idle_since = time.monotonic()
            total = len(seen)
            print(
                f"analyzing {len(log_files)} new telemetry logs from {args.log_root} "
                f"with jobs={jobs} total_seen={total}",
                flush=True,
            )
            next_index = run_batch(log_files, next_index, total)
        if not args.watch:
            break
        idle_elapsed = time.monotonic() - idle_since
        if idle_elapsed >= args.watch_idle_seconds:
            print(f"watch idle for {idle_elapsed:.0f}s; exiting", flush=True)
            break
        time.sleep(args.watch_interval)

    write_partial_outputs()
    elapsed = time.monotonic() - start
    print(
        f"analyzed={len(rows_by_index)} failed={len(failures_by_index)} "
        f"output={args.output} elapsed={elapsed:.1f}s"
    )
    return 0 if rows_by_index else 1


def summarize(args):
    from .stats import fit_log_linear_model, plot_summary, read_rows, summarize_rows, write_model_outputs

    rows = read_rows(args.runs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows, args.group_by)
    summary_path = output_dir / "summary.csv"
    write_csv(summary_path, summary)

    factors = tuple(args.factors or DEFAULT_FACTORS)
    model = fit_log_linear_model(rows, args.response, factors, include_interactions=not args.no_interactions)
    if model:
        write_model_outputs(output_dir, model)

    if args.plot_x:
        plot_summary(summary, output_dir, args.plot_x, f"{args.response}_mean")

    print(f"summary={summary_path}")
    if model:
        print(f"model={output_dir / 'model_fit.json'}")
    return 0


def generate_assets_cmd(args):
    generate_assets(args.output_dir, px4_dir=args.px4_dir)
    print(f"generated assets in {args.output_dir}")
    return 0


def generate_sobol_cmd(args):
    from .campaign import campaign_summary, generate_sobol_campaign, read_airframes

    airframes = read_airframes(args.airframes)
    rows = generate_sobol_campaign(
        airframes,
        samples_per_airframe=args.samples_per_airframe,
        repetitions=args.repetitions,
        seed=args.seed,
        plan=args.plan,
        region=args.region,
    )
    write_csv(args.output, rows)
    summary = campaign_summary(rows)
    print(
        f"design={args.output} airframes={summary['airframe_count']} "
        f"configurations={summary['configuration_count']} runs={summary['run_count']}"
    )
    return 0


def generate_baseline_cmd(args):
    from .campaign import generate_baseline_campaign, read_airframes

    rows = generate_baseline_campaign(
        read_airframes(args.airframes), repetitions=args.repetitions, seed=args.seed
    )
    write_csv(args.output, rows)
    print(f"design={args.output} airframes={len(rows) // args.repetitions} runs={len(rows)}")
    return 0


def smoke_flight_cmd(args):
    from .sim import smoke_flight

    smoke_flight(
        system_address=args.system_address,
        takeoff_altitude=args.takeoff_altitude,
        north_m=args.north,
        east_m=args.east,
        hold_seconds=args.hold_seconds,
        timeout=args.timeout,
    )
    print("smoke flight completed and landed")
    return 0


def generate_models_cmd(args):
    px4_dir = Path(args.px4_dir).expanduser()
    base_sdf = args.base_sdf or px4_dir / "Tools/simulation/gz/models/x500_base/model.sdf"
    motor_sdf = args.motor_sdf or px4_dir / "Tools/simulation/gz/models/x500/model.sdf"
    generate_models(
        args.design_csv,
        args.output_dir,
        base_sdf,
        motor_sdf=motor_sdf,
        mrs_dir=Path(args.mrs_dir).expanduser() if args.mrs_dir else None,
    )
    print(f"generated models in {args.output_dir}")
    return 0


def generate_worlds_cmd(args):
    generate_worlds(args.output_dir, Path(args.base_world).expanduser())
    print(f"generated worlds in {args.output_dir}")
    return 0


def generate_plans_cmd(args):
    generate_plans(args.output_dir)
    print(f"generated plans in {args.output_dir}")
    return 0


def generate_controllers_cmd(args):
    generate_controller_scripts(args.output_dir, args.design_csv)
    print(f"generated controller scripts in {args.output_dir}")
    return 0


def split_design_cmd(args):
    rows = read_csv_rows(args.design_csv)
    if args.chunks <= 0:
        raise ValueError("--chunks must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.design_csv).stem
    size = (len(rows) + args.chunks - 1) // args.chunks
    for chunk_index in range(args.chunks):
        start = chunk_index * size
        chunk_rows = rows[start : start + size]
        if not chunk_rows and not args.include_empty:
            continue
        path = output_dir / f"{stem}_chunk_{chunk_index:03d}_of_{args.chunks:03d}.csv"
        write_csv(path, chunk_rows)
        print(f"{path}: rows={len(chunk_rows)}")
    return 0


def run_batch_cmd(args):
    from .sim import run_batch

    run_batch(
        args.design_csv,
        px4_dir=args.px4_dir,
        assets_dir=args.assets_dir,
        output_dir=args.output_dir,
        instance=args.instance,
        limit=args.limit,
        run_id=args.run_id,
        timeout=args.timeout,
        gui=args.gui,
        failures_path=args.failures,
        overwrite=args.overwrite,
        land_after_route=args.land_after_route,
        retries=args.retries,
        skip_existing=not args.rerun_existing,
        claim=not args.no_claim,
        lock_stale_seconds=args.lock_stale_seconds,
        start_index=args.start_index,
        count=args.count,
        chunk_index=args.chunk_index,
        chunk_count=args.chunk_count,
        validate_ulog=not args.no_validate_ulog,
        px4_runtime_dir=args.px4_runtime_dir,
        keep_px4_runtime=args.keep_px4_runtime,
        minimal_logging=not args.full_px4_logging,
        sim_speed_factor=args.sim_speed_factor,
        takeoff_sim_speed_factor=args.takeoff_sim_speed_factor,
        fallback_sim_speed_factor=args.fallback_sim_speed_factor,
        fallback_takeoff_sim_speed_factor=args.fallback_takeoff_sim_speed_factor,
        waypoint_acceptance_m=args.waypoint_acceptance_m,
        takeoff_timeout=args.takeoff_timeout,
        waypoint_timeout=args.waypoint_timeout,
    )
    print(f"completed batch in {args.output_dir}")
    return 0


def classify_run_cmd(args):
    from .sim import _read_run_status, _record_attempt, _utc_now

    target = Path(args.run)
    case_dir = target if target.is_dir() else target.parent
    run_id = case_dir.name
    status = _read_run_status(case_dir, run_id)
    failed = [item for item in status["attempts"] if item.get("outcome") == "failed"]
    if args.attempt is None:
        if not failed:
            raise ValueError(f"{run_id} has no failed attempts to classify")
        attempt_record = failed[-1]
    else:
        matches = [
            item for item in failed if int(item.get("attempt", -1)) == args.attempt
        ]
        if not matches:
            raise ValueError(f"{run_id} has no failed attempt {args.attempt}")
        attempt_record = matches[0]
    attempt_record["final_classification"] = args.classification
    attempt_record["classification_notes"] = args.notes or ""
    attempt_record["classified_at_utc"] = _utc_now()
    _record_attempt(case_dir, run_id, attempt_record)
    print(
        f"classified {run_id} attempt {attempt_record['attempt']} "
        f"as {args.classification}"
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="tdresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assets = subparsers.add_parser("generate-assets", help="Generate designs, wind worlds, plans, and runner script.")
    assets.add_argument("--output-dir", default="generated")
    assets.add_argument("--px4-dir", default="~/PX4-Autopilot")
    assets.set_defaults(func=generate_assets_cmd)

    sobol = subparsers.add_parser(
        "generate-sobol-design",
        help="Generate a deterministic multi-airframe Sobol campaign CSV.",
    )
    sobol.add_argument("--airframes", default="generated/airframes.csv")
    sobol.add_argument("--output", default="generated/designs/multi_airframe_sobol.csv")
    sobol.add_argument("--samples-per-airframe", type=int, default=2048)
    sobol.add_argument("--repetitions", type=int, default=10)
    sobol.add_argument("--seed", type=int, default=20260717)
    sobol.add_argument("--plan", default="straight")
    sobol.add_argument(
        "--region",
        choices=("primary", "boundary"),
        default="primary",
        help="Wind envelope: primary=0-6 m/s mean and 0-3 m/s RMS; boundary retains 0-10/0-6.",
    )
    sobol.set_defaults(func=generate_sobol_cmd)

    baseline = subparsers.add_parser(
        "generate-baseline-design", help="Generate nominal calm-air controls for every airframe."
    )
    baseline.add_argument("--airframes", default="generated/airframes.csv")
    baseline.add_argument("--output", default="generated/designs/multi_airframe_baseline.csv")
    baseline.add_argument("--repetitions", type=int, default=1)
    baseline.add_argument("--seed", type=int, default=20260717)
    baseline.set_defaults(func=generate_baseline_cmd)

    models = subparsers.add_parser("generate-models", help="Generate SDF model variants from a design CSV.")
    models.add_argument("design_csv")
    models.add_argument("--output-dir", default="generated/models")
    models.add_argument("--px4-dir", default="~/PX4-Autopilot")
    models.add_argument("--base-sdf")
    models.add_argument("--motor-sdf")
    models.add_argument(
        "--mrs-dir",
        help="Pinned checkout of ctu-mrs/mrs_uav_gazebo_simulator (ros2 branch).",
    )
    models.set_defaults(func=generate_models_cmd)

    worlds = subparsers.add_parser("generate-worlds", help="Generate calm/weak/moderate/strong wind worlds.")
    worlds.add_argument("--output-dir", default="generated/worlds")
    worlds.add_argument("--base-world", default="~/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf")
    worlds.set_defaults(func=generate_worlds_cmd)

    plans = subparsers.add_parser("generate-plans", help="Generate route-shape QGroundControl plans.")
    plans.add_argument("--output-dir", default="generated/plans")
    plans.set_defaults(func=generate_plans_cmd)

    controllers = subparsers.add_parser("generate-controllers", help="Generate MAVSDK parameter sets for fixed and retuned controller modes.")
    controllers.add_argument("design_csv")
    controllers.add_argument("--output-dir", default="generated/controllers")
    controllers.set_defaults(func=generate_controllers_cmd)

    split = subparsers.add_parser("split-design", help="Split a design CSV into scheduler-friendly contiguous chunks.")
    split.add_argument("design_csv")
    split.add_argument("--output-dir", default="generated/chunks")
    split.add_argument("--chunks", type=int, required=True)
    split.add_argument("--include-empty", action="store_true")
    split.set_defaults(func=split_design_cmd)

    analyze_parser = subparsers.add_parser("analyze", help="Compute SOB2 and validation metrics from CSV or PX4 ULog telemetry.")
    analyze_parser.add_argument("log_root")
    analyze_parser.add_argument("--output", default="results/runs.csv")
    analyze_parser.add_argument("--failures", default="results/failures.csv")
    analyze_parser.add_argument("--metadata", help="Optional design CSV for run metadata.")
    analyze_parser.add_argument("--mission-index", type=int, default=2)
    analyze_parser.add_argument(
        "--flight-segment",
        choices=("armed", "cruise"),
        default="armed",
        help="For ULogs, analyze the full armed interval or only commanded horizontal cruise.",
    )
    analyze_parser.add_argument("--recovery-threshold", type=float, default=2.0)
    analyze_parser.add_argument("--recovery-below", type=float, default=1.0)
    analyze_parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes for ULog parsing.")
    analyze_parser.add_argument("--watch", action="store_true", help="Keep polling log_root for new telemetry logs.")
    analyze_parser.add_argument("--watch-interval", type=float, default=60.0, help="Seconds between watch-mode scans.")
    analyze_parser.add_argument(
        "--watch-idle-seconds",
        type=float,
        default=1800.0,
        help="Exit watch mode after this many seconds with no new logs.",
    )
    analyze_parser.set_defaults(func=analyze)

    summarize_parser = subparsers.add_parser("summarize", help="Aggregate runs and fit log-linear sensitivity models.")
    summarize_parser.add_argument("runs_csv")
    summarize_parser.add_argument("--output-dir", default="results")
    summarize_parser.add_argument("--group-by", nargs="+", default=["experiment", "model_name", "wind_level", "plan", "controller_mode", "mass_pct", "rotor_pct", "inertia_mode"])
    summarize_parser.add_argument("--response", default="sob2")
    summarize_parser.add_argument("--factors", nargs="*")
    summarize_parser.add_argument("--no-interactions", action="store_true")
    summarize_parser.add_argument("--plot-x")
    summarize_parser.set_defaults(func=summarize)

    smoke = subparsers.add_parser("smoke-flight", help="Fly a short MAVSDK SITL takeoff/waypoint/RTL test.")
    smoke.add_argument("--system-address", default="udpin://0.0.0.0:14540")
    smoke.add_argument("--takeoff-altitude", type=float, default=5.0)
    smoke.add_argument("--north", type=float, default=20.0)
    smoke.add_argument("--east", type=float, default=10.0)
    smoke.add_argument("--hold-seconds", type=float, default=10.0)
    smoke.add_argument("--timeout", type=float, default=60.0)
    smoke.set_defaults(func=smoke_flight_cmd)

    batch = subparsers.add_parser("run-batch", help="Run PX4 v1.16.2/Gazebo Harmonic experiment cases.")
    batch.add_argument("design_csv")
    batch.add_argument("--px4-dir", default="~/PX4-Autopilot")
    batch.add_argument("--assets-dir", default="generated")
    batch.add_argument("--output-dir", default="runs")
    batch.add_argument("--instance", type=int, default=1)
    batch.add_argument("--limit", type=int)
    batch.add_argument("--run-id")
    batch.add_argument("--timeout", type=float, default=600.0)
    batch.add_argument("--gui", action="store_true")
    batch.add_argument("--failures", help="Failure CSV path for this batch.")
    batch.add_argument("--overwrite", action="store_true", help="Replace existing ULog outputs for selected runs.")
    batch.add_argument("--land-after-route", action="store_true", help="Return to launch and land after the planned route.")
    batch.add_argument("--retries", type=int, default=2, help="Retry each failed run this many times before recording failure.")
    batch.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Explicitly rerun completed or terminally failed cases instead of skipping them.",
    )
    batch.add_argument("--no-claim", action="store_true", help="Disable per-run lock files for shared output directories.")
    batch.add_argument(
        "--lock-stale-seconds",
        type=float,
        default=3600.0,
        help="Treat run.lock files older than this as stale. Default: 1 hour.",
    )
    batch.add_argument("--start-index", type=int, help="1-based start row after run-id filtering.")
    batch.add_argument("--count", type=int, help="Number of rows to run after start-index/chunk filtering.")
    batch.add_argument("--chunk-index", type=int, help="Zero-based contiguous chunk index.")
    batch.add_argument("--chunk-count", type=int, help="Total number of contiguous chunks.")
    batch.add_argument("--no-validate-ulog", action="store_true", help="Skip post-run ULog parse validation.")
    batch.add_argument(
        "--px4-runtime-dir",
        default="/tmp/px4",
        help="Writable local directory for PX4 rootfs state/logs. Default: /tmp/px4.",
    )
    batch.add_argument("--keep-px4-runtime", action="store_true", help="Keep per-run PX4 runtime dirs after copying ULogs.")
    batch.add_argument(
        "--full-px4-logging",
        action="store_true",
        help="Use PX4's default logger profile instead of the minimal analysis topic set.",
    )
    batch.add_argument(
        "--sim-speed-factor",
        type=float,
        default=10.0,
        help="Requested Gazebo/PX4 lockstep speedup. Use 1 for realtime. Default: 10.",
    )
    batch.add_argument(
        "--takeoff-sim-speed-factor",
        type=float,
        default=2.0,
        help="Use this Gazebo speed through takeoff, then switch to --sim-speed-factor. Default: 2.",
    )
    batch.add_argument(

        "--fallback-sim-speed-factor",
        type=float,
        help="Cruise speed used on retries. Defaults to --sim-speed-factor.",
    )
    batch.add_argument(
        "--fallback-takeoff-sim-speed-factor",
        type=float,
        default=1.0,
        help="Takeoff speed used on retries. Default: realtime (1).",
    )
    batch.add_argument(
        "--waypoint-acceptance-m",
        type=float,
        default=15.0,
        help="Horizontal distance treated as waypoint reached by the runner. Default: 15 m.",
    )
    batch.add_argument(
        "--takeoff-timeout",
        type=float,
        default=90.0,
        help="Seconds allowed to reach takeoff altitude before retrying. Default: 90.",
    )
    batch.add_argument(
        "--waypoint-timeout",
        type=float,
        default=120.0,
        help="Seconds allowed to reach each waypoint before retrying. Default: 120.",
    )
    batch.set_defaults(func=run_batch_cmd)
    classify = subparsers.add_parser(
        "classify-run",
        help="Assign an analyst-reviewed classification to a failed run attempt.",
    )
    classify.add_argument("run", help="Run directory or run_status.json path.")
    classify.add_argument("--attempt", type=int, help="Attempt number; defaults to latest failure.")
    classify.add_argument(
        "--classification",
        required=True,
        choices=("flight_mechanics", "infrastructure", "indeterminate"),
    )
    classify.add_argument("--notes", default="")
    classify.set_defaults(func=classify_run_cmd)


    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
