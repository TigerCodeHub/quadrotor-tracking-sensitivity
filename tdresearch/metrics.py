import csv
import gzip
import json
import math
import shutil
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = (
    "missionItemIndex",
    "localPosition.x",
    "localPosition.y",
    "localPosition.z",
    "localPositionSetpoint.x",
    "localPositionSetpoint.y",
    "localPositionSetpoint.z",
)

TIME_COLUMNS = (
    "timestamp",
    "timestamp[us]",
    "time",
    "time_usec",
    "time_boot_ms",
    "rosbagTimestamp",
)

ULOG_TOPICS = (
    "vehicle_status",
    "vehicle_local_position",
    "vehicle_local_position_setpoint",
    "actuator_motors",
)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def scale(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def norm2(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self):
        return math.sqrt(self.norm2())


@dataclass
class TelemetryRun:
    log_file: Path
    actual: list
    setpoint: list
    times: list
    controls: list

    @property
    def diffs(self):
        return [a - b for a, b in zip(self.actual, self.setpoint)]


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _time_seconds(value, column):
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    if column in ("timestamp", "timestamp[us]", "time_usec"):
        return numeric / 1_000_000.0
    if column == "time_boot_ms":
        return numeric / 1_000.0
    return numeric


def read_telemetry(log_file, mission_index=2, flight_segment="armed"):
    path = Path(log_file)
    if _is_ulog_path(path):
        return read_ulog(path, flight_segment=flight_segment)

    actual = []
    setpoint = []
    times = []
    controls = []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

        time_column = next((name for name in TIME_COLUMNS if name in reader.fieldnames), None)
        control_columns = [
            name
            for name in reader.fieldnames
            if name.startswith("actuatorControls.")
            or name.startswith("actuator_outputs.")
            or name.startswith("actuatorOutputs.")
            or name.startswith("motor_speed.")
            or name.startswith("motorSpeed.")
        ]
        for row in reader:
            try:
                row_mission_index = int(float(row["missionItemIndex"]))
            except (TypeError, ValueError):
                continue
            if row_mission_index != mission_index:
                continue

            actual.append(
                Vec3(
                    float(row["localPosition.x"]),
                    float(row["localPosition.y"]),
                    float(row["localPosition.z"]),
                )
            )
            setpoint.append(
                Vec3(
                    float(row["localPositionSetpoint.x"]),
                    float(row["localPositionSetpoint.y"]),
                    float(row["localPositionSetpoint.z"]),
                )
            )
            if time_column:
                times.append(_time_seconds(row.get(time_column), time_column))
            control_values = [_float_or_none(row.get(column)) for column in control_columns]
            control_values = [value for value in control_values if value is not None]
            controls.append(control_values)

    if len(actual) < 2:
        raise ValueError(f"{path} has fewer than two samples for missionItemIndex={mission_index}")

    if len(times) != len(actual) or any(value is None for value in times):
        times = [float(index) for index in range(len(actual))]
    else:
        start = times[0]
        times = [value - start for value in times]

    return TelemetryRun(path, actual, setpoint, times, controls)


def _ulog_dataset(ulog, name):
    for dataset in ulog.data_list:
        if dataset.name == name and dataset.multi_id == 0:
            return dataset.data
    raise ValueError(f"ULog is missing topic: {name}")


def _armed_interval(ulog):
    status = _ulog_dataset(ulog, "vehicle_status")
    intervals = []
    active_start = None
    previous_timestamp = None
    for timestamp, state in zip(status["timestamp"], status["arming_state"]):
        timestamp = int(timestamp)
        armed = int(state) == 2
        if armed and active_start is None:
            active_start = timestamp
        elif not armed and active_start is not None:
            intervals.append((active_start, previous_timestamp or timestamp))
            active_start = None
        previous_timestamp = timestamp
    if active_start is not None:
        intervals.append((active_start, previous_timestamp))
    if not intervals:
        raise ValueError("ULog contains no armed flight interval")
    return intervals[-1]


def _nearest_index(timestamps, target):
    index = bisect_left(timestamps, target)
    if index <= 0:
        return 0
    if index >= len(timestamps):
        return len(timestamps) - 1
    before = timestamps[index - 1]
    after = timestamps[index]
    return index - 1 if target - before <= after - target else index


def _cruise_start_timestamp(setpoint_times, x_values, y_values, start, end, minimum_progress_m=1.0):
    """Find the straight-route start from commanded horizontal progress.

    The minimal logger profile does not include mission-item progress. During
    takeoff, however, PX4 holds the horizontal setpoint fixed; it starts moving
    along the route after the takeoff phase. Requiring one metre of commanded
    horizontal progress avoids triggering on float noise while discarding only
    a negligible portion of the roughly 1 km route.
    """
    origin = None
    for timestamp, x_value, y_value in zip(setpoint_times, x_values, y_values):
        timestamp = int(timestamp)
        x_value = float(x_value)
        y_value = float(y_value)
        if timestamp < start or timestamp > end or not all(math.isfinite(value) for value in (x_value, y_value)):
            continue
        if origin is None:
            origin = (x_value, y_value)
            continue
        if math.hypot(x_value - origin[0], y_value - origin[1]) >= minimum_progress_m:
            return timestamp
    raise ValueError("ULog contains no commanded horizontal cruise segment")


def read_ulog(log_file, flight_segment="armed"):
    path = Path(log_file)
    if path.name.lower().endswith(".ulg.gz"):
        with tempfile.TemporaryDirectory(prefix="tdresearch-ulog-") as directory:
            temporary = Path(directory) / path.name[:-3]
            with gzip.open(path, "rb") as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            return _read_ulog_uncompressed(
                temporary,
                original_path=path,
                flight_segment=flight_segment,
            )
    return _read_ulog_uncompressed(path, original_path=path, flight_segment=flight_segment)


def _read_ulog_uncompressed(log_file, original_path, flight_segment):
    try:
        from pyulog import ULog
    except ImportError as exc:
        raise RuntimeError("reading PX4 ULog files requires the pyulog package") from exc

    path = Path(log_file)
    original_path = Path(original_path)
    ulog = ULog(str(path), ULOG_TOPICS)
    start, end = _armed_interval(ulog)
    position = _ulog_dataset(ulog, "vehicle_local_position")
    setpoint = _ulog_dataset(ulog, "vehicle_local_position_setpoint")

    position_times = [int(value) for value in position["timestamp"]]
    setpoint_times = [int(value) for value in setpoint["timestamp"]]
    if flight_segment == "cruise":
        start = _cruise_start_timestamp(
            setpoint_times,
            setpoint["x"],
            setpoint["y"],
            start,
            end,
        )
    elif flight_segment != "armed":
        raise ValueError(f"unknown flight segment: {flight_segment}")
    actual = []
    desired = []
    times = []
    selected_timestamps = []

    for index, timestamp in enumerate(setpoint_times):
        if timestamp < start or timestamp > end:
            continue
        target = Vec3(float(setpoint["x"][index]), float(setpoint["y"][index]), float(setpoint["z"][index]))
        if not all(math.isfinite(value) for value in (target.x, target.y, target.z)):
            continue
        position_index = _nearest_index(position_times, timestamp)
        point = Vec3(
            float(position["x"][position_index]),
            float(position["y"][position_index]),
            float(position["z"][position_index]),
        )
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            continue
        actual.append(point)
        desired.append(target)
        times.append((timestamp - start) / 1_000_000.0)
        selected_timestamps.append(timestamp)

    if len(actual) < 2:
        raise ValueError(
            f"{original_path} has fewer than two finite setpoint samples during armed flight"
        )

    controls = [[] for _ in actual]
    try:
        motors = _ulog_dataset(ulog, "actuator_motors")
        motor_times = [int(value) for value in motors["timestamp"]]
        control_names = sorted(
            (name for name in motors if name.startswith("control[")),
            key=lambda name: int(name[8:-1]),
        )
        for row_index, timestamp in enumerate(selected_timestamps):
            motor_index = _nearest_index(motor_times, timestamp)
            values = [float(motors[name][motor_index]) for name in control_names]
            controls[row_index] = [value for value in values if math.isfinite(value)]
    except ValueError:
        pass

    return TelemetryRun(original_path, actual, desired, times, controls)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _rms(values):
    values = list(values)
    return math.sqrt(_mean(value * value for value in values)) if values else 0.0


def _sob2(values):
    values = list(values)
    integral = sum(value * value for value in values)
    derivative = sum((value - previous) ** 2 for previous, value in zip(values, values[1:]))
    return math.sqrt(integral + derivative) / len(values)


def _sob2_vec(vectors):
    vectors = list(vectors)
    integral = sum(vector.norm2() for vector in vectors)
    derivative = sum((value - previous).norm2() for previous, value in zip(vectors, vectors[1:]))
    return math.sqrt(integral + derivative) / len(vectors)


def _sob2_h1(vectors, times, tau=1.0):
    """Correctly-discretized H^1 seminorm of the deviation (audit #4).

    The legacy `_sob2` / `_sob2_vec` divide a plain SUM by the sample count N and use a bare
    first difference, so the score (a) shrinks with flight duration and (b) depends on the
    sample rate -- both documented pathologies that bias cross-route comparisons. This form
    fixes both:

        sqrt( mean_i ||d_i||^2  +  tau^2 * mean_i ||(d_i - d_{i-1}) / dt_i||^2 )

    * MEAN (not sum/N) over samples -> invariant to duration.
    * the derivative is a true rate ||Δd/Δt|| -> invariant to sample rate.
    * `tau` (seconds) is an explicit time scale so both terms share units (metres).

    `times` are per-sample timestamps in seconds (same length as `vectors`). Non-positive
    dt are skipped. Prefer a fixed-rate resample upstream for full comparability; this already
    removes the two mechanical sensitivities on irregular samples.
    """
    vectors = list(vectors)
    times = list(times)
    n = len(vectors)
    if n < 2:
        return float("nan")
    pos_sq = sum(v.norm2() for v in vectors) / n
    deriv_terms = []
    for (p, v), (tp, tv) in zip(zip(vectors, vectors[1:]), zip(times, times[1:])):
        dt = tv - tp
        if dt > 0:
            deriv_terms.append((v - p).norm2() / (dt * dt))
    deriv_sq = (sum(deriv_terms) / len(deriv_terms)) if deriv_terms else 0.0
    return math.sqrt(pos_sq + tau * tau * deriv_sq)


def _distance_to_segment(point, start, end):
    direction = end - start
    denom = direction.norm2()
    if denom == 0:
        return (point - start).norm()
    relative = point - start
    t = max(0.0, min(1.0, relative.dot(direction) / denom))
    projection = start + direction.scale(t)
    return (point - projection).norm()


def _cross_track_errors(actual, setpoint):
    # NAMING NOTE (2026-10-10): despite the legacy name, the `cross_track_rmse` column this
    # feeds is the *3-D position-tracking RMSE* (3-D PTE) in the paper -- the 3-D (incl.
    # altitude) shortest distance from each actual sample to the two locally bracketing
    # segments of the commanded `vehicle_local_position_setpoint` polyline (the controller's
    # commanded path, NOT the raw mission-waypoint corridor). It is NOT textbook lateral
    # cross-track error. Horizontal-only error is recoverable as
    # sqrt(rms_error**2 - altitude_rmse**2); the perpendicular/along-track split needs the
    # raw ULog timeseries. See DATA_MANIFEST.md "Metric naming note".
    errors = []
    if len(setpoint) < 2:
        return [(point - target).norm() for point, target in zip(actual, setpoint)]
    for index, point in enumerate(actual):
        candidates = []
        if index > 0:
            candidates.append((setpoint[index - 1], setpoint[index]))
        if index < len(setpoint) - 1:
            candidates.append((setpoint[index], setpoint[index + 1]))
        errors.append(min(_distance_to_segment(point, start, end) for start, end in candidates))
    return errors


def _recovery_time(times, deviations, threshold, recovered_below):
    longest = 0.0
    active_start = None
    last_time = times[0]

    for time_value, deviation in zip(times, deviations):
        if active_start is None and deviation >= threshold:
            active_start = time_value
        elif active_start is not None and deviation <= recovered_below:
            longest = max(longest, time_value - active_start)
            active_start = None
        last_time = time_value

    if active_start is not None:
        longest = max(longest, last_time - active_start)
    return longest


def compute_metrics(run, recovery_threshold=2.0, recovery_below=1.0):
    diffs = run.diffs
    deviations = [diff.norm() for diff in diffs]
    x_errors = [diff.x for diff in diffs]
    y_errors = [diff.y for diff in diffs]
    z_errors = [diff.z for diff in diffs]
    cross_track = _cross_track_errors(run.actual, run.setpoint)

    duration = run.times[-1] - run.times[0] if len(run.times) > 1 else 0.0
    metrics = {
        "log_file": str(run.log_file),
        "samples": len(run.actual),
        "duration_s": duration,
        "sob2": _sob2_vec(diffs),
        "sob2_x": _sob2(x_errors),
        "sob2_y": _sob2(y_errors),
        "sob2_z": _sob2(z_errors),
        "rms_error": _rms(deviations),
        "max_error": max(deviations),
        "mean_error": _mean(deviations),
        "rms_x": _rms(x_errors),
        "rms_y": _rms(y_errors),
        "rms_z": _rms(z_errors),
        "max_abs_x": max(abs(value) for value in x_errors),
        "max_abs_y": max(abs(value) for value in y_errors),
        "max_abs_z": max(abs(value) for value in z_errors),
        "cross_track_rmse": _rms(cross_track),
        "cross_track_max": max(cross_track),
        "altitude_rmse": _rms(z_errors),
        "recovery_time_s": _recovery_time(run.times, deviations, recovery_threshold, recovery_below),
    }
    effort = [math.sqrt(sum(value * value for value in values)) for values in run.controls if values]
    if effort:
        metrics.update(
            {
                "control_effort_rms": _rms(effort),
                "control_effort_mean": _mean(effort),
                "control_effort_max": max(effort),
            }
        )
    return metrics


def infer_metadata(log_file, root):
    path = Path(log_file)
    root = Path(root)
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path.name

    metadata = {}
    parts = relative.parts if hasattr(relative, "parts") else (str(relative),)
    if len(parts) > 1:
        metadata["case"] = parts[0]
        if "_" in parts[0]:
            case, plan = parts[0].rsplit("_", 1)
            metadata["model"] = case
            metadata["plan"] = plan
    metadata["run_name"] = (
        path.name[:-7] if path.name.lower().endswith(".ulg.gz") else path.stem
    )
    return metadata


def _is_ulog_path(path):
    name = Path(path).name.lower()
    return name.endswith(".ulg") or name.endswith(".ulg.gz")


def iter_csv_files(root):
    root = Path(root)
    if root.is_file() and (root.suffix.lower() == ".csv" or _is_ulog_path(root)):
        yield root
        return
    bookkeeping = {"runs.csv", "summary.csv"}
    for path in sorted(
        (*root.rglob("*.csv"), *root.rglob("*.ulg"), *root.rglob("*.ulg.gz"))
    ):
        if path.name in bookkeeping or path.name.startswith("failures"):
            continue
        if _is_ulog_path(path) and path.parent.name.startswith("attempt_"):
            status_path = path.parent / "attempt_status.json"
            if status_path.exists():
                try:
                    attempt_status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    attempt_status = {}
                if attempt_status.get("outcome") == "success":
                    case_dir = path.parent.parent.parent
                    canonical = case_dir / f"{attempt_status.get('run_id', case_dir.name)}.ulg"
                    compressed = canonical.with_name(f"{canonical.name}.gz")
                    if canonical.exists() or compressed.exists():
                        continue
        yield path


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
