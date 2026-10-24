import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from .metrics import write_csv


SUMMARY_METRICS = (
    "sob2",
    "sob2_x",
    "sob2_y",
    "sob2_z",
    "rms_error",
    "max_error",
    "cross_track_rmse",
    "altitude_rmse",
    "recovery_time_s",
)


def _as_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _group_key(row, columns):
    return tuple(row.get(column, "") for column in columns)


def summarize_rows(rows, group_by):
    grouped = {}
    for row in rows:
        grouped.setdefault(_group_key(row, group_by), []).append(row)

    summary_rows = []
    for key, group in sorted(grouped.items()):
        out = {column: value for column, value in zip(group_by, key)}
        out["n"] = len(group)
        for metric in SUMMARY_METRICS:
            values = [_as_float(row, metric) for row in group]
            values = [value for value in values if value is not None and math.isfinite(value)]
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = std / math.sqrt(len(values)) if values else 0.0
            ci95 = 1.96 * sem
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_sem"] = sem
            out[f"{metric}_ci95"] = ci95
        summary_rows.append(out)
    return summary_rows


def _numeric_columns(rows, candidates):
    usable = []
    for column in candidates:
        values = [_as_float(row, column) for row in rows]
        if all(value is not None and math.isfinite(value) for value in values):
            usable.append(column)
    return usable


def fit_log_linear_model(rows, response, factors, include_interactions=True):
    clean = []
    factor_columns = _numeric_columns(rows, factors)
    for row in rows:
        y = _as_float(row, response)
        if y is None or y <= 0:
            continue
        factor_values = [_as_float(row, column) for column in factor_columns]
        if any(value is None for value in factor_values):
            continue
        clean.append((row, math.log(y), factor_values))

    if len(clean) < 3 or not factor_columns:
        return None

    terms = ["intercept"] + factor_columns
    interaction_pairs = list(combinations(range(len(factor_columns)), 2)) if include_interactions else []
    terms.extend(f"{factor_columns[i]}:{factor_columns[j]}" for i, j in interaction_pairs)

    design = []
    y_values = []
    for _, y, factor_values in clean:
        row = [1.0]
        row.extend(factor_values)
        row.extend(factor_values[i] * factor_values[j] for i, j in interaction_pairs)
        design.append(row)
        y_values.append(y)

    x = np.asarray(design, dtype=float)
    y = np.asarray(y_values, dtype=float)
    beta, residuals, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ beta
    residual = y - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    dof = max(0, len(y) - rank)

    coefficient_rows = []
    covariance = None
    if dof > 0:
        mse = ss_res / dof
        try:
            covariance = mse * np.linalg.inv(x.T @ x)
        except np.linalg.LinAlgError:
            covariance = None

    for index, (term, estimate) in enumerate(zip(terms, beta)):
        row = {"term": term, "estimate": float(estimate)}
        if covariance is not None:
            se = math.sqrt(max(0.0, float(covariance[index, index])))
            t_stat = float(estimate / se) if se else math.inf
            p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), dof)) if math.isfinite(t_stat) else 0.0
            row.update({"std_error": se, "t": t_stat, "p_value": p_value})
        coefficient_rows.append(row)

    residual_rows = []
    for (source_row, observed, _), pred, res in zip(clean, predicted, residual):
        residual_rows.append(
            {
                "log_file": source_row.get("log_file", ""),
                "observed_log": float(observed),
                "predicted_log": float(pred),
                "residual_log": float(res),
            }
        )

    return {
        "response": response,
        "factors": factor_columns,
        "terms": terms,
        "n": len(clean),
        "rank": int(rank),
        "r2": r2,
        "rmse_log": math.sqrt(ss_res / len(clean)),
        "coefficients": coefficient_rows,
        "residuals": residual_rows,
    }


def write_model_outputs(output_dir, model):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_summary = {key: value for key, value in model.items() if key not in ("coefficients", "residuals")}
    (output_dir / "model_fit.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "model_coefficients.csv", model["coefficients"])
    write_csv(output_dir / "model_residuals.csv", model["residuals"])


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def plot_summary(summary_rows, output_dir, x_column, y_column="sob2_mean"):
    import matplotlib.pyplot as plt

    values = []
    for row in summary_rows:
        try:
            values.append((float(row[x_column]), float(row[y_column])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None

    values.sort()
    x, y = zip(*values)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{y_column}_vs_{x_column}.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
