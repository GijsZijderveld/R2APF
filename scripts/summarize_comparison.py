#!/usr/bin/env python
"""Summarize raw results from the controlled all-planner comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "planner",
    "scenario",
    "seed",
    "success",
    "failure_reason",
    "collision_count",
    "executed_path_length",
    "execution_steps",
    "final_goal_distance",
    "planning_calls",
    "failed_planner_calls",
    "total_planning_time_s",
    "maximum_planning_time_s",
    "replans",
}

OPTIONAL_COUNT_DIAGNOSTICS = (
    "planner_unique_midpoint_rejections",
    "planner_selection_changes_due_to_midpoint_checking",
    "planner_repair_events",
    "planner_recovery_events",
    "planner_artificial_obstacles_inserted",
)

OPTIONAL_RATIO_DIAGNOSTICS = (
    "planner_path_reuse_ratio",
)


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not bool(valid.all()):
        invalid = sorted(normalized.loc[~valid].unique())
        raise ValueError(f"Invalid success values: {invalid}")
    return normalized.isin({"true", "1"})


def load_results(path: str | Path) -> pd.DataFrame:
    results_path = Path(path)
    frame = pd.read_csv(results_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"{results_path} is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{results_path} contains no result rows.")

    frame = frame.copy()
    frame["success"] = _as_boolean(frame["success"])
    frame["collision_episode"] = (
        pd.to_numeric(frame["collision_count"], errors="raise") > 0
    )
    return frame


def _numeric(group: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(group[column], errors="coerce").dropna()


def _distribution(record: dict, name: str, values: pd.Series) -> None:
    if values.empty:
        return
    record[f"{name}_mean"] = float(values.mean())
    record[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    record[f"{name}_median"] = float(values.median())
    record[f"{name}_p95"] = float(values.quantile(0.95))
    record[f"{name}_max"] = float(values.max())


def _summarize_group(group: pd.DataFrame) -> dict:
    successes = group.loc[group["success"]]
    planning_calls = _numeric(group, "planning_calls")
    planning_time = _numeric(group, "total_planning_time_s")

    record = {
        "episodes": int(len(group)),
        "successes": int(group["success"].sum()),
        "success_rate": float(group["success"].mean()),
        "collision_episodes": int(group["collision_episode"].sum()),
        "collision_episode_rate": float(group["collision_episode"].mean()),
        "planning_calls_total": int(planning_calls.sum()),
        "failed_planner_calls_total": int(
            _numeric(group, "failed_planner_calls").sum()
        ),
        "total_planning_time_s_sum": float(planning_time.sum()),
        "planning_time_per_call_s": (
            float(planning_time.sum() / planning_calls.sum())
            if planning_calls.sum() > 0
            else np.nan
        ),
    }

    for column in (
        "collision_count",
        "executed_path_length",
        "execution_steps",
        "final_goal_distance",
        "planning_calls",
        "failed_planner_calls",
        "replans",
        "total_planning_time_s",
        "maximum_planning_time_s",
    ):
        _distribution(record, column, _numeric(group, column))

    # Successful-path length is kept separate because failed episodes often
    # terminate early and would otherwise make a planner look artificially short.
    _distribution(
        record,
        "successful_executed_path_length",
        _numeric(successes, "executed_path_length"),
    )

    for column in OPTIONAL_COUNT_DIAGNOSTICS:
        if column not in group:
            continue
        values = _numeric(group, column)
        if values.empty:
            continue
        record[f"{column}_sum"] = float(values.sum())
        record[f"{column}_mean"] = float(values.mean())
        record[f"{column}_max"] = float(values.max())

    for column in OPTIONAL_RATIO_DIAGNOSTICS:
        if column not in group:
            continue
        values = _numeric(group, column)
        if values.empty:
            continue
        record[f"{column}_observations"] = int(len(values))
        record[f"{column}_mean"] = float(values.mean())
        record[f"{column}_median"] = float(values.median())

    return record


def summarize(results: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    records = []
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for keys, group in results.groupby(grouper, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record.update(_summarize_group(group))
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_failures(
    results: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    failures = results.loc[~results["success"]].copy()
    if failures.empty:
        return pd.DataFrame(
            columns=[*group_columns, "failure_reason", "count", "episode_rate"]
        )
    failures["failure_reason"] = failures["failure_reason"].fillna("unspecified")
    counts = (
        failures.groupby([*group_columns, "failure_reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        results.groupby(list(group_columns), dropna=False)
        .size()
        .rename("episodes")
        .reset_index()
    )
    counts = counts.merge(totals, on=list(group_columns), how="left")
    counts["episode_rate"] = counts["count"] / counts["episodes"]
    return counts.drop(columns="episodes")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a controlled planner-comparison CSV."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="experiments/comparisons/results_22092026.csv",
        help="Raw CSV produced by scripts/run_comparison.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/comparisons/summary",
        help="Directory for the generated summary CSV files.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results = load_results(args.input)
    output_dir = Path(args.output_dir)

    outputs = {
        output_dir / "summary_22092026_overall.csv": summarize(results, ["planner"]),
        output_dir / "summary_22092026_by_scenario.csv": summarize(
            results, ["planner", "scenario"]
        ),
        output_dir / "failure_reasons_22092026.csv": summarize_failures(
            results, ["planner", "scenario"]
        ),
    }
    for path, frame in outputs.items():
        _write(frame, path)
        print(f"Wrote {len(frame)} rows to {path}")


if __name__ == "__main__":
    main()
