#!/usr/bin/env python3
"""Compile per-run and grouped mean/variance tables from evaluated saved runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from artifacts import model_path, output_path
from postprocessing.predict_saved_runs import predict_run
from postprocessing.run_discovery import discover_runs, filter_runs, load_manifest

METRICS = (
    "mse_data",
    "mse_derivative",
    "physics_loss_relative",
    "algebraic_violation_relative",
    "relative_state_error",
    "inference_time_seconds_per_batch",
)
EXCLUDE_AUTO = {
    "seed", "output_dir", "data_dir", "source_csv", "print_every", "validation_every",
    "plot", "plots", "smoke", "reuse_data", "train_limit", "validation_limit", "test_limit",
}


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _load_metrics(run: Path, force_predict: bool) -> dict[str, Any]:
    path = output_path(run) / "metrics.json"
    if force_predict or not path.exists():
        return predict_run(run, force=force_predict)
    return json.loads(path.read_text(encoding="utf-8"))


def _auto_group_keys(rows: list[dict[str, Any]]) -> list[str]:
    # A key is a swept hyperparameter only when it varies within at least one
    # model family.  This avoids treating model-specific defaults such as
    # lambda_derivative=0 for MLP/PINN as sweep axes.
    candidates: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        model = str(row["model"])
        for key, value in row["run_config"].items():
            if key in EXCLUDE_AUTO or key in {"model", "projection"} or not _scalar(value):
                continue
            candidates[key][model].add(json.dumps(value, sort_keys=True))
    return [
        key
        for key, per_model in sorted(candidates.items())
        if any(len(values) > 1 for values in per_model.values())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return "NaN" if not math.isfinite(value) else f"{value:.6e}"
    return str(value)


def _write_md(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(keys) + " |\n")
        handle.write("|" + "|".join(["---"]*len(keys)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(_format(row.get(key, "")) for key in keys) + " |\n")


def _write_tex(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0])
    escaped = [key.replace("_", r"\_") for key in keys]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(r"\begin{tabular}{" + "l"*len(keys) + "}\n")
        handle.write(r"\toprule" + "\n")
        handle.write(" & ".join(escaped) + r" \\" + "\n")
        handle.write(r"\midrule" + "\n")
        for row in rows:
            handle.write(" & ".join(_format(row.get(key, "")) for key in keys) + r" \\" + "\n")
        handle.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--results-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("mlp", "pinn", "daehn"))
    parser.add_argument("--example", type=int)
    parser.add_argument("--group-by", action="append", default=[], help="run_config key; repeat")
    parser.add_argument("--force-predict", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    from project_paths import result_path
    args.output_dir = result_path(args.output_dir, Path(__file__).resolve().parents[2] / "train.py")
    roots = [*args.run_dir, *args.results_dir]
    if not roots:
        raise ValueError("provide --run-dir or --results-dir")
    runs = filter_runs(discover_runs(roots), models=set(args.models) if args.models else None, example=args.example)
    if not runs:
        raise RuntimeError("no runs matched")
    records = []
    seen = set()
    signatures = defaultdict(set)
    for run in runs:
        manifest = load_manifest(run)
        signature = manifest.get("run_signature", str(run))
        if signature in seen:
            continue
        seen.add(signature)
        metrics = _load_metrics(run, bool(args.force_predict))
        manifest = load_manifest(run)
        record = {
            "run_dir": str(run),
            "example": int(metrics["example"]),
            "system": metrics["system"],
            "model": metrics["model"],
            "seed": int(metrics["seed"]),
            "run_signature": metrics.get("run_signature", manifest.get("run_signature")),
            "run_config": metrics.get("run_config", {}),
            **{name: metrics.get(name, float("nan")) for name in METRICS},
        }
        records.append(record)
        signatures[(record["example"], record["system"], record["model"])].add(record["run_signature"])

    group_keys = list(args.group_by) or _auto_group_keys(records)
    per_run = []
    for row in records:
        flat = {key: value for key, value in row.items() if key != "run_config"}
        for key in group_keys:
            flat[key] = row["run_config"].get(key, float("nan"))
        per_run.append(flat)

    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (row["example"], row["system"], row["model"], *[row["run_config"].get(name) for name in group_keys])
        grouped[key].append(row)
    aggregate = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        base = {"example": key[0], "system": key[1], "model": key[2]}
        for name, value in zip(group_keys, key[3:]):
            base[name] = value if value is not None else float("nan")
        base["N"] = len(members)
        for metric in METRICS:
            values = np.asarray([member[metric] for member in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                stats = {"mean": np.nan, "variance": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
            else:
                stats = {
                    "mean": float(np.mean(finite)),
                    "variance": float(np.var(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "median": float(np.median(finite)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                }
            for stat, value in stats.items():
                base[f"{metric}_{stat}"] = value
        aggregate.append(base)

    print(json.dumps({"runs": len(records), "groups": len(aggregate), "group_by": group_keys, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
