#!/usr/bin/env python3
"""Generate training and trajectory comparison plots from saved model runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from artifacts import model_path, output_path
from postprocessing.case_registry import CASE_SPECS, read_manifest, read_run_config, transform_states
from postprocessing.plotting import aggregate_histories, plot_predictions, plot_training_metrics
from postprocessing.predict_saved_runs import predict_run
from postprocessing.run_discovery import discover_runs, filter_runs


def _parse_match(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()


def _matches(config: dict[str, Any], filters: list[tuple[str, str]]) -> bool:
    for key, requested in filters:
        value = config.get(key)
        if value is None:
            return False
        try:
            if not np.isclose(float(value), float(requested), rtol=1e-12, atol=1e-15):
                return False
        except (TypeError, ValueError):
            if str(value) != requested:
                return False
    return True


def _load_history(run: Path):
    return json.loads((output_path(run) / "training_history.json").read_text(encoding="utf-8"))


def _load_prediction(run: Path):
    path = output_path(run) / "trajectory_predictions.npz"
    if not path.exists():
        path = output_path(run) / "predictions.npz"
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--results-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("mlp", "pinn", "daehn"), default=("mlp", "pinn", "daehn"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate-seeds", action="store_true")
    parser.add_argument("--match", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force-predict", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    from project_paths import result_path
    args.output_dir = result_path(args.output_dir, Path(__file__).resolve().parents[2] / "train.py")
    roots = [*args.run_dir, *args.results_dir]
    if not roots:
        raise ValueError("provide --run-dir or --results-dir")
    runs = filter_runs(discover_runs(roots), models=set(args.models))
    filters = [_parse_match(text) for text in args.match]
    runs = [run for run in runs if _matches(read_run_config(run), filters)]
    if not runs:
        raise RuntimeError("no runs matched")
    examples = {int(read_manifest(run)["example"]) for run in runs}
    if len(examples) != 1:
        raise ValueError(f"plot one example at a time, found {sorted(examples)}")
    example = examples.pop()
    spec = CASE_SPECS[example]
    by_model: dict[str, list[Path]] = {}
    for run in runs:
        manifest = read_manifest(run)
        by_model.setdefault(manifest["model"], []).append(run)
    if args.seed is not None:
        by_model = {
            model: [run for run in group if int(read_manifest(run)["seed"]) == args.seed]
            for model, group in by_model.items()
        }
    elif not args.aggregate_seeds:
        common = None
        for group in by_model.values():
            seeds = {int(read_manifest(run)["seed"]) for run in group}
            common = seeds if common is None else common & seeds
        if not common:
            raise RuntimeError("no common seed across requested models; use --aggregate-seeds or --seed")
        selected_seed = min(common)
        by_model = {
            model: [run for run in group if int(read_manifest(run)["seed"]) == selected_seed]
            for model, group in by_model.items()
        }
        print(f"Using common seed {selected_seed}")

    histories = {}
    means = {}
    stds = {}
    exact_reference = None
    coordinate_reference = None
    for model, group in by_model.items():
        if not group:
            continue
        model_histories = [_load_history(run) for run in group]
        histories[model] = aggregate_histories(model_histories)
        predictions = []
        exacts = []
        coordinates = []
        for run in group:
            predict_run(run, force=bool(args.force_predict))
            payload = _load_prediction(run)
            predictions.append(transform_states(example, payload["prediction"]))
            exacts.append(transform_states(example, payload["exact"]))
            coordinates.append(payload["coordinates"])
        base_coordinate = coordinates[0]
        base_exact = exacts[0]
        for coordinate, exact in zip(coordinates[1:], exacts[1:]):
            if not np.array_equal(coordinate, base_coordinate):
                raise ValueError(f"evaluation grids differ for model {model}")
            if not np.allclose(exact, base_exact, rtol=1e-12, atol=1e-12):
                raise ValueError(f"ground truth differs for model {model}")
        stack = np.stack(predictions)
        means[model] = np.mean(stack, axis=0)
        stds[model] = np.std(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0])
        if exact_reference is None:
            exact_reference = base_exact
            coordinate_reference = base_coordinate
        else:
            if not np.array_equal(coordinate_reference, base_coordinate):
                raise ValueError("evaluation grids differ across models")
            if not np.allclose(exact_reference, base_exact, rtol=1e-12, atol=1e-12):
                raise ValueError("ground truth differs across models")

    output = args.output_dir.resolve()
    plot_training_metrics(output, histories)
    plot_predictions(
        output,
        coordinate_reference,
        exact_reference,
        means,
        spec.latex_labels,
        stds=stds if args.aggregate_seeds else None,
        layout=spec.layout,
    )
    print(json.dumps({"output_dir": str(output), "example": example, "models": sorted(means), "runs": sum(len(v) for v in by_model.values())}, indent=2))


if __name__ == "__main__":
    main()
