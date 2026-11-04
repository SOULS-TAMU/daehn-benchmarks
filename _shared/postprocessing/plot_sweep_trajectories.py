#!/usr/bin/env python3
"""Plot parameter-configuration rows against physical-state columns."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from artifacts import model_path, output_path
from postprocessing.case_registry import CASE_SPECS, read_manifest, read_run_config, transform_states
from postprocessing.predict_saved_runs import predict_run
from postprocessing.run_discovery import discover_runs, filter_runs

COLORS = {"mlp": "g", "pinn": "b", "daehn": "r"}
LABELS = {"mlp": "MLP", "pinn": "PINN", "daehn": "DAE-HardNet"}


def _configuration(text: str):
    result = {}
    for part in text.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError("configuration must be KEY=VALUE pairs")
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _matches(config, requested):
    for key, value in requested.items():
        if key not in config:
            return False
        try:
            if not np.isclose(float(config[key]), float(value), rtol=1e-12, atol=1e-15):
                return False
        except (TypeError, ValueError):
            if str(config[key]) != value:
                return False
    return True


def _load(run):
    path = output_path(run) / "trajectory_predictions.npz"
    if not path.exists():
        path = output_path(run) / "predictions.npz"
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration", action="append", required=True, help="KEY=VALUE[,KEY=VALUE]")
    parser.add_argument("--models", nargs="+", choices=("mlp", "pinn", "daehn"), default=("mlp", "pinn", "daehn"))
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--stem", default="sweep_trajectory_comparison")
    args = parser.parse_args()
    from project_paths import result_path
    args.output_dir = result_path(args.output_dir, Path(__file__).resolve().parents[2] / "train.py")

    configurations = [_configuration(text) for text in args.configuration]
    all_runs = filter_runs(discover_runs([args.results_dir]), models=set(args.models))
    if not all_runs:
        raise RuntimeError("no runs found")
    examples = {int(read_manifest(run)["example"]) for run in all_runs}
    if len(examples) != 1:
        raise ValueError("one example per figure is required")
    example = examples.pop()
    spec = CASE_SPECS[example]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams.update({"font.size": 16, "axes.linewidth": 1.5, "xtick.major.size": 6, "xtick.major.width": 1.5, "ytick.major.size": 6, "ytick.major.width": 1.5, "legend.frameon": False, "pdf.fonttype": 42})
    rows, cols = len(configurations), len(spec.latex_labels)
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5.5*rows), squeeze=False, sharex="col")

    for row_index, requested in enumerate(configurations):
        matched = [run for run in all_runs if _matches(read_run_config(run), requested)]
        if not matched:
            raise RuntimeError(f"no runs matched configuration {requested}")
        by_model = {}
        for run in matched:
            by_model.setdefault(read_manifest(run)["model"], []).append(run)
        exact_ref = None
        coord_ref = None
        aggregates = {}
        spreads = {}
        for model, group in by_model.items():
            predictions = []
            for run in group:
                predict_run(run, force=bool(args.force_predict))
                payload = _load(run)
                coordinate = payload["coordinates"]
                exact = transform_states(example, payload["exact"])
                prediction = transform_states(example, payload["prediction"])
                if coord_ref is None:
                    coord_ref, exact_ref = coordinate, exact
                elif not np.array_equal(coord_ref, coordinate) or not np.allclose(exact_ref, exact, rtol=1e-12, atol=1e-12):
                    raise ValueError("evaluation grids or ground truth differ within a configuration")
                predictions.append(prediction)
            stack = np.stack(predictions)
            aggregates[model] = np.mean(stack, axis=0)
            spreads[model] = np.std(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(stack[0])
        order = np.argsort(coord_ref[:, 0])
        time = coord_ref[order, 0]
        for column, label in enumerate(spec.latex_labels):
            axis = axes[row_index, column]
            axis.plot(time, exact_ref[order, column], color="k", lw=1.2, label="Ground Truth")
            for model in ("mlp", "pinn", "daehn"):
                if model not in aggregates:
                    continue
                mean = aggregates[model][order, column]
                std = spreads[model][order, column]
                axis.plot(time, mean, color=COLORS[model], lw=1.0, label=LABELS[model])
                axis.fill_between(time, mean-std, mean+std, color=COLORS[model], alpha=0.15, linewidth=0)
            if row_index == rows-1:
                axis.set_xlabel(r"$t$", fontsize=22)
            if row_index == 0:
                axis.set_title(label, fontsize=22)
            axis.tick_params(axis="both", labelsize=16)
        row_label = ", ".join(f"{key}={value}" for key, value in requested.items())
        axes[row_index, 0].text(-0.28, 0.5, row_label, transform=axes[row_index, 0].transAxes, rotation=90, ha="center", va="center", fontsize=16)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=18, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0.03, 0.07, 1, 1])
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    plt.show()
    plt.close(fig)
    print(json.dumps({"example": example, "configurations": configurations, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
