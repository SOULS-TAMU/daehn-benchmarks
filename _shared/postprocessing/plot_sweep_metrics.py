#!/usr/bin/env python3
"""Plot mean metric with seed standard deviation across a swept parameter."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from postprocessing.predict_saved_runs import predict_run
from postprocessing.run_discovery import discover_runs, filter_runs, load_manifest

COLORS = {"mlp": "g", "pinn": "b", "daehn": "r"}
LABELS = {"mlp": "MLP", "pinn": "PINN", "daehn": "DAE-HardNet"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x", required=True, help="run_config parameter")
    parser.add_argument("--metric", default="mse_data")
    parser.add_argument("--models", nargs="+", choices=("mlp", "pinn", "daehn"), default=("mlp", "pinn", "daehn"))
    parser.add_argument("--force-predict", action="store_true")
    args = parser.parse_args()
    from project_paths import result_path
    args.output_dir = result_path(args.output_dir, Path(__file__).resolve().parents[2] / "train.py")
    runs = filter_runs(discover_runs([args.results_dir]), models=set(args.models))
    grouped = defaultdict(list)
    for run in runs:
        metrics = predict_run(run, force=bool(args.force_predict))
        config = metrics.get("run_config", {})
        if args.x not in config:
            continue
        grouped[(metrics["model"], float(config[args.x]))].append(float(metrics.get(args.metric, np.nan)))
    if not grouped:
        raise RuntimeError("no matching metric data")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    for model in ("mlp", "pinn", "daehn"):
        points = sorted((x, values) for (name, x), values in grouped.items() if name == model)
        if not points:
            continue
        xs = np.asarray([x for x, _ in points])
        means = np.asarray([np.nanmean(values) for _, values in points])
        stds = np.asarray([np.nanstd(values, ddof=1) if np.isfinite(values).sum() > 1 else 0.0 for _, values in points])
        ax.errorbar(xs, means, yerr=stds, color=COLORS[model], marker="o", lw=1.0, capsize=3, label=LABELS[model])
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.metric)
    if np.all([value > 0 for values in grouped.values() for value in values if np.isfinite(value)]):
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.metric}_vs_{args.x}"
    plt.show()
    plt.close(fig)
    print(json.dumps({"groups": len(grouped), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
