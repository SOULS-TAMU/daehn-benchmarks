#!/usr/bin/env python3
"""Load trained JAX checkpoints, predict trajectories, and save metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))
from project_paths import find_project_root
PROJECT_ROOT = find_project_root(EXAMPLES_ROOT)
for _candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from case_study_utils import evaluate_trainer_predictions, load_parameter_checkpoint
from artifacts import model_path, output_path
from postprocessing.case_registry import build_inference_trainer, read_manifest
from postprocessing.run_discovery import discover_runs, filter_runs


def _prediction_signature(run_dir: Path, manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(manifest.get("checkpoint_sha256", "")).encode())
    digest.update(str(manifest.get("evaluation_data_sha256", "")).encode())
    digest.update(str(manifest.get("trajectory_data_sha256", "")).encode())
    digest.update(str(manifest.get("run_signature", "")).encode())
    return digest.hexdigest()


def predict_run(run_dir: Path, *, force: bool = False) -> dict[str, Any]:
    run_dir = model_path(run_dir)
    if not run_dir.is_relative_to((PROJECT_ROOT / "results").resolve()):
        raise ValueError("Saved model prediction must write inside central results")
    manifest = read_manifest(run_dir)
    destination = output_path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    signature = _prediction_signature(run_dir, manifest)
    metrics_path = destination / "metrics.json"
    predictions_path = destination / "predictions.npz"
    trajectory_predictions_path = destination / "trajectory_predictions.npz"
    if not force and metrics_path.exists() and predictions_path.exists() and trajectory_predictions_path.exists():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("prediction_signature") == signature:
            return cached
    trainer, projected, manifest, config, data, _module = build_inference_trainer(run_dir)
    parameters = load_parameter_checkpoint(run_dir)
    payload, computed = evaluate_trainer_predictions(
        trainer,
        parameters,
        coordinates=data["coordinates"],
        physical_parameters=data["physical_parameters"],
        targets=data["targets"],
        derivative_targets=data["derivative_targets"],
        projected=projected,
        model_name=manifest["model"],
    )
    np.savez(predictions_path, **payload)
    trajectory_path = run_dir / manifest.get("trajectory_data_file", "trajectory_data.npz")
    with np.load(trajectory_path) as archive:
        trajectory_data = {name: np.asarray(archive[name]) for name in archive.files}
    trajectory_payload, _trajectory_metrics = evaluate_trainer_predictions(
        trainer,
        parameters,
        coordinates=trajectory_data["coordinates"],
        physical_parameters=trajectory_data["physical_parameters"],
        targets=trajectory_data["targets"],
        derivative_targets=trajectory_data["derivative_targets"],
        projected=projected,
        model_name=manifest["model"],
    )
    np.savez(trajectory_predictions_path, **trajectory_payload)
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "prediction_signature": signature,
        "status": "evaluated",
        "example": int(manifest["example"]),
        "system": manifest["system"],
        "model": manifest["model"],
        "model_display": manifest["model_display"],
        "seed": int(manifest["seed"]),
        "projection": manifest["projection"],
        "training_wall_seconds": float(manifest["training_wall_seconds"]),
        "epochs": int(manifest["epochs"]),
        "pretrain_epochs": int(manifest["pretrain_epochs"]),
        "run_signature": manifest["run_signature"],
        "run_config": config,
        "split_info": manifest.get("split_info", {}),
        **computed,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--results-dir", type=Path, action="append", default=[])
    parser.add_argument("--models", nargs="+", choices=("mlp", "pinn", "daehn"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--example", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [*args.run_dir, *args.results_dir]
    if not roots:
        raise ValueError("provide --run-dir or --results-dir")
    runs = discover_runs(roots)
    runs = filter_runs(
        runs,
        models=set(args.models) if args.models else None,
        seeds=set(args.seeds) if args.seeds else None,
        example=args.example,
    )
    if not runs:
        raise RuntimeError("no trained runs matched the filters")
    rows = []
    for run in runs:
        print(f"[predict] {run}")
        metrics = predict_run(run, force=bool(args.force))
        rows.append({"run_dir": str(run), **metrics})
    print(json.dumps({"runs_evaluated": len(rows), "runs": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
