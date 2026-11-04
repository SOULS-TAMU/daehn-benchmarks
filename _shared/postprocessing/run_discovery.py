"""Discover self-describing trained model directories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def is_run_dir(path: Path) -> bool:
    return (Path(path) / "checkpoint_manifest.json").exists()


def discover_runs(paths: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if is_run_dir(path):
            found.add(path)
            continue
        if path.exists():
            for manifest in path.rglob("checkpoint_manifest.json"):
                found.add(manifest.parent.resolve())
    return sorted(found)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads((Path(path) / "checkpoint_manifest.json").read_text(encoding="utf-8"))


def filter_runs(
    runs: Iterable[Path],
    *,
    models: set[str] | None = None,
    seeds: set[int] | None = None,
    example: int | None = None,
) -> list[Path]:
    result = []
    for run in runs:
        manifest = load_manifest(run)
        if models is not None and manifest.get("model") not in models:
            continue
        if seeds is not None and int(manifest.get("seed", -1)) not in seeds:
            continue
        if example is not None and int(manifest.get("example", -1)) != int(example):
            continue
        result.append(Path(run))
    return result
