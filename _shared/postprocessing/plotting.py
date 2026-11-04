"""Manuscript-style offline plotting for saved DAEHN case studies."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

MODEL_ORDER = ("mlp", "pinn", "daehn")
MODEL_DISPLAY = {"mlp": "MLP", "pinn": "PINN", "daehn": "DAE-HardNet"}
MODEL_COLORS = {"mlp": "g", "pinn": "b", "daehn": "r"}


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams.update({
        "font.size": 16,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def _save(fig, output_dir: Path, stem: str):
    import matplotlib.pyplot as plt
    plt.show()


def _positive(values):
    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array) & (array > 0), np.maximum(array, 1e-12), np.nan)


def aggregate_histories(histories: Sequence[Sequence[dict]]) -> list[dict]:
    if not histories:
        return []
    by_epoch: dict[int, list[dict]] = {}
    for history in histories:
        for row in history:
            by_epoch.setdefault(int(row["epoch"]), []).append(row)
    result = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        keys = set().union(*(row.keys() for row in rows))
        merged = {"epoch": epoch}
        for key in keys:
            if key in {"epoch", "phase"}:
                continue
            values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
            merged[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        result.append(merged)
    return result


def plot_training_metrics(output_dir: Path, histories: Mapping[str, Sequence[dict]]):
    plt = _setup()
    fig, (ax_rmse, ax_violation) = plt.subplots(1, 2, figsize=(12, 4), dpi=600)
    for model in MODEL_ORDER:
        if model not in histories:
            continue
        rows = list(histories[model])
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        train_data = _positive([row.get("train_data_loss", np.nan) for row in rows])
        val_data = _positive([row.get("validation_data_loss", np.nan) for row in rows])
        train_rmse = np.sqrt(train_data)
        val_rmse = np.sqrt(val_data)
        color = MODEL_COLORS[model]
        label = MODEL_DISPLAY[model]
        ax_rmse.plot(epochs, train_rmse, color=color, lw=0.5, label=f"{label} (train)")
        ax_rmse.plot(epochs, val_rmse, color=color, lw=0.5, ls="--", label=f"{label} (val)")
        train_v = _positive([row.get("train_violation", np.nan) for row in rows])
        val_v = _positive([row.get("validation_violation", np.nan) for row in rows])
        if np.isfinite(train_v).any():
            ax_violation.plot(epochs, train_v, color=color, lw=0.5, label=f"{label} (train)")
        if np.isfinite(val_v).any():
            ax_violation.plot(epochs, val_v, color=color, lw=0.5, ls="--", label=f"{label} (val)")
    ax_rmse.set(xlabel="Epoch", ylabel="RMSE")
    ax_violation.set(xlabel="Epoch", ylabel="Absolute Violation")
    ax_rmse.set_yscale("log")
    ax_violation.set_yscale("log")
    handles, labels = ax_rmse.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)//2), fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, output_dir, "all_models_rmse_and_violation")
    plt.close(fig)


def _axes_for_states(plt, n_states: int, layout: str):
    if layout == "centered_4_3" and n_states == 7:
        fig = plt.figure(figsize=(28, 12))
        grid = fig.add_gridspec(2, 8)
        axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[0, 6:8]), fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3:5]), fig.add_subplot(grid[1, 5:7])]
        return fig, axes
    cols = min(4, n_states)
    rows = int(math.ceil(n_states / cols))
    fig, axes_array = plt.subplots(rows, cols, figsize=(7*cols, 6*rows), squeeze=False, sharex=True)
    axes = list(axes_array.ravel())
    for axis in axes[n_states:]:
        axis.set_visible(False)
    return fig, axes[:n_states]


def plot_predictions(
    output_dir: Path,
    coordinates: np.ndarray,
    exact: np.ndarray,
    means: Mapping[str, np.ndarray],
    labels: Sequence[str],
    *,
    stds: Mapping[str, np.ndarray] | None = None,
    layout: str = "auto",
    stem: str = "all_models_prediction_comparison",
):
    plt = _setup()
    coordinate = np.asarray(coordinates)[:, 0]
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    exact = np.asarray(exact)[order]
    means = {key: np.asarray(value)[order] for key, value in means.items()}
    stds = {key: np.asarray(value)[order] for key, value in (stds or {}).items()}
    fig, axes = _axes_for_states(plt, len(labels), layout)
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(coordinate, exact[:, index], color="k", lw=1.2, label="Ground Truth")
        for model in MODEL_ORDER:
            if model not in means:
                continue
            color = MODEL_COLORS[model]
            values = means[model][:, index]
            axis.plot(coordinate, values, color=color, lw=1.0, label=MODEL_DISPLAY[model])
            if model in stds:
                spread = stds[model][:, index]
                axis.fill_between(coordinate, values-spread, values+spread, color=color, alpha=0.15, linewidth=0)
        axis.set_xlabel(r"$t$", fontsize=24)
        axis.set_ylabel(label, fontsize=24)
        axis.tick_params(axis="both", which="major", labelsize=18)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", fontsize=24, ncol=len(legend_labels), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, output_dir, stem)
    plt.close(fig)