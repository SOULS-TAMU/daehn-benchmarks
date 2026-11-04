"""Shared, reproducible and efficient runtime for DAEHN case studies 08 and 10--17.

This module changes only the experiment protocol and execution path.  Each
case-study script keeps its original equations, Taylor-neighborhood
reconstruction, variable scaling, and distance-minimization projection.

Training objectives
-------------------
MLP
    lambda_data * data MSE
PINN
    lambda_data * data MSE + lambda_physics * physical residual MSE
DAEHN pretraining
    lambda_data * data MSE
DAEHN projected phase
    lambda_data * data MSE
    + lambda_derivative * derivative-label MSE
    + lambda_physics * physical residual MSE

The data-only path never evaluates derivatives, physical residuals, initial
conditions, or the projection.  PINN never evaluates derivative-label MSE.
All train/validation/test samples are selected from the same configured
coordinate horizon.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from artifacts import model_path, output_path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from daehn.dae import DAETrainer, DAETrainingConfig, DAETrainingResult
from daehn.dae.model import extract_network_derivatives
from daehn.train.projected_neural import apply_updates

Array = Any
PyTree = Any
MODEL_ORDER = ("mlp", "pinn", "daehn")
MODEL_DISPLAY = {"mlp": "MLP", "pinn": "PINN", "daehn": "DAE-HardNet"}
TABLE_COLUMNS = (
    "Model",
    "MSE (data)",
    "MSE (Derivative)",
    "Physics Loss (Relative)",
    "Algebraic Violation (Relative)",
    "Relative State Error",
    "Inference Time (s/batch)",
)


def set_reproducible_seed(seed: int) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class CaseStudyTrainingConfig(DAETrainingConfig):
    data_weight: float = 1.0
    physics_weight: float = 1.0
    validation_every: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.data_weight < 0.0:
            raise ValueError("data_weight must be nonnegative")
        if self.physics_weight < 0.0:
            raise ValueError("physics_weight must be nonnegative")
        if self.validation_every < 1:
            raise ValueError("validation_every must be positive")


class CaseMetrics(NamedTuple):
    loss: Array
    data_loss: Array
    derivative_loss: Array
    physics_loss: Array
    differential_violation: Array
    algebraic_violation: Array
    inequality_violation: Array
    projection_converged_fraction: Array
    gradient_eligible_fraction: Array
    mean_projection_iterations: Array
    update_accepted: Array


def _masked_mean(values: Array, mask: Array) -> Array:
    values = jnp.asarray(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _tree_all_finite(tree: PyTree) -> Array:
    result = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        result = result & jnp.all(jnp.isfinite(leaf))
    return result


def _tree_select(condition: Array, candidate: PyTree, original: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(condition, new, old), candidate, original
    )


def _zero(dtype=jnp.float64) -> Array:
    return jnp.asarray(0.0, dtype=dtype)


def _mean_norm(values: Array, mask: Array) -> Array:
    values = jnp.asarray(values)
    if values.shape[-1] == 0:
        return _zero(values.dtype)
    return _masked_mean(jnp.linalg.norm(values, axis=-1), mask)


def _physics_mse(blocks: tuple[Array, Array, Array, Array], mask: Array) -> Array:
    combined = blocks[-1]
    if combined.shape[-1] == 0:
        return _zero(combined.dtype)
    return _masked_mean(jnp.mean(combined**2, axis=-1), mask)


def _metrics_to_dict(metrics: CaseMetrics) -> dict[str, float]:
    host = jax.device_get(metrics)
    return {name: float(value) for name, value in zip(metrics._fields, host)}


def _batch_plan(n: int, batch_size: int, indices: np.ndarray | None = None):
    if n < 1:
        raise ValueError("cannot batch an empty dataset")
    if indices is None:
        indices = np.arange(n, dtype=np.int32)
    else:
        indices = np.asarray(indices, dtype=np.int32)
    n_batches = int(math.ceil(len(indices) / batch_size))
    total = n_batches * batch_size
    padded = np.empty(total, dtype=np.int32)
    mask = np.zeros(total, dtype=bool)
    padded[: len(indices)] = indices
    mask[: len(indices)] = True
    if len(indices) < total:
        padded[len(indices) :] = indices[-1]
    return padded.reshape(n_batches, batch_size), mask.reshape(n_batches, batch_size)


def _aggregate_scanned(metrics: CaseMetrics, masks: Array) -> CaseMetrics:
    counts = jnp.sum(masks, axis=1, dtype=jnp.float64)
    denominator = jnp.maximum(jnp.sum(counts), 1.0)
    return jax.tree_util.tree_map(
        lambda values: jnp.sum(values * counts) / denominator,
        metrics,
    )


class WeightedLossTrainerMixin:
    """Efficient model-specific steps layered on the existing DAE trainer.

    The mixin deliberately does not modify the projection model.  It only
    selects which computations are executed for each model family.
    """

    def _build_steps(self) -> None:  # called by DAETrainer.__init__
        config = self.config
        data_weight = float(getattr(config, "data_weight", 1.0))
        physics_weight = float(getattr(config, "physics_weight", config.violation_weight))
        derivative_weight = float(config.derivative_weight)

        def data_compute(parameters, coordinates, _physical_parameters, targets, _derivative_targets, mask):
            states = self.backbone.apply(parameters, coordinates)
            data_loss = _masked_mean(jnp.mean((states - targets) ** 2, axis=-1), mask)
            loss = data_weight * data_loss
            one = jnp.asarray(1.0, dtype=states.dtype)
            zero = _zero(states.dtype)
            nan = jnp.asarray(jnp.nan, dtype=states.dtype)
            metrics = CaseMetrics(loss, data_loss, nan, nan, nan, nan, nan, one, one, zero, one)
            return loss, metrics

        def pinn_compute(parameters, coordinates, physical_parameters, targets, _derivative_targets, mask):
            states = self.backbone.apply(parameters, coordinates)
            derivatives = extract_network_derivatives(
                self.backbone.apply,
                parameters,
                coordinates,
                self.compiled.discretization.derivatives,
                self.compiled.system.differential_states,
            )
            blocks = self._physical_blocks(coordinates, states, derivatives, physical_parameters)
            data_loss = _masked_mean(jnp.mean((states - targets) ** 2, axis=-1), mask)
            physics_loss = _physics_mse(blocks, mask)
            loss = data_weight * data_loss + physics_weight * physics_loss
            differential, algebraic, inequalities, _combined = blocks
            one = jnp.asarray(1.0, dtype=states.dtype)
            zero = _zero(states.dtype)
            nan = jnp.asarray(jnp.nan, dtype=states.dtype)
            metrics = CaseMetrics(
                loss,
                data_loss,
                nan,
                physics_loss,
                _mean_norm(differential, mask),
                _mean_norm(algebraic, mask),
                _mean_norm(inequalities, mask),
                one,
                one,
                zero,
                one,
            )
            return loss, metrics

        def projected_compute(parameters, coordinates, physical_parameters, targets, derivative_targets, mask):
            decoded, info, _context, _target = self.model.forward(
                parameters, coordinates, physical_parameters
            )
            states = decoded["states"]
            finite_mask = mask & info.forward_state_finite
            model_derivatives = self.full_model_derivative_plan.evaluate(
                self.full_model_derivative_model,
                parameters,
                coordinates,
                physical_parameters,
                center_states=states,
            )
            data_loss = _masked_mean(
                jnp.mean((states - targets) ** 2, axis=-1), finite_mask
            )
            if model_derivatives.shape[-1] == 0:
                derivative_loss = _zero(states.dtype)
            else:
                derivative_loss = _masked_mean(
                    jnp.mean((model_derivatives - derivative_targets) ** 2, axis=-1),
                    finite_mask,
                )
            blocks = self._physical_blocks(
                coordinates, states, model_derivatives, physical_parameters
            )
            physics_loss = _physics_mse(blocks, finite_mask)
            loss = (
                data_weight * data_loss
                + derivative_weight * derivative_loss
                + physics_weight * physics_loss
            )
            differential, algebraic, inequalities, _combined = blocks
            metrics = CaseMetrics(
                loss,
                data_loss,
                derivative_loss,
                physics_loss,
                _mean_norm(differential, finite_mask),
                _mean_norm(algebraic, finite_mask),
                _mean_norm(inequalities, finite_mask),
                _masked_mean(info.forward_converged.astype(jnp.float64), mask),
                _masked_mean(info.gradient_eligible.astype(jnp.float64), mask),
                _masked_mean(info.forward_iterations.astype(jnp.float64), mask),
                jnp.asarray(1.0, dtype=states.dtype),
            )
            return loss, metrics

        def make_update(compute):
            def update(parameters, optimizer_state, coordinates, physical_parameters, targets, derivative_targets, mask):
                (loss, metrics), gradients = jax.value_and_grad(compute, has_aux=True)(
                    parameters,
                    coordinates,
                    physical_parameters,
                    targets,
                    derivative_targets,
                    mask,
                )
                updates, candidate_optimizer_state = self.optimizer.update(
                    gradients, optimizer_state, parameters
                )
                candidate_parameters = apply_updates(parameters, updates)
                accepted = (
                    jnp.isfinite(loss)
                    & _tree_all_finite(gradients)
                    & _tree_all_finite(candidate_parameters)
                )
                parameters = _tree_select(accepted, candidate_parameters, parameters)
                optimizer_state = _tree_select(
                    accepted, candidate_optimizer_state, optimizer_state
                )
                metrics = metrics._replace(update_accepted=accepted.astype(jnp.float64))
                return parameters, optimizer_state, metrics

            return update

        def make_epoch_update(compute):
            update = make_update(compute)

            @jax.jit
            def epoch_update(parameters, optimizer_state, coordinates, physical_parameters, targets, derivative_targets, batch_indices, batch_masks):
                def body(carry, batch):
                    current_parameters, current_optimizer_state = carry
                    indices, mask = batch
                    current_parameters, current_optimizer_state, metrics = update(
                        current_parameters,
                        current_optimizer_state,
                        coordinates[indices],
                        physical_parameters[indices],
                        targets[indices],
                        derivative_targets[indices],
                        mask,
                    )
                    return (current_parameters, current_optimizer_state), metrics

                (parameters, optimizer_state), scanned = jax.lax.scan(
                    body, (parameters, optimizer_state), (batch_indices, batch_masks)
                )
                return parameters, optimizer_state, _aggregate_scanned(scanned, batch_masks)

            return epoch_update

        def make_epoch_eval(compute):
            @jax.jit
            def epoch_eval(parameters, coordinates, physical_parameters, targets, derivative_targets, batch_indices, batch_masks):
                def body(batch):
                    indices, mask = batch
                    _loss, metrics = compute(
                        parameters,
                        coordinates[indices],
                        physical_parameters[indices],
                        targets[indices],
                        derivative_targets[indices],
                        mask,
                    )
                    return metrics

                scanned = jax.lax.map(body, (batch_indices, batch_masks))
                return _aggregate_scanned(scanned, batch_masks)

            return epoch_eval

        self.case_train_steps = {
            "data": make_epoch_update(data_compute),
            "pinn": make_epoch_update(pinn_compute),
            "projected": make_epoch_update(projected_compute),
        }
        self.case_eval_steps = {
            "data": make_epoch_eval(data_compute),
            "pinn": make_epoch_eval(pinn_compute),
            "projected": make_epoch_eval(projected_compute),
        }
        self.case_compute = {
            "data": jax.jit(data_compute),
            "pinn": jax.jit(pinn_compute),
            "projected": jax.jit(projected_compute),
        }


class WeightedDAETrainer(WeightedLossTrainerMixin, DAETrainer):
    pass


class _TrainingLogger:
    def __init__(self, label: str, mode: str, epochs: int, print_every: int):
        self.label = label
        self.mode = mode
        self.epochs = int(epochs)
        self.print_every = int(print_every)

    def start(self, train_samples: int, validation_samples: int, batch_size: int) -> None:
        print(f"\n{self.label}")
        print(
            f"  samples: train={train_samples} validation={validation_samples} "
            f"batch_size={batch_size}"
        )
        if self.print_every <= 0:
            return
        if self.mode == "data":
            print("=" * 86)
            print("      Epoch |       Training Data |     Validation Data")
            print("=" * 86)
        elif self.mode == "pinn":
            print("=" * 158)
            print(
                "      Epoch |  Train Data  Train Physics      Train DE     Train Alg | "
                "  Val Data    Val Physics        Val DE       Val Alg"
            )
            print("=" * 158)
        else:
            print("=" * 188)
            print(
                "      Epoch |  Train Data Train Derivative Train Physics      Train DE     Train Alg | "
                "  Val Data   Val Derivative   Val Physics        Val DE       Val Alg | Conv."
            )
            print("=" * 188)

    def epoch(self, epoch: int, train: Mapping[str, float], validation: Mapping[str, float]) -> None:
        if self.print_every <= 0:
            return
        if epoch != 1 and epoch != self.epochs and epoch % self.print_every:
            return
        if self.mode == "data":
            print(
                f"{epoch:7d}/{self.epochs:<7d} | {train['data_loss']:19.6e} | "
                f"{validation['data_loss']:19.6e}"
            )
        elif self.mode == "pinn":
            print(
                f"{epoch:7d}/{self.epochs:<7d} | {train['data_loss']:11.4e} "
                f"{train['physics_loss']:14.4e} {train['differential_violation']:13.4e} "
                f"{train['algebraic_violation']:13.4e} | {validation['data_loss']:11.4e} "
                f"{validation['physics_loss']:14.4e} {validation['differential_violation']:13.4e} "
                f"{validation['algebraic_violation']:13.4e}"
            )
        else:
            print(
                f"{epoch:7d}/{self.epochs:<7d} | {train['data_loss']:11.4e} "
                f"{train['derivative_loss']:16.4e} {train['physics_loss']:13.4e} "
                f"{train['differential_violation']:13.4e} {train['algebraic_violation']:13.4e} | "
                f"{validation['data_loss']:11.4e} {validation['derivative_loss']:16.4e} "
                f"{validation['physics_loss']:13.4e} {validation['differential_violation']:13.4e} "
                f"{validation['algebraic_violation']:13.4e} | "
                f"{validation['projection_converged_fraction']:5.3f}"
            )

    @staticmethod
    def finish(wall_seconds: float) -> None:
        print(f"Training wall time = {wall_seconds:.4f} seconds")


def _to_device_dataset(*arrays: np.ndarray) -> tuple[Array, ...]:
    return tuple(jnp.asarray(np.asarray(value, dtype=np.float64)) for value in arrays)


def _fit_mode(
    trainer,
    *,
    mode: str,
    label: str,
    epochs: int,
    train_data: tuple[np.ndarray, ...],
    validation_data: tuple[np.ndarray, ...],
    initial_parameters: PyTree | None,
    seed: int,
    print_every: int,
    validation_every: int,
    checkpoint_epoch: int | None = None,
) -> tuple[DAETrainingResult, PyTree | None]:
    if mode not in {"data", "pinn", "projected"}:
        raise ValueError(f"unsupported mode {mode}")
    train_coordinates, train_parameters, train_targets, train_derivatives = train_data
    val_coordinates, val_parameters, val_targets, val_derivatives = validation_data
    train_device = _to_device_dataset(*train_data)
    validation_device = _to_device_dataset(*validation_data)

    if initial_parameters is None:
        parameters = trainer.backbone.init(
            jax.random.PRNGKey(int(seed)),
            train_coordinates.shape[1],
            train_targets.shape[1],
            jnp.float64,
        )
    else:
        parameters = initial_parameters
    optimizer_state = trainer.optimizer.init(parameters)

    train_indices = np.arange(len(train_coordinates), dtype=np.int32)
    validation_indices, validation_masks = _batch_plan(
        len(val_coordinates), trainer.config.batch_size
    )
    validation_indices = jnp.asarray(validation_indices)
    validation_masks = jnp.asarray(validation_masks)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    logger = _TrainingLogger(label, mode, epochs, print_every)
    logger.start(len(train_coordinates), len(val_coordinates), trainer.config.batch_size)
    start = time.perf_counter()
    last_validation: dict[str, float] | None = None
    checkpoint_parameters = None

    for epoch in range(1, int(epochs) + 1):
        if trainer.config.shuffle:
            rng.shuffle(train_indices)
        batch_indices, batch_masks = _batch_plan(
            len(train_coordinates), trainer.config.batch_size, train_indices
        )
        parameters, optimizer_state, train_metrics_device = trainer.case_train_steps[mode](
            parameters,
            optimizer_state,
            *train_device,
            jnp.asarray(batch_indices),
            jnp.asarray(batch_masks),
        )
        should_validate = (
            epoch == 1
            or epoch == int(epochs)
            or epoch % max(1, int(validation_every)) == 0
            or (int(print_every) > 0 and epoch % int(print_every) == 0)
        )
        if should_validate:
            validation_metrics_device = trainer.case_eval_steps[mode](
                parameters,
                *validation_device,
                validation_indices,
                validation_masks,
            )
            train_metrics, validation_metrics = jax.device_get(
                (train_metrics_device, validation_metrics_device)
            )
            train_metrics = {
                name: float(value)
                for name, value in zip(CaseMetrics._fields, train_metrics)
            }
            validation_metrics = {
                name: float(value)
                for name, value in zip(CaseMetrics._fields, validation_metrics)
            }
            last_validation = validation_metrics
        else:
            train_metrics = _metrics_to_dict(train_metrics_device)
            if last_validation is None:
                raise RuntimeError("validation metrics were not initialized")
            validation_metrics = last_validation

        history.append(
            {
                "epoch": epoch,
                "phase": mode,
                "validation_evaluated": bool(should_validate),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                # Compatibility keys used by existing plotting/statistics code.
                "train_violation_loss": train_metrics.get("physics_loss", float("nan")),
                "validation_violation_loss": validation_metrics.get("physics_loss", float("nan")),
                "train_violation": (
                    train_metrics.get("differential_violation", 0.0)
                    + train_metrics.get("algebraic_violation", 0.0)
                    + train_metrics.get("inequality_violation", 0.0)
                ),
                "validation_violation": (
                    validation_metrics.get("differential_violation", 0.0)
                    + validation_metrics.get("algebraic_violation", 0.0)
                    + validation_metrics.get("inequality_violation", 0.0)
                ),
            }
        )
        logger.epoch(epoch, train_metrics, validation_metrics)
        if checkpoint_epoch is not None and epoch == int(checkpoint_epoch):
            checkpoint_parameters = parameters

    first_leaf = jax.tree_util.tree_leaves(parameters)[0]
    jax.block_until_ready(first_leaf)
    wall_seconds = time.perf_counter() - start
    logger.finish(wall_seconds)
    return (
        DAETrainingResult(
            parameters=parameters,
            history=tuple(history),
            projection_start_epoch=1 if mode == "projected" else None,
            switch_reason=None,
            wall_seconds=float(wall_seconds),
        ),
        checkpoint_parameters,
    )


def _make_config(
    *,
    epochs: int,
    learning_rate: float,
    hidden_sizes: tuple[int, ...],
    batch_size: int,
    lambda_data: float,
    lambda_derivative: float,
    lambda_physics: float,
    derivative_steps: tuple[float, ...],
    print_every: int,
    validation_every: int,
    seed: int,
) -> CaseStudyTrainingConfig:
    return CaseStudyTrainingConfig(
        epochs=int(epochs),
        pretrain_epochs=0,
        eta=None,
        learning_rate=float(learning_rate),
        hidden_sizes=tuple(hidden_sizes),
        batch_size=int(batch_size),
        data_weight=float(lambda_data),
        physics_weight=float(lambda_physics),
        derivative_weight=float(lambda_derivative),
        derivative_consistency_weight=0.0,
        violation_weight=float(lambda_physics),
        consistency_weight=0.0,
        initial_weight=0.0,
        boundary_weight=0.0,
        full_model_derivative_steps=tuple(float(v) for v in derivative_steps),
        print_every=int(print_every),
        validation_every=int(validation_every),
        seed=int(seed),
        shuffle=True,
    )


def _trainer(
    trainer_cls,
    trainer_kwargs: Mapping[str, Any],
    *,
    epochs: int,
    learning_rate: float,
    hidden_sizes: tuple[int, ...],
    batch_size: int,
    lambda_data: float,
    lambda_derivative: float,
    lambda_physics: float,
    derivative_steps: tuple[float, ...],
    print_every: int,
    validation_every: int,
    seed: int,
):
    return trainer_cls(
        config=_make_config(
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_sizes=hidden_sizes,
            batch_size=batch_size,
            lambda_data=lambda_data,
            lambda_derivative=lambda_derivative,
            lambda_physics=lambda_physics,
            derivative_steps=derivative_steps,
            print_every=print_every,
            validation_every=validation_every,
            seed=seed,
        ),
        **trainer_kwargs,
    )


def same_horizon_indices(
    coordinate_values: np.ndarray,
    valid_indices: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    split_mode: str,
    seed: int,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create disjoint reproducible splits within one common coordinate horizon."""
    values = np.asarray(coordinate_values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_indices, dtype=np.int64)
    if valid.size < 3:
        raise ValueError("at least three valid samples are required")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if float(train_fraction) + float(validation_fraction) >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if split_mode not in {"random", "chronological"}:
        raise ValueError("split_mode must be random or chronological")

    ordered = valid[np.argsort(values[valid], kind="stable")]
    if split_mode == "random":
        ordered = np.random.default_rng(int(seed)).permutation(ordered)
    n = len(ordered)
    n_train = max(1, int(math.floor(float(train_fraction) * n)))
    n_validation = max(1, int(math.floor(float(validation_fraction) * n)))
    if n_train + n_validation >= n:
        n_validation = max(1, n - n_train - 1)
    train = ordered[:n_train]
    validation = ordered[n_train : n_train + n_validation]
    test = ordered[n_train + n_validation :]

    def limited(index: np.ndarray, count: int | None) -> np.ndarray:
        index = np.asarray(index, dtype=np.int64)
        if count is None or int(count) <= 0 or len(index) <= int(count):
            return index
        if split_mode == "chronological":
            positions = np.linspace(0, len(index) - 1, int(count)).round().astype(int)
            return index[positions]
        return index[: int(count)]

    train = limited(train, train_limit)
    validation = limited(validation, validation_limit)
    test = limited(test, test_limit)
    if min(len(train), len(validation), len(test)) < 1:
        raise ValueError("one or more dataset splits are empty")
    horizon_start = float(np.min(values[valid]))
    horizon_end = float(np.max(values[valid]))
    metadata = {
        "protocol": "same_horizon",
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        # Compatibility aliases.  There is no longer a distinct t3 horizon.
        "t1": horizon_start,
        "t2": horizon_end,
        "t3": horizon_end,
        "split_mode": split_mode,
    }
    return train, validation, test, metadata


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _relative_mse(residual: np.ndarray, reference: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if residual.size == 0:
        return 0.0
    numerator = float(np.mean(residual**2))
    denominator = float(np.mean(reference**2)) if reference.size else 0.0
    return numerator / max(denominator, 1.0e-30)


def _relative_state_error(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-30))


def _fixed_prediction_batches(coordinates: np.ndarray, parameters: np.ndarray, batch_size: int):
    n = len(coordinates)
    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        x = np.zeros((batch_size, coordinates.shape[1]), dtype=np.float64)
        p = np.zeros((batch_size, parameters.shape[1]), dtype=np.float64)
        x[:count] = coordinates[start : start + count]
        p[:count] = parameters[start : start + count]
        if count < batch_size:
            x[count:] = x[count - 1]
            p[count:] = p[count - 1]
        yield x, p, count


def _final_raw_pass(trainer, parameters, coordinates, physical_parameters):
    @jax.jit
    def batch(parameters, coordinates, physical_parameters):
        states = trainer.backbone.apply(parameters, coordinates)
        derivatives = extract_network_derivatives(
            trainer.backbone.apply,
            parameters,
            coordinates,
            trainer.compiled.discretization.derivatives,
            trainer.compiled.system.differential_states,
        )
        blocks = trainer._physical_blocks(coordinates, states, derivatives, physical_parameters)
        return states, derivatives, blocks

    state_rows, derivative_rows = [], []
    block_rows = [[], [], []]
    for x, p, count in _fixed_prediction_batches(
        np.asarray(coordinates, dtype=np.float64),
        np.asarray(physical_parameters, dtype=np.float64),
        trainer.config.batch_size,
    ):
        states, derivatives, blocks = batch(parameters, jnp.asarray(x), jnp.asarray(p))
        states, derivatives, blocks = jax.device_get((states, derivatives, blocks))
        state_rows.append(np.asarray(states)[:count])
        derivative_rows.append(np.asarray(derivatives)[:count])
        for index in range(3):
            block_rows[index].append(np.asarray(blocks[index])[:count])
    return (
        np.concatenate(state_rows),
        np.concatenate(derivative_rows),
        tuple(np.concatenate(rows) for rows in block_rows),
        None,
    )


def _final_projected_pass(trainer, parameters, coordinates, physical_parameters):
    @jax.jit
    def batch(parameters, coordinates, physical_parameters):
        decoded, info, _context, _target = trainer.model.forward(
            parameters, coordinates, physical_parameters
        )
        states = decoded["states"]
        model_derivatives = trainer.full_model_derivative_plan.evaluate(
            trainer.full_model_derivative_model,
            parameters,
            coordinates,
            physical_parameters,
            center_states=states,
        )
        blocks = trainer._physical_blocks(
            coordinates, states, model_derivatives, physical_parameters
        )
        return (
            states,
            model_derivatives,
            decoded["derivatives"],
            blocks,
            info.forward_converged,
            info.gradient_eligible,
            info.forward_iterations,
            info.forward_residual_inf_norm,
        )

    state_rows, derivative_rows, projected_derivative_rows = [], [], []
    block_rows = [[], [], []]
    info_rows = {"converged": [], "eligible": [], "iterations": [], "residual": []}
    for x, p, count in _fixed_prediction_batches(
        np.asarray(coordinates, dtype=np.float64),
        np.asarray(physical_parameters, dtype=np.float64),
        trainer.config.batch_size,
    ):
        result = jax.device_get(batch(parameters, jnp.asarray(x), jnp.asarray(p)))
        states, derivatives, projected_derivatives, blocks, converged, eligible, iterations, residual = result
        state_rows.append(np.asarray(states)[:count])
        derivative_rows.append(np.asarray(derivatives)[:count])
        projected_derivative_rows.append(np.asarray(projected_derivatives)[:count])
        for index in range(3):
            block_rows[index].append(np.asarray(blocks[index])[:count])
        for key, values in (
            ("converged", converged),
            ("eligible", eligible),
            ("iterations", iterations),
            ("residual", residual),
        ):
            info_rows[key].append(np.asarray(values)[:count])
    return (
        np.concatenate(state_rows),
        np.concatenate(derivative_rows),
        tuple(np.concatenate(rows) for rows in block_rows),
        {
            "projected_derivatives": np.concatenate(projected_derivative_rows),
            **{key: np.concatenate(rows) for key, rows in info_rows.items()},
        },
    )


def inference_seconds_per_batch(
    trainer,
    parameters,
    coordinates: np.ndarray,
    physical_parameters: np.ndarray,
    *,
    projected: bool,
    warmup: int = 2,
    repeats: int = 10,
) -> float:
    batch_size = int(trainer.config.batch_size)
    x = np.asarray(coordinates[:batch_size], dtype=np.float64)
    p = np.asarray(physical_parameters[:batch_size], dtype=np.float64)
    if len(x) < batch_size:
        pad = batch_size - len(x)
        x = np.concatenate((x, np.repeat(x[-1:], pad, axis=0)), axis=0)
        p = np.concatenate((p, np.repeat(p[-1:], pad, axis=0)), axis=0)
    xj, pj = jnp.asarray(x), jnp.asarray(p)
    if projected:
        @jax.jit
        def inference(x_value, p_value):
            decoded, _info, _context, _target = trainer.model.forward(
                parameters, x_value, p_value
            )
            return decoded["states"]
    else:
        @jax.jit
        def inference(x_value, _p_value):
            return trainer.backbone.apply(parameters, x_value)
    for _ in range(max(1, warmup)):
        jax.block_until_ready(inference(xj, pj))
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        jax.block_until_ready(inference(xj, pj))
    return float((time.perf_counter() - start) / max(1, repeats))


def _write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(TABLE_COLUMNS) + " |\n")
        handle.write("|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|\n")
        for row in rows:
            rendered = []
            for key in TABLE_COLUMNS:
                value = row.get(key)
                if isinstance(value, float):
                    rendered.append("NaN" if not math.isfinite(value) else f"{value:.6e}")
                else:
                    rendered.append(str(value))
            handle.write("| " + " | ".join(rendered) + " |\n")


def write_comparison_tables(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    print(json.dumps([_json_ready(row) for row in rows], indent=2))


def _save_figure_png_pdf(figure, output_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt
    plt.show()


def _positive_or_nan(values: np.ndarray, minimum: float = 1.0e-12) -> np.ndarray:
    """Clip positive finite values for log axes while preserving unavailable data."""

    array = np.asarray(values, dtype=np.float64)
    return np.where(
        np.isfinite(array) & (array > 0.0),
        np.maximum(array, minimum),
        np.nan,
    )


def save_dae_hardnet_style_loss_plots(
    output_dir: Path,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Save the DAE-HardNet manuscript-style 1x2 training-metrics figure.

    The visual configuration follows the upstream DAE-HardNet comparison plot:
    MLP/PINN/DAE-HardNet are green/blue/red, training curves are solid,
    validation curves are dashed, both panels use logarithmic y axes, and a
    shared legend is placed below the two panels.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
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

    color_map = {"mlp": "g", "pinn": "b", "daehn": "r"}
    series: dict[str, dict[str, np.ndarray]] = {}
    for name, history in histories.items():
        rows = list(history)
        train_data = _positive_or_nan(
            np.asarray([row["train_data_loss"] for row in rows])
        )
        validation_data = _positive_or_nan(
            np.asarray([row["validation_data_loss"] for row in rows])
        )
        train_physics = _positive_or_nan(
            np.asarray([row.get("train_physics_loss", np.nan) for row in rows])
        )
        validation_physics = _positive_or_nan(
            np.asarray([row.get("validation_physics_loss", np.nan) for row in rows])
        )
        train_violation = _positive_or_nan(
            np.asarray([row.get("train_violation", np.nan) for row in rows])
        )
        validation_violation = _positive_or_nan(
            np.asarray([row.get("validation_violation", np.nan) for row in rows])
        )
        series[name] = {
            "train_data": train_data,
            "validation_data": validation_data,
            "train_physics": train_physics,
            "validation_physics": validation_physics,
            "train_violation": train_violation,
            "validation_violation": validation_violation,
        }

    figure, (axis_data, axis_violation) = plt.subplots(
        1, 2, figsize=(12, 4), dpi=600
    )
    for name in MODEL_ORDER:
        if name not in series:
            continue
        values = series[name]
        epochs = np.arange(len(values["train_data"]))
        color = color_map[name]
        label = MODEL_DISPLAY[name]
        train_rmse = np.sqrt(values["train_data"])
        validation_rmse = np.sqrt(values["validation_data"])
        axis_data.plot(
            epochs,
            train_rmse,
            label=f"{label} (train)",
            linewidth=0.5,
            color=color,
        )
        axis_data.plot(
            epochs,
            validation_rmse,
            label=f"{label} (val)",
            linestyle="--",
            linewidth=0.5,
            color=color,
        )
        for values_rmse, linestyle in (
            (train_rmse, "-"),
            (validation_rmse, "--"),
        ):
            finite = values_rmse[np.isfinite(values_rmse)]
            if finite.size:
                axis_data.axhline(
                    y=float(np.min(finite)),
                    linestyle=linestyle,
                    linewidth=0.5,
                    color=color,
                )

        train_violation = values["train_violation"]
        validation_violation = values["validation_violation"]
        if np.isfinite(train_violation).any():
            axis_violation.plot(
                epochs,
                train_violation,
                label=f"{label} (train)",
                linewidth=0.5,
                color=color,
            )
            finite = train_violation[np.isfinite(train_violation)]
            axis_violation.axhline(
                y=float(np.min(finite)),
                linestyle="-",
                linewidth=0.5,
                color=color,
            )
        if np.isfinite(validation_violation).any():
            axis_violation.plot(
                epochs,
                validation_violation,
                label=f"{label} (val)",
                linestyle="--",
                linewidth=0.5,
                color=color,
            )
            finite = validation_violation[np.isfinite(validation_violation)]
            axis_violation.axhline(
                y=float(np.min(finite)),
                linestyle="--",
                linewidth=0.5,
                color=color,
            )

    axis_data.set(xlabel="Epoch", ylabel="RMSE")
    axis_data.set_yscale("log")
    axis_violation.set(xlabel="Epoch", ylabel="Absolute Violation")
    axis_violation.set_yscale("log")

    handles, labels = axis_data.get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels) // 2),
            fontsize=14,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    _save_figure_png_pdf(
        figure, output_dir, "all_models_rmse_and_violation"
    )
    plt.close(figure)


def save_state_comparison_plot(
    output_dir: Path,
    coordinates: np.ndarray,
    exact: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    state_names: Sequence[str],
) -> None:
    """Save compact state-data versus prediction panels in PNG and PDF.

    Layout is selected from the number of plotted entities: one row for up to
    four states and two rows for five to seven states.  Thus the seven-state
    CSTR plot is 2x4 with the unused eighth axis hidden, while the three-state
    catalysis plot is 1x3.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    coordinate = np.asarray(coordinates, dtype=np.float64)[:, 0]
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    exact = np.asarray(exact, dtype=np.float64)[order]
    ordered_predictions = {
        name: np.asarray(values, dtype=np.float64)[order]
        for name, values in predictions.items()
    }
    if exact.ndim != 2 or exact.shape[1] != len(state_names):
        raise ValueError("state_names must match the plotted state dimension")

    n_states = len(state_names)
    n_columns = min(4, n_states)
    n_rows = int(math.ceil(n_states / n_columns))
    color_map = {"mlp": "g", "pinn": "b", "daehn": "r"}
    plt.rcParams.update({
        "font.size": 20,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7.0 * n_columns, 6.0 * n_rows),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for index, state_name in enumerate(state_names):
        axis = flat_axes[index]
        axis.plot(
            coordinate,
            exact[:, index],
            color="k",
            linewidth=1.2,
            label="Ground Truth",
        )
        for name in MODEL_ORDER:
            if name not in ordered_predictions:
                continue
            axis.plot(
                coordinate,
                ordered_predictions[name][:, index],
                color=color_map[name],
                linewidth=1.0,
                label=MODEL_DISPLAY[name],
            )
        axis.set_xlabel("t", fontsize=24)
        axis.set_ylabel(str(state_name), fontsize=24)
        axis.tick_params(axis="both", which="major", labelsize=18)

    for axis in flat_axes[n_states:]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            fontsize=24,
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    figure.tight_layout(rect=[0, 0.10, 1, 1])
    _save_figure_png_pdf(
        figure, output_dir, "all_models_prediction_comparison"
    )
    plt.close(figure)

def _combine_pretrain_projected(
    pretraining: DAETrainingResult,
    projected: DAETrainingResult,
    pretrain_epochs: int,
) -> DAETrainingResult:
    history = []
    for row in pretraining.history[:pretrain_epochs]:
        item = dict(row)
        item["phase"] = "pretrain"
        history.append(item)
    for row in projected.history:
        item = dict(row)
        item["epoch"] = int(item["epoch"]) + int(pretrain_epochs)
        item["phase"] = "projected"
        history.append(item)
    return DAETrainingResult(
        parameters=projected.parameters,
        history=tuple(history),
        projection_start_epoch=int(pretrain_epochs) + 1,
        switch_reason="data-only MLP pretraining completed; optimizer reset",
        wall_seconds=float(pretraining.wall_seconds + projected.wall_seconds),
    )



def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_ready(dict(payload)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _backbone_manifest(backbone: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"class": type(backbone).__name__}
    if is_dataclass(backbone):
        values: dict[str, Any] = {}
        for item in fields(backbone):
            value = getattr(backbone, item.name)
            if item.name == "activation":
                values[item.name] = getattr(value, "__name__", "tanh")
            else:
                values[item.name] = _json_ready(value)
        result["kwargs"] = values
    else:
        result["kwargs"] = {}
    return result


def save_parameter_checkpoint(directory: Path, parameters: PyTree) -> dict[str, Any]:
    """Save an exact JAX parameter PyTree without a duplicate weight archive."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    host_tree = jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), parameters)
    checkpoint_path = directory / "parameter_tree.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(host_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
    leaves = jax.tree_util.tree_leaves(host_tree)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "checkpoint_file": checkpoint_path.name,
        "parameter_leaf_count": len(leaves),
        "checkpoint_sha256": digest,
    }


def load_parameter_checkpoint(directory: Path) -> PyTree:
    path = Path(directory) / "parameter_tree.pkl"
    if not path.exists():
        raise FileNotFoundError(f"saved parameter tree not found: {path}")
    with path.open("rb") as handle:
        tree = pickle.load(handle)
    return jax.tree_util.tree_map(jnp.asarray, tree)


def make_inference_trainer(
    trainer_cls,
    trainer_kwargs: Mapping[str, Any],
    *,
    hidden_sizes: tuple[int, ...],
    batch_size: int,
    derivative_steps: tuple[float, ...],
    seed: int,
    lambda_data: float = 1.0,
    lambda_derivative: float = 1.0,
    lambda_physics: float = 1.0,
):
    """Construct a trainer for loaded-model inference without running training."""
    return _trainer(
        trainer_cls,
        trainer_kwargs,
        epochs=1,
        learning_rate=1.0e-3,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        lambda_data=lambda_data,
        lambda_derivative=lambda_derivative,
        lambda_physics=lambda_physics,
        derivative_steps=derivative_steps,
        print_every=1,
        validation_every=1,
        seed=seed,
    )


def evaluate_trainer_predictions(
    trainer,
    parameters: PyTree,
    *,
    coordinates: np.ndarray,
    physical_parameters: np.ndarray,
    targets: np.ndarray,
    derivative_targets: np.ndarray,
    projected: bool,
    model_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one cached evaluation pass and calculate the standard comparison metrics."""
    if projected:
        prediction, model_derivatives, blocks, info = _final_projected_pass(
            trainer, parameters, coordinates, physical_parameters
        )
    else:
        prediction, model_derivatives, blocks, info = _final_raw_pass(
            trainer, parameters, coordinates, physical_parameters
        )
    differential, algebraic, inequalities = blocks
    targets = np.asarray(targets, dtype=np.float64)
    derivative_targets = np.asarray(derivative_targets, dtype=np.float64)
    data_mse = float(np.mean((prediction - targets) ** 2))
    derivative_mse = (
        float(np.mean((model_derivatives - derivative_targets) ** 2))
        if str(model_name) == "daehn"
        else float("nan")
    )
    physics_residual = np.concatenate(
        (
            np.asarray(differential).reshape((len(prediction), -1)),
            np.asarray(inequalities).reshape((len(prediction), -1)),
        ),
        axis=1,
    )
    metrics: dict[str, Any] = {
        "mse_data": data_mse,
        "mse_derivative": derivative_mse,
        "physics_loss_relative": _relative_mse(physics_residual, derivative_targets),
        "algebraic_violation_relative": _relative_mse(np.asarray(algebraic), targets),
        "relative_state_error": _relative_state_error(prediction, targets),
        "inference_time_seconds_per_batch": inference_seconds_per_batch(
            trainer,
            parameters,
            coordinates,
            physical_parameters,
            projected=projected,
        ),
        "differential_violation": float(np.mean(np.linalg.norm(differential, axis=-1)))
        if differential.shape[-1]
        else 0.0,
        "algebraic_violation": float(np.mean(np.linalg.norm(algebraic, axis=-1)))
        if algebraic.shape[-1]
        else 0.0,
        "inequality_violation": float(np.mean(np.linalg.norm(inequalities, axis=-1)))
        if inequalities.shape[-1]
        else 0.0,
        "evaluation_samples": int(len(prediction)),
    }
    payload: dict[str, np.ndarray] = {
        "coordinates": np.asarray(coordinates),
        "physical_parameters": np.asarray(physical_parameters),
        "exact": targets,
        "exact_derivatives": derivative_targets,
        "prediction": np.asarray(prediction),
        "model_derivatives": np.asarray(model_derivatives),
        "differential_residual": np.asarray(differential),
        "algebraic_residual": np.asarray(algebraic),
        "inequality_residual": np.asarray(inequalities),
    }
    if info is not None:
        payload.update(
            {
                "projected_derivatives": np.asarray(info["projected_derivatives"]),
                "converged": np.asarray(info["converged"]),
                "eligible": np.asarray(info["eligible"]),
                "iterations": np.asarray(info["iterations"]),
                "solver_residual": np.asarray(info["residual"]),
            }
        )
        metrics.update(
            {
                "projection_converged_fraction": float(np.mean(info["converged"])),
                "gradient_eligible_fraction": float(np.mean(info["eligible"])),
                "mean_projection_iterations": float(np.mean(info["iterations"])),
                "max_projection_iterations": int(np.max(info["iterations"])),
                "max_projection_solver_residual": float(np.max(info["residual"])),
            }
        )
    return payload, metrics


def run_model_suite(
    *,
    example: int,
    system_name: str,
    model: str,
    trainer_cls,
    trainer_kwargs: Mapping[str, Any],
    train_coordinates: np.ndarray,
    train_parameters: np.ndarray,
    train_targets: np.ndarray,
    train_derivative_targets: np.ndarray,
    validation_coordinates: np.ndarray,
    validation_parameters: np.ndarray,
    validation_targets: np.ndarray,
    validation_derivative_targets: np.ndarray,
    evaluation_coordinates: np.ndarray,
    evaluation_parameters: np.ndarray,
    evaluation_targets: np.ndarray,
    evaluation_derivative_targets: np.ndarray,
    output_dir: Path,
    hidden_sizes: tuple[int, ...],
    batch_size: int,
    epochs: int,
    pretrain_epochs: int,
    learning_rate: float,
    projected_learning_rate: float,
    lambda_data: float,
    lambda_derivative: float,
    lambda_physics: float,
    derivative_steps: tuple[float, ...],
    seed: int,
    print_every: int,
    validation_every: int,
    state_names: Sequence[str],
    split_info: Mapping[str, Any],
    run_config: Mapping[str, Any] | None = None,
    plot: bool = False,
    plot_state_names: Sequence[str] | None = None,
    plot_state_transform: Any = None,
    case_script: str | None = None,
    trajectory_coordinates: np.ndarray | None = None,
    trajectory_parameters: np.ndarray | None = None,
    trajectory_targets: np.ndarray | None = None,
    trajectory_derivative_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train and save models only; prediction, plotting and tables are offline tasks."""
    set_reproducible_seed(seed)
    from project_paths import result_path
    output_dir = result_path(output_dir, Path(__file__).resolve().parents[1] / "train.py")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = MODEL_ORDER if model == "all" else (str(model).lower(),)
    if any(name not in MODEL_ORDER for name in selected):
        raise ValueError("model must be one of mlp, pinn, daehn, all")
    if "daehn" in selected and not 0 < int(pretrain_epochs) < int(epochs):
        raise ValueError("DAEHN requires 0 < pretrain_epochs < epochs")
    if plot:
        print(
            "Note: --plot is retained for CLI compatibility but training no longer "
            "creates figures. Use postprocessing/plot_saved_runs.py after prediction."
        )

    train_data = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (train_coordinates, train_parameters, train_targets, train_derivative_targets)
    )
    validation_data = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            validation_coordinates,
            validation_parameters,
            validation_targets,
            validation_derivative_targets,
        )
    )
    evaluation_arrays = {
        "coordinates": np.asarray(evaluation_coordinates, dtype=np.float64),
        "physical_parameters": np.asarray(evaluation_parameters, dtype=np.float64),
        "targets": np.asarray(evaluation_targets, dtype=np.float64),
        "derivative_targets": np.asarray(evaluation_derivative_targets, dtype=np.float64),
        "initial_coordinates": np.asarray(trainer_kwargs.get("initial_coordinates"), dtype=np.float64),
        "initial_targets": np.asarray(trainer_kwargs.get("initial_targets"), dtype=np.float64),
    }
    evaluation_sha = _array_hash(evaluation_arrays)
    trajectory_arrays = {
        "coordinates": np.asarray(trajectory_coordinates if trajectory_coordinates is not None else evaluation_coordinates, dtype=np.float64),
        "physical_parameters": np.asarray(trajectory_parameters if trajectory_parameters is not None else evaluation_parameters, dtype=np.float64),
        "targets": np.asarray(trajectory_targets if trajectory_targets is not None else evaluation_targets, dtype=np.float64),
        "derivative_targets": np.asarray(trajectory_derivative_targets if trajectory_derivative_targets is not None else evaluation_derivative_targets, dtype=np.float64),
    }
    trajectory_sha = _array_hash(trajectory_arrays)

    trained: dict[str, tuple[Any, DAETrainingResult, bool]] = {}
    mlp_checkpoint = None
    mlp_pretraining_result = None
    for model_name in selected:
        set_reproducible_seed(seed)
        print(f"\n=== Training {MODEL_DISPLAY[model_name]} ===")
        if model_name == "mlp":
            trainer = _trainer(
                trainer_cls,
                trainer_kwargs,
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_sizes=hidden_sizes,
                batch_size=batch_size,
                lambda_data=lambda_data,
                lambda_derivative=0.0,
                lambda_physics=0.0,
                derivative_steps=derivative_steps,
                print_every=print_every,
                validation_every=validation_every,
                seed=seed,
            )
            training, checkpoint = _fit_mode(
                trainer,
                mode="data",
                label="MLP data-only training",
                epochs=epochs,
                train_data=train_data,
                validation_data=validation_data,
                initial_parameters=None,
                seed=seed,
                print_every=print_every,
                validation_every=validation_every,
                checkpoint_epoch=pretrain_epochs if model == "all" else None,
            )
            if model == "all":
                mlp_checkpoint = checkpoint
                mlp_pretraining_result = DAETrainingResult(
                    parameters=checkpoint,
                    history=tuple(training.history[:pretrain_epochs]),
                    projection_start_epoch=None,
                    switch_reason=None,
                    wall_seconds=float(training.wall_seconds * pretrain_epochs / epochs),
                )
            trained[model_name] = (trainer, training, False)
        elif model_name == "pinn":
            trainer = _trainer(
                trainer_cls,
                trainer_kwargs,
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_sizes=hidden_sizes,
                batch_size=batch_size,
                lambda_data=lambda_data,
                lambda_derivative=0.0,
                lambda_physics=lambda_physics,
                derivative_steps=derivative_steps,
                print_every=print_every,
                validation_every=validation_every,
                seed=seed,
            )
            training, _ = _fit_mode(
                trainer,
                mode="pinn",
                label="PINN data + physics training",
                epochs=epochs,
                train_data=train_data,
                validation_data=validation_data,
                initial_parameters=None,
                seed=seed,
                print_every=print_every,
                validation_every=validation_every,
            )
            trained[model_name] = (trainer, training, False)
        else:
            if mlp_checkpoint is None:
                pretrainer = _trainer(
                    trainer_cls,
                    trainer_kwargs,
                    epochs=pretrain_epochs,
                    learning_rate=learning_rate,
                    hidden_sizes=hidden_sizes,
                    batch_size=batch_size,
                    lambda_data=lambda_data,
                    lambda_derivative=0.0,
                    lambda_physics=0.0,
                    derivative_steps=derivative_steps,
                    print_every=print_every,
                    validation_every=validation_every,
                    seed=seed,
                )
                pretraining, _ = _fit_mode(
                    pretrainer,
                    mode="data",
                    label="DAE-HardNet data-only MLP pretraining",
                    epochs=pretrain_epochs,
                    train_data=train_data,
                    validation_data=validation_data,
                    initial_parameters=None,
                    seed=seed,
                    print_every=print_every,
                    validation_every=validation_every,
                )
                initial_parameters = pretraining.parameters
            else:
                print(
                    f"Reusing the data-only MLP checkpoint at epoch {pretrain_epochs} "
                    "for DAE-HardNet pretraining."
                )
                pretraining = mlp_pretraining_result
                initial_parameters = mlp_checkpoint
            projected_epochs = int(epochs) - int(pretrain_epochs)
            trainer = _trainer(
                trainer_cls,
                trainer_kwargs,
                epochs=projected_epochs,
                learning_rate=projected_learning_rate,
                hidden_sizes=hidden_sizes,
                batch_size=batch_size,
                lambda_data=lambda_data,
                lambda_derivative=lambda_derivative,
                lambda_physics=lambda_physics,
                derivative_steps=derivative_steps,
                print_every=print_every,
                validation_every=validation_every,
                seed=seed,
            )
            projected_result, _ = _fit_mode(
                trainer,
                mode="projected",
                label="DAE-HardNet projected training",
                epochs=projected_epochs,
                train_data=train_data,
                validation_data=validation_data,
                initial_parameters=initial_parameters,
                seed=seed,
                print_every=print_every,
                validation_every=validation_every,
            )
            training = _combine_pretrain_projected(
                pretraining, projected_result, int(pretrain_epochs)
            )
            trained[model_name] = (trainer, training, True)

    saved_runs: list[dict[str, Any]] = []
    for model_name in selected:
        trainer, training, projected = trained[model_name]
        model_dir = model_path(output_dir) / model_name / f"seed={seed}"
        output_run = output_path(model_dir)
        output_run.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_info = save_parameter_checkpoint(model_dir, training.parameters)
        np.savez(model_dir / "evaluation_data.npz", **evaluation_arrays)
        np.savez(model_dir / "trajectory_data.npz", **trajectory_arrays)
        history_path = output_run / "training_history.json"
        history_path.write_text(
            json.dumps([_json_ready(value) for value in training.history], indent=2) + "\n",
            encoding="utf-8",
        )
        effective_config = dict(run_config or {})
        effective_config["model"] = model_name
        effective_config["seed"] = int(seed)
        effective_config["output_dir"] = str(model_dir)
        effective_config["lambda_data"] = float(lambda_data)
        effective_config["lambda_derivative"] = float(lambda_derivative if projected else 0.0)
        effective_config["lambda_physics"] = float(
            lambda_physics if model_name != "mlp" else 0.0
        )
        effective_config["projection"] = "fb" if projected else "none"
        (model_dir / "run_config.json").write_text(
            json.dumps(_json_ready(effective_config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        backbone_info = _backbone_manifest(trainer.backbone)
        manifest = {
            "schema_version": 2,
            "status": "trained",
            "example": int(example),
            "system": str(system_name),
            "model": model_name,
            "model_display": MODEL_DISPLAY[model_name],
            "seed": int(seed),
            "case_script": case_script,
            "trainer_class": type(trainer).__name__,
            "backbone": backbone_info,
            "projected": bool(projected),
            "projection": "fb" if projected else "none",
            "hidden_sizes": list(hidden_sizes),
            "batch_size": int(batch_size),
            "derivative_steps": list(derivative_steps),
            "state_names": list(state_names),
            "plot_state_names": list(plot_state_names or state_names),
            "plot_transform": "catalysis_physical" if plot_state_transform is not None else "identity",
            "plot_layout": "centered_4_3" if int(example) == 8 else "auto",
            "history_file": history_path.name,
            "evaluation_data_file": "evaluation_data.npz",
            "evaluation_data_sha256": evaluation_sha,
            "trajectory_data_file": "trajectory_data.npz",
            "trajectory_data_sha256": trajectory_sha,
            "run_config_file": "run_config.json",
            "split_info": dict(split_info),
            "training_wall_seconds": float(training.wall_seconds),
            "epochs": int(epochs),
            "pretrain_epochs": int(pretrain_epochs if projected else 0),
            **checkpoint_info,
        }
        manifest["run_signature"] = _json_hash(
            {
                "schema_version": manifest["schema_version"],
                "example": manifest["example"],
                "system": manifest["system"],
                "model": manifest["model"],
                "backbone": manifest["backbone"],
                "projection": manifest["projection"],
                "evaluation_data_sha256": evaluation_sha,
                "run_config": effective_config,
            }
        )
        (model_dir / "checkpoint_manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "example": int(example),
            "system": str(system_name),
            "model": model_name,
            "seed": int(seed),
            "status": "trained",
            "training_wall_seconds": float(training.wall_seconds),
            "final_history": _json_ready(training.history[-1]) if training.history else None,
            "checkpoint_manifest": str(model_dir / "checkpoint_manifest.json"),
        }
        (output_run / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        saved_runs.append(summary)

    combined = {
        "schema_version": 2,
        "example": int(example),
        "system": str(system_name),
        "seed": int(seed),
        "models": list(selected),
        "status": "trained",
        "split_info": dict(split_info),
        "runs": saved_runs,
        "postprocessing": {
            "predict": "postprocessing/predict_saved_runs.py",
            "plot": "postprocessing/plot_saved_runs.py",
            "compare": "postprocessing/compare_saved_runs.py",
        },
    }
    print(json.dumps(_json_ready(combined), indent=2, sort_keys=True))
    return combined
