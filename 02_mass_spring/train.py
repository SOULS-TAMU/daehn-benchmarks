#!/usr/bin/env python
r"""Generate mass--spring data and train DAE-HardNet.

Physical system
---------------
    dx/dt = v
    dv/dt = -x

Conserved energy
----------------
    E(x, v) = 0.5 * (x**2 + v**2) = E0.

The projected phase hard-projects only the energy invariant.  The two original
ODE residuals are evaluated with derivatives of the complete
NN -> projection -> reconstruction mapping.  Training uses two optimizer
stages: raw pretraining followed by projected DAE-HardNet training with a fresh,
smaller-step Adam optimizer.

Example (PowerShell)
--------------------
    python .\examples\12_mass_spring_daehn.py `
        --data-dir data\example_12_mass_spring `
        --output-dir results\example_12_mass_spring_daehn `
        --plot

Quick validation
----------------
    python .\examples\12_mass_spring_daehn.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Direct execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from project_paths import find_project_root, result_path

_REPOSITORY_ROOT = find_project_root(Path(__file__).parent)
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from daehn.dae import (  # noqa: E402
    Coordinate,
    DAEData,
    DAETrainer,
    DAETrainingConfig,
    DAETrainingResult,
    DAESystem,
    DerivativeSpec,
    DifferentialState,
    EquationBlock,
    Parameter,
    ScalingConfig,
    TaylorNeighborhood,
    compile_dae,
)
from daehn.opt import ImplicitGradientConfig, ProjectionConfig  # noqa: E402
from daehn.train.projected_neural import MLPBackbone  # noqa: E402
from case_study_utils import (  # noqa: E402
    WeightedLossTrainerMixin,
    same_horizon_indices,
    run_model_suite,
    set_reproducible_seed,
)

jax.config.update("jax_enable_x64", True)

STATE_NAMES = ("position_x", "velocity_v")
DATA_DERIVATIVE_NAMES = ("dposition_dt", "dvelocity_dt")


def _empty(_coordinates, _states, _derivatives, _parameters):
    return jnp.empty((0,), dtype=jnp.float64)


@dataclass(frozen=True)
class GeneratedDataset:
    data: DAEData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    energy0: float
    amplitude: float
    horizon: float
    time_step: float


@dataclass(frozen=True)
class TimeScaledMLPBackbone:
    """MLP evaluated on normalized time with physical output scaling."""

    hidden_sizes: tuple[int, ...]
    time_scale: float
    state_scale: float
    activation: Any = jax.nn.tanh

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_scale) or self.time_scale <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        if not np.isfinite(self.state_scale) or self.state_scale <= 0.0:
            raise ValueError("state_scale must be finite and positive")

    def init(self, key, input_dim, output_dim, dtype=jnp.float64):
        return MLPBackbone(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).init(key, input_dim, output_dim, dtype)

    def apply(self, parameters, inputs):
        values = jnp.asarray(inputs)
        scaled_time = values / jnp.asarray(self.time_scale, dtype=values.dtype)
        raw = MLPBackbone(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).apply(parameters, scaled_time)
        # The exact orbit lies in [-amplitude, amplitude].  A modestly larger
        # scale keeps the network numerically balanced without hard saturation.
        return jnp.asarray(self.state_scale, dtype=raw.dtype) * raw


class MassSpringTrainer(WeightedLossTrainerMixin, DAETrainer):
    """Project energy only, but train/report the complete physical ODE."""

    def __init__(self, *, physical_system: DAESystem, **kwargs):
        self.physical_system = physical_system
        super().__init__(**kwargs)

    def _physical_blocks(self, coordinates, states, derivatives, physical_parameters):
        system = self.physical_system
        differential = jax.vmap(system.differential_equations.function)(
            coordinates, states, derivatives, physical_parameters
        )
        algebraic = jax.vmap(system.algebraic_equalities.function)(
            coordinates, states, derivatives, physical_parameters
        )
        inequalities = jax.vmap(system.algebraic_inequalities.function)(
            coordinates, states, derivatives, physical_parameters
        )
        batch = coordinates.shape[0]
        differential = jnp.asarray(differential).reshape((batch, -1))
        algebraic = jnp.asarray(algebraic).reshape((batch, -1))
        inequalities = jnp.asarray(inequalities).reshape((batch, -1))
        positive_inequalities = jnp.maximum(inequalities, 0.0)
        combined = jnp.concatenate(
            (differential, algebraic, positive_inequalities), axis=-1
        )
        return differential, algebraic, positive_inequalities, combined


def exact_solution(
    time: np.ndarray,
    initial_position: float,
    initial_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic solution and derivatives for the unit mass--spring system."""

    t = np.asarray(time, dtype=np.float64)
    x0 = float(initial_position)
    v0 = float(initial_velocity)
    position = x0 * np.cos(t) + v0 * np.sin(t)
    velocity = v0 * np.cos(t) - x0 * np.sin(t)
    states = np.column_stack((position, velocity))
    derivatives = np.column_stack((velocity, -position))
    return states, derivatives


def make_splits(
    count: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 3:
        raise ValueError("at least three samples are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than one")

    n_train = max(1, int(np.floor(train_fraction * count)))
    n_validation = max(1, int(np.floor(validation_fraction * count)))
    if n_train + n_validation >= count:
        n_train, n_validation = count - 2, 1

    indices = np.arange(count, dtype=np.int64)
    if mode == "random":
        indices = np.random.default_rng(seed).permutation(indices)
    elif mode != "chronological":
        raise ValueError(f"unsupported split mode: {mode!r}")

    train = np.sort(indices[:n_train])
    validation = np.sort(indices[n_train : n_train + n_validation])
    test = np.sort(indices[n_train + n_validation :])
    return train, validation, test


def generate_dataset(args: argparse.Namespace) -> GeneratedDataset:
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    time = np.linspace(0.0, float(args.horizon), int(args.samples), dtype=np.float64)
    states, derivatives = exact_solution(
        time,
        float(args.initial_position),
        float(args.initial_velocity),
    )
    energy0 = 0.5 * (
        float(args.initial_position) ** 2 + float(args.initial_velocity) ** 2
    )
    amplitude = float(np.sqrt(2.0 * energy0))
    energy = 0.5 * np.sum(states**2, axis=1)
    energy_residual = energy - energy0

    train, validation, test = make_splits(
        len(time),
        float(args.train_fraction),
        float(args.validation_fraction),
        int(args.split_seed),
        str(args.split_mode),
    )
    parameters = np.full((len(time), 1), energy0, dtype=np.float64)
    data = DAEData(
        coordinates=time[:, None],
        parameters=parameters,
        states=states,
        derivatives=derivatives,
        initial_coordinates=np.asarray([[time[0]]], dtype=np.float64),
        initial_targets=states[:1],
    )
    data.save(data_dir / "data.npz")
    np.savez(
        data_dir / "splits.npz",
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
    )
    np.savez(
        data_dir / "trajectory.npz",
        time=time,
        states=states,
        derivatives=derivatives,
        energy=energy,
        energy_residual=energy_residual,
        energy0=np.asarray(energy0),
        initial_condition=np.asarray(
            [args.initial_position, args.initial_velocity], dtype=np.float64
        ),
    )
    np.savetxt(
        data_dir / "trajectory.csv",
        np.column_stack(
            (
                time,
                states[:, 0],
                states[:, 1],
                derivatives[:, 0],
                derivatives[:, 1],
                energy,
                energy_residual,
            )
        ),
        delimiter=",",
        comments="",
        header=(
            "time,position_x,velocity_v,dposition_dt,dvelocity_dt,"
            "energy,energy_residual"
        ),
        fmt="%.17e",
    )
    metadata = {
        "problem": "unit_mass_spring_energy_invariant",
        "equations": {
            "dx_dt": "v",
            "dv_dt": "-x",
            "energy": "0.5*(x^2 + v^2)",
            "algebraic_residual": "0.5*(x^2 + v^2) - energy0",
        },
        "initial_condition": {
            "position_x0": float(args.initial_position),
            "velocity_v0": float(args.initial_velocity),
        },
        "energy0": energy0,
        "amplitude": amplitude,
        "horizon": float(args.horizon),
        "samples": int(args.samples),
        "time_step": float(time[1] - time[0]),
        "maximum_energy_residual": float(np.max(np.abs(energy_residual))),
        "splits": {
            "mode": str(args.split_mode),
            "seed": int(args.split_seed),
            "split_seed": int(args.split_seed),
            "train_count": int(train.size),
            "validation_count": int(validation.size),
            "test_count": int(test.size),
        },
        "state_order": list(STATE_NAMES),
        "data_derivative_order": list(DATA_DERIVATIVE_NAMES),
    }
    (data_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Mass--spring data generation finished.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {len(time)}")
    print(f"  initial condition: x0={args.initial_position:.8g}, v0={args.initial_velocity:.8g}")
    print(f"  energy0: {energy0:.12e}")
    print(f"  max |E-E0|: {np.max(np.abs(energy_residual)):.6e}")
    print(
        "  split counts: "
        f"train={train.size}, validation={validation.size}, test={test.size}"
    )

    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        energy0=energy0,
        amplitude=amplitude,
        horizon=float(args.horizon),
        time_step=float(time[1] - time[0]),
    )


def load_dataset(args: argparse.Namespace) -> GeneratedDataset:
    data_dir = args.data_dir.resolve()
    data = DAEData.load(data_dir / "data.npz")
    with np.load(data_dir / "splits.npz") as archive:
        train = np.asarray(archive["train_indices"], dtype=np.int64)
        validation = np.asarray(archive["validation_indices"], dtype=np.int64)
        test = np.asarray(archive["test_indices"], dtype=np.int64)

    states = np.asarray(data.states, dtype=np.float64)
    coordinates = np.asarray(data.coordinates, dtype=np.float64)
    parameters = np.asarray(data.parameters, dtype=np.float64)
    derivatives = np.asarray(data.derivatives, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError("mass--spring states must have shape (N, 2)")
    if coordinates.shape != (states.shape[0], 1):
        raise ValueError("coordinates must have shape (N, 1)")
    if parameters.shape != (states.shape[0], 1):
        raise ValueError("parameters must have shape (N, 1)")
    if derivatives.shape != (states.shape[0], 2):
        raise ValueError("derivatives must have shape (N, 2)")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(derivatives)):
        raise ValueError("dataset contains non-finite values")
    if not np.all(np.diff(coordinates[:, 0]) > 0.0):
        raise ValueError("time coordinates must be strictly increasing")

    energy0 = float(parameters[0, 0])
    if not np.allclose(parameters[:, 0], energy0, atol=1e-12, rtol=0.0):
        raise ValueError("energy0 must be identical in every parameter row")
    energy_residual = 0.5 * np.sum(states**2, axis=1) - energy0
    if np.max(np.abs(energy_residual)) > 1e-10:
        raise ValueError("stored trajectory does not satisfy the energy invariant")

    horizon = float(coordinates[-1, 0] - coordinates[0, 0])
    time_step = float(np.median(np.diff(coordinates[:, 0])))
    amplitude = float(np.sqrt(2.0 * energy0))
    print("Loaded existing mass--spring dataset.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {states.shape[0]}")
    print(f"  max |E-E0|: {np.max(np.abs(energy_residual)):.6e}")
    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        energy0=energy0,
        amplitude=amplitude,
        horizon=horizon,
        time_step=time_step,
    )


def make_mass_spring_system(
    horizon: float,
    state_scale: float,
    *,
    projection_only: bool,
) -> DAESystem:
    time = Coordinate("time", 0.0, float(horizon))
    position = DifferentialState("position_x", scale=float(state_scale))
    velocity = DifferentialState("velocity_v", scale=float(state_scale))
    energy0 = Parameter("energy0")

    derivatives = (
        DerivativeSpec(position, (1,)),
        DerivativeSpec(velocity, (1,)),
    )

    def differential_equations(_coordinates, states, derivative_values, _parameters):
        x, v = states
        derivative_map = {
            spec.state.name: derivative_values[index]
            for index, spec in enumerate(derivatives)
        }
        dx_dt = derivative_map["position_x"]
        dv_dt = derivative_map["velocity_v"]
        return jnp.asarray((dx_dt - v, dv_dt + x))

    def energy_equation(_coordinates, states, _derivative_values, parameters):
        x, v = states
        e0 = parameters[0]
        return jnp.asarray((0.5 * (x * x + v * v) - e0,))

    differential_block = (
        EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="differential_equations_not_in_projection",
        )
        if projection_only
        else EquationBlock(
            differential_equations,
            size=2,
            affine_in_projection_variables=True,
            name="mass_spring_differential_equations",
        )
    )

    return DAESystem(
        coordinates=(time,),
        differential_states=(position, velocity),
        algebraic_states=(),
        parameters=(energy0,),
        derivatives=derivatives,
        differential_equations=differential_block,
        algebraic_equalities=EquationBlock(
            energy_equation,
            size=1,
            affine_in_projection_variables=False,
            name="mass_spring_energy_invariant",
        ),
        algebraic_inequalities=EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="no_inequalities",
        ),
        name=(
            "mass_spring_energy_projection"
            if projection_only
            else "mass_spring_physical_system"
        ),
    )


def make_discretization(delta: float) -> TaylorNeighborhood:
    return TaylorNeighborhood(
        deltas=(float(delta),),
        order=1,
        radii=(1.0,),
        aggregation="equal_average",
    )


def limit_indices(indices: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or limit >= indices.size:
        return np.asarray(indices, dtype=np.int64)
    if limit < 1:
        raise ValueError("sample limits must be positive")
    positions = np.linspace(0, indices.size - 1, int(limit), dtype=np.int64)
    return np.asarray(indices[positions], dtype=np.int64)


def save_plot(
    path: Path,
    coordinates: np.ndarray,
    exact: np.ndarray,
    raw: np.ndarray,
    projected: np.ndarray,
    energy0: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(coordinates[:, 0])
    t = coordinates[order, 0]
    exact = exact[order]
    raw = raw[order]
    projected = projected[order]

    raw_energy_residual = 0.5 * np.sum(raw**2, axis=1) - energy0
    projected_energy_residual = 0.5 * np.sum(projected**2, axis=1) - energy0

    figure, axes = plt.subplots(4, 1, figsize=(9.0, 10.0), constrained_layout=True)
    axes[0].plot(t, exact[:, 0], label="exact x")
    axes[0].plot(t, raw[:, 0], "--", label="raw x")
    axes[0].plot(t, projected[:, 0], ":", label="projected x")
    axes[0].set_ylabel("position x")
    axes[0].legend(ncol=3)

    axes[1].plot(t, exact[:, 1], label="exact v")
    axes[1].plot(t, raw[:, 1], "--", label="raw v")
    axes[1].plot(t, projected[:, 1], ":", label="projected v")
    axes[1].set_ylabel("velocity v")
    axes[1].legend(ncol=3)

    axes[2].plot(exact[:, 0], exact[:, 1], label="exact orbit")
    axes[2].plot(projected[:, 0], projected[:, 1], "--", label="projected orbit")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("v")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend()

    axes[3].plot(t, raw_energy_residual, "--", label="raw")
    axes[3].plot(t, projected_energy_residual, ":", label="projected")
    axes[3].set_xlabel("time")
    axes[3].set_ylabel(r"$E(x,v)-E_0$")
    axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[3].legend()

    figure.savefig(path, dpi=180)
    plt.close(figure)


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate unit mass--spring data and train DAE-HardNet with a hard "
            "energy invariant."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "02_mass_spring" / "model" / "new_runs",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--initial-position", type=float, default=1.0)
    parser.add_argument("--initial-velocity", type=float, default=0.0)
    parser.add_argument("--horizon", type=float, default=2.0 * np.pi)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-mode", choices=("random", "chronological"), default="random"
    )

    parser.add_argument(
        "--projection",
        choices=("fb", "fischer_burmeister"),
        default="fb",
    )
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--derivative-step", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--projected-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--lambda-derivative", "--derivative-weight", dest="derivative_weight", type=float, default=0.5)
    parser.add_argument("--lambda-physics", "--violation-weight", dest="violation_weight", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--backtracking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=31, help="model/training seed")
    parser.add_argument("--split-seed", type=int, default=31, help="fixed dataset and train/validation/test split seed")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--model", choices=("mlp", "pinn", "daehn", "all"), default="daehn")
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir = result_path(args.output_dir, __file__)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke:
        args.samples = 240
        args.epochs = 20
        args.pretrain_epochs = 10
        args.batch_size = 16
        args.hidden_size = 16
        args.hidden_layers = 2
        args.print_every = 1
        args.validation_every = 1
        args.train_limit = 96
        args.validation_limit = 32
        args.test_limit = 32
    if not np.isfinite(args.initial_position) or not np.isfinite(args.initial_velocity):
        raise ValueError("initial conditions must be finite")
    if args.initial_position == 0.0 and args.initial_velocity == 0.0:
        raise ValueError("the zero initial condition has a degenerate energy manifold")
    if not np.isfinite(args.horizon) or args.horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if args.samples < 20:
        raise ValueError("samples must be at least 20")
    if args.model in ("daehn", "all") and not 0 < args.pretrain_epochs < args.epochs:
        raise ValueError("pretrain_epochs must lie strictly between 0 and epochs for DAEHN")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.delta <= 0.0 or args.derivative_step <= 0.0:
        raise ValueError("delta and derivative_step must be positive")
    if args.delta + args.derivative_step >= 0.25 * args.horizon:
        raise ValueError("projection and derivative stencils are too large for the horizon")
    if args.learning_rate <= 0.0 or args.projected_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not 1 <= args.max_iterations <= 100:
        raise ValueError("max_iterations must lie in [1, 100]")
    if args.noise < 0.0:
        raise ValueError("noise must be nonnegative")


def main() -> None:
    args = parse_args()
    set_reproducible_seed(int(args.seed))
    validate_args(args)

    if args.reuse_data:
        dataset = load_dataset(args)
    else:
        dataset = generate_dataset(args)

    data = dataset.data
    train_indices = limit_indices(dataset.train_indices, args.train_limit)
    validation_indices = limit_indices(dataset.validation_indices, args.validation_limit)
    test_indices = limit_indices(dataset.test_indices, args.test_limit)

    times = np.asarray(data.coordinates[:, 0], dtype=np.float64)
    margin = float(args.delta) + float(args.derivative_step)
    t_min = float(times[0])
    t_max = float(times[-1])

    def keep_interior(indices: np.ndarray, name: str) -> np.ndarray:
        selected = times[indices]
        valid = (selected >= t_min + margin) & (selected <= t_max - margin)
        filtered = np.asarray(indices[valid], dtype=np.int64)
        if filtered.size == 0:
            raise ValueError(f"{name} split is empty after stencil filtering")
        removed = int(indices.size - filtered.size)
        if removed:
            print(f"  removed {removed} {name} samples near boundaries (margin={margin:.6g})")
        return filtered

    train_indices = keep_interior(train_indices, "training")
    validation_indices = keep_interior(validation_indices, "validation")
    test_indices = keep_interior(test_indices, "test")

    state_scale = max(1.0, 1.1 * dataset.amplitude)
    physical_system = make_mass_spring_system(
        dataset.horizon, state_scale, projection_only=False
    )
    projection_system = make_mass_spring_system(
        dataset.horizon, state_scale, projection_only=True
    )
    discretization = make_discretization(float(args.delta))
    compiled_unscaled = compile_dae(
        projection_system,
        discretization=discretization,
        scaling=ScalingConfig(residual_mode="none"),
    )

    # Scale only the projection equation. Physical DE/Alg violations remain
    # unscaled because MassSpringTrainer evaluates the original system.
    energy_scale = max(dataset.energy0, 1.0e-3)
    projection_row_scales = jnp.asarray((energy_scale,), dtype=jnp.float64)
    raw_equalities = compiled_unscaled.raw_equalities
    compiled = replace(
        compiled_unscaled,
        residual_scales=projection_row_scales,
        scaled_equalities=lambda context, variables: (
            raw_equalities(context, variables) / projection_row_scales
        ),
    )

    # Reorder derivative targets to the deterministic compiled derivative order.
    data_column = {"position_x": 0, "velocity_v": 1}
    derivative_column_order = np.asarray(
        [data_column[spec.state.name] for spec in compiled.discretization.derivatives],
        dtype=np.int64,
    )
    derivative_targets = np.asarray(data.derivatives, dtype=np.float64)[
        :, derivative_column_order
    ]
    compiled_derivative_names = tuple(
        "dposition_dt" if spec.state.name == "position_x" else "dvelocity_dt"
        for spec in compiled.discretization.derivatives
    )

    print(f"  compiled derivative order: {compiled_derivative_names}")
    print(f"  Taylor projection step: {args.delta:.8e}")
    print(f"  full-model derivative step: {args.derivative_step:.8e}")
    print(f"  projection energy scale: {energy_scale:.8e}")

    projection_config = ProjectionConfig(
        tolerance=float(args.tolerance),
        max_iterations=int(args.max_iterations),
        backtracking=bool(args.backtracking),
        implicit_gradient=ImplicitGradientConfig(
            differentiate_problem_parameters=True,
            policy="residual_checked",
            inexact_tolerance=max(float(args.tolerance), 1.0e-7),
            adjoint_tolerance=max(float(args.tolerance), 1.0e-7),
            invalid_batch_policy="use_valid",
        ),
    )
    prepared_projection = compiled.prepare_projection(
        batch_size=int(args.batch_size),
        method=str(args.projection),
        config=projection_config,
        dtype=jnp.float64,
        backend="cpu",
    )

    hidden_sizes = (int(args.hidden_size),) * int(args.hidden_layers)

    # Reproducible disjoint train/validation/test splits on one common horizon.
    all_valid_indices = keep_interior(np.arange(len(times), dtype=np.int64), "all")
    train_indices, validation_indices, test_indices, split_info = same_horizon_indices(
        times,
        all_valid_indices,
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        split_mode=str(args.split_mode),
        seed=int(args.split_seed),
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  common train/validation/test horizon: "
        f"[{split_info['horizon_start']:.6g}, {split_info['horizon_end']:.6g}] "
        f"({split_info['split_mode']} split)"
    )

    backbone = TimeScaledMLPBackbone(
        hidden_sizes=hidden_sizes,
        time_scale=dataset.horizon,
        state_scale=state_scale,
        activation=jax.nn.tanh,
    )
    trainer_kwargs = dict(
        physical_system=physical_system,
        compiled=compiled,
        prepared_projection=prepared_projection,
        backbone=backbone,
        initial_coordinates=data.initial_coordinates,
        initial_targets=data.initial_targets,
    )
    train_targets = np.asarray(data.states[train_indices], dtype=np.float64).copy()
    if args.noise > 0.0:
        train_targets += np.random.default_rng(int(args.seed) + 17).normal(
            scale=float(args.noise), size=train_targets.shape
        )

    run_model_suite(
        example=12,
        system_name=physical_system.name,
        model=args.model,
        trainer_cls=MassSpringTrainer,
        trainer_kwargs=trainer_kwargs,
        train_coordinates=data.coordinates[train_indices],
        train_parameters=data.parameters[train_indices],
        train_targets=train_targets,
        train_derivative_targets=derivative_targets[train_indices],
        validation_coordinates=data.coordinates[validation_indices],
        validation_parameters=data.parameters[validation_indices],
        validation_targets=data.states[validation_indices],
        validation_derivative_targets=derivative_targets[validation_indices],
        evaluation_coordinates=data.coordinates[test_indices],
        evaluation_parameters=data.parameters[test_indices],
        evaluation_targets=data.states[test_indices],
        evaluation_derivative_targets=derivative_targets[test_indices],
        output_dir=output_dir,
        hidden_sizes=hidden_sizes,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        pretrain_epochs=int(args.pretrain_epochs),
        learning_rate=float(args.learning_rate),
        projected_learning_rate=float(args.projected_learning_rate),
        lambda_data=float(args.lambda_data),
        lambda_derivative=float(args.derivative_weight),
        lambda_physics=float(args.violation_weight),
        derivative_steps=(float(args.derivative_step),),
        seed=int(args.seed),
        print_every=int(args.print_every),
        validation_every=int(args.validation_every),
        state_names=STATE_NAMES,
        split_info=split_info,
        run_config=vars(args),
        plot=bool(args.plot),
        trajectory_coordinates=data.coordinates[all_valid_indices],
        trajectory_parameters=data.parameters[all_valid_indices],
        trajectory_targets=data.states[all_valid_indices],
        trajectory_derivative_targets=derivative_targets[all_valid_indices],
        case_script=Path(__file__).name,
    )


if __name__ == "__main__":
    main()
