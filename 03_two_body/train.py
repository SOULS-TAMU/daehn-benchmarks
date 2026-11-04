#!/usr/bin/env python
r"""Generate planar two-body data and train DAE-HardNet.

Physical system
---------------
    dq1/dt = p1
    dq2/dt = p2
    dp1/dt = -mu*q1 / (q1**2 + q2**2)**(3/2)
    dp2/dt = -mu*q2 / (q1**2 + q2**2)**(3/2)

Conserved angular momentum
--------------------------
    L(q1, q2, p1, p2) = q1*p2 - q2*p1 = L0.

The projected phase hard-projects only the angular-momentum invariant. The four
original ODE residuals are evaluated using derivatives of the complete
NN -> projection -> reconstruction map. Training uses two optimizer stages:
raw pretraining followed by projected DAE-HardNet training with a fresh,
smaller-step Adam optimizer.

The default initial condition [1, 0, 0, 1] with mu=1 gives the unit circular
orbit and remains safely away from the collision singularity.

PowerShell
----------
    python .\examples\13_two_body_daehn.py `
        --data-dir data\example_13_two_body `
        --output-dir results\example_13_two_body_daehn `
        --plot

Quick pipeline check
--------------------
    python .\examples\13_two_body_daehn.py --smoke
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
from scipy.integrate import solve_ivp

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

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

STATE_NAMES = ("position_q1", "position_q2", "momentum_p1", "momentum_p2")
DATA_DERIVATIVE_NAMES = ("dq1_dt", "dq2_dt", "dp1_dt", "dp2_dt")


def _empty(_coordinates, _states, _derivatives, _parameters):
    return jnp.empty((0,), dtype=jnp.float64)


@dataclass(frozen=True)
class GeneratedDataset:
    data: DAEData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    angular_momentum0: float
    mu: float
    horizon: float
    time_step: float
    state_scales: tuple[float, float, float, float]
    minimum_radius: float


@dataclass(frozen=True)
class TimeScaledMLPBackbone:
    """MLP on normalized time with componentwise physical output scales."""

    hidden_sizes: tuple[int, ...]
    time_scale: float
    state_scales: tuple[float, ...]
    activation: Any = jax.nn.tanh

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_scale) or self.time_scale <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        scales = np.asarray(self.state_scales, dtype=np.float64)
        if scales.ndim != 1 or scales.size != 4:
            raise ValueError("state_scales must contain four entries")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("state_scales must be finite and positive")

    def init(self, key, input_dim, output_dim, dtype=jnp.float64):
        if int(output_dim) != 4:
            raise ValueError(f"two-body backbone expects four outputs, got {output_dim}")
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
        scales = jnp.asarray(self.state_scales, dtype=raw.dtype)
        return raw * scales


class TwoBodyTrainer(WeightedLossTrainerMixin, DAETrainer):
    """Project angular momentum only; train/report the complete physical ODE."""

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


def two_body_rhs(_time: float, state: np.ndarray, mu: float) -> np.ndarray:
    q1, q2, p1, p2 = np.asarray(state, dtype=np.float64)
    radius_squared = q1 * q1 + q2 * q2
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        raise FloatingPointError("two-body trajectory reached the collision singularity")
    inverse_radius_cubed = radius_squared ** (-1.5)
    return np.asarray(
        (
            p1,
            p2,
            -float(mu) * q1 * inverse_radius_cubed,
            -float(mu) * q2 * inverse_radius_cubed,
        ),
        dtype=np.float64,
    )


def angular_momentum(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    return values[..., 0] * values[..., 3] - values[..., 1] * values[..., 2]


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

    initial = np.asarray(args.initial_condition, dtype=np.float64)
    time = np.linspace(0.0, float(args.horizon), int(args.samples), dtype=np.float64)
    solution = solve_ivp(
        fun=lambda t, state: two_body_rhs(t, state, float(args.mu)),
        t_span=(0.0, float(args.horizon)),
        y0=initial,
        t_eval=time,
        method="DOP853",
        rtol=float(args.rtol),
        atol=float(args.atol),
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 integration failed: {solution.message}")

    states = np.asarray(solution.y.T, dtype=np.float64)
    if states.shape != (time.size, 4) or not np.all(np.isfinite(states)):
        raise RuntimeError("generated two-body trajectory is invalid")
    radii = np.sqrt(np.sum(states[:, :2] ** 2, axis=1))
    minimum_radius = float(np.min(radii))
    if minimum_radius <= float(args.minimum_radius):
        raise RuntimeError(
            "trajectory approached the collision singularity: "
            f"minimum radius={minimum_radius:.6e} <= {args.minimum_radius:.6e}"
        )

    derivatives = np.stack(
        [two_body_rhs(t, state, float(args.mu)) for t, state in zip(time, states)],
        axis=0,
    )
    momentum = angular_momentum(states)
    momentum0 = float(angular_momentum(initial[None, :])[0])
    momentum_residual = momentum - momentum0
    max_momentum_residual = float(np.max(np.abs(momentum_residual)))
    if max_momentum_residual > float(args.invariant_tolerance):
        raise RuntimeError(
            "angular-momentum drift exceeded tolerance: "
            f"{max_momentum_residual:.6e} > {args.invariant_tolerance:.6e}"
        )

    train, validation, test = make_splits(
        len(time),
        float(args.train_fraction),
        float(args.validation_fraction),
        int(args.split_seed),
        str(args.split_mode),
    )
    parameter_vector = np.asarray((args.mu, momentum0), dtype=np.float64)
    parameters = np.broadcast_to(parameter_vector, (len(time), 2)).copy()
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
        radius=radii,
        angular_momentum=momentum,
        angular_momentum_residual=momentum_residual,
        angular_momentum0=np.asarray(momentum0),
        mu=np.asarray(float(args.mu)),
        initial_condition=initial,
    )
    np.savetxt(
        data_dir / "trajectory.csv",
        np.column_stack(
            (
                time,
                states,
                derivatives,
                radii,
                momentum,
                momentum_residual,
            )
        ),
        delimiter=",",
        comments="",
        header=(
            "time,q1,q2,p1,p2,dq1_dt,dq2_dt,dp1_dt,dp2_dt,"
            "radius,angular_momentum,angular_momentum_residual"
        ),
        fmt="%.17e",
    )

    state_scales_array = np.maximum(
        1.0, 1.1 * np.max(np.abs(states), axis=0)
    ).astype(np.float64)
    state_scales = tuple(float(value) for value in state_scales_array)
    metadata = {
        "problem": "planar_two_body_angular_momentum_invariant",
        "equations": {
            "dq1_dt": "p1",
            "dq2_dt": "p2",
            "dp1_dt": "-mu*q1/(q1^2+q2^2)^(3/2)",
            "dp2_dt": "-mu*q2/(q1^2+q2^2)^(3/2)",
            "angular_momentum": "q1*p2 - q2*p1",
            "algebraic_residual": "q1*p2 - q2*p1 - angular_momentum0",
        },
        "mu": float(args.mu),
        "initial_condition": initial.tolist(),
        "angular_momentum0": momentum0,
        "horizon": float(args.horizon),
        "samples": int(args.samples),
        "time_step": float(time[1] - time[0]),
        "minimum_radius": minimum_radius,
        "maximum_angular_momentum_residual": max_momentum_residual,
        "state_scales": list(state_scales),
        "solver": {
            "method": "DOP853",
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "function_evaluations": int(solution.nfev),
        },
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

    print("Two-body data generation finished.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {len(time)}")
    print(f"  initial condition: {initial.tolist()}")
    print(f"  mu: {args.mu:.8g}")
    print(f"  angular momentum L0: {momentum0:.12e}")
    print(f"  minimum radius: {minimum_radius:.8e}")
    print(f"  max |L-L0|: {max_momentum_residual:.6e}")
    print(
        "  split counts: "
        f"train={train.size}, validation={validation.size}, test={test.size}"
    )

    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        angular_momentum0=momentum0,
        mu=float(args.mu),
        horizon=float(args.horizon),
        time_step=float(time[1] - time[0]),
        state_scales=state_scales,
        minimum_radius=minimum_radius,
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
    count = states.shape[0]
    if states.shape != (count, 4):
        raise ValueError("two-body states must have shape (N, 4)")
    if coordinates.shape != (count, 1):
        raise ValueError("coordinates must have shape (N, 1)")
    if parameters.shape != (count, 2):
        raise ValueError("parameters must have shape (N, 2)")
    if derivatives.shape != (count, 4):
        raise ValueError("derivatives must have shape (N, 4)")
    for name, values in (
        ("states", states),
        ("coordinates", coordinates),
        ("parameters", parameters),
        ("derivatives", derivatives),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.diff(coordinates[:, 0]) > 0.0):
        raise ValueError("time coordinates must be strictly increasing")

    mu = float(parameters[0, 0])
    momentum0 = float(parameters[0, 1])
    if not np.allclose(parameters, parameters[:1], atol=1e-12, rtol=0.0):
        raise ValueError("all parameter rows must be identical")
    momentum_residual = angular_momentum(states) - momentum0
    if np.max(np.abs(momentum_residual)) > 1e-8:
        raise ValueError("stored trajectory does not satisfy angular momentum")
    radii = np.sqrt(np.sum(states[:, :2] ** 2, axis=1))
    minimum_radius = float(np.min(radii))
    horizon = float(coordinates[-1, 0] - coordinates[0, 0])
    time_step = float(np.median(np.diff(coordinates[:, 0])))
    state_scales = tuple(
        float(value)
        for value in np.maximum(1.0, 1.1 * np.max(np.abs(states), axis=0))
    )

    print("Loaded existing two-body dataset.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {count}")
    print(f"  angular momentum L0: {momentum0:.12e}")
    print(f"  minimum radius: {minimum_radius:.8e}")
    print(f"  max |L-L0|: {np.max(np.abs(momentum_residual)):.6e}")
    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        angular_momentum0=momentum0,
        mu=mu,
        horizon=horizon,
        time_step=time_step,
        state_scales=state_scales,
        minimum_radius=minimum_radius,
    )


def make_two_body_system(
    horizon: float,
    state_scales: tuple[float, float, float, float],
    *,
    projection_only: bool,
    radius_floor: float,
) -> DAESystem:
    time = Coordinate("time", 0.0, float(horizon))
    q1 = DifferentialState("position_q1", scale=float(state_scales[0]))
    q2 = DifferentialState("position_q2", scale=float(state_scales[1]))
    p1 = DifferentialState("momentum_p1", scale=float(state_scales[2]))
    p2 = DifferentialState("momentum_p2", scale=float(state_scales[3]))
    mu = Parameter("mu")
    momentum0 = Parameter("angular_momentum0")

    # Match DAEHN's deterministic derivative ordering. The compiled vector is
    # [dp1/dt, dp2/dt, dq1/dt, dq2/dt], not declaration/state order.
    derivatives = (
        DerivativeSpec(p1, (1,)),
        DerivativeSpec(p2, (1,)),
        DerivativeSpec(q1, (1,)),
        DerivativeSpec(q2, (1,)),
    )

    def differential_equations(_coordinates, states, derivative_values, parameters):
        q1_value, q2_value, p1_value, p2_value = states
        derivative_map = {
            spec.state.name: derivative_values[index]
            for index, spec in enumerate(derivatives)
        }
        dq1_dt = derivative_map["position_q1"]
        dq2_dt = derivative_map["position_q2"]
        dp1_dt = derivative_map["momentum_p1"]
        dp2_dt = derivative_map["momentum_p2"]
        mu_value = parameters[0]
        radius_squared = q1_value * q1_value + q2_value * q2_value
        # Smoothly continue the singular force outside the collision-free
        # training region. On the generated trajectory r >= minimum_radius, so
        # this has no effect on the physical residual where data are present.
        floor_squared = jnp.asarray(radius_floor * radius_floor, dtype=radius_squared.dtype)
        radius_squared = jnp.maximum(radius_squared, floor_squared)
        inverse_radius_cubed = radius_squared ** (-1.5)
        return jnp.asarray(
            (
                dq1_dt - p1_value,
                dq2_dt - p2_value,
                dp1_dt + mu_value * q1_value * inverse_radius_cubed,
                dp2_dt + mu_value * q2_value * inverse_radius_cubed,
            )
        )

    def momentum_equation(_coordinates, states, _derivative_values, parameters):
        q1_value, q2_value, p1_value, p2_value = states
        l0 = parameters[1]
        return jnp.asarray((q1_value * p2_value - q2_value * p1_value - l0,))

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
            size=4,
            affine_in_projection_variables=False,
            name="two_body_differential_equations",
        )
    )

    return DAESystem(
        coordinates=(time,),
        differential_states=(q1, q2, p1, p2),
        algebraic_states=(),
        parameters=(mu, momentum0),
        derivatives=derivatives,
        differential_equations=differential_block,
        algebraic_equalities=EquationBlock(
            momentum_equation,
            size=1,
            affine_in_projection_variables=False,
            name="two_body_angular_momentum_invariant",
        ),
        algebraic_inequalities=EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="no_inequalities",
        ),
        name=(
            "two_body_angular_momentum_projection"
            if projection_only
            else "two_body_physical_system"
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
    momentum0: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(coordinates[:, 0])
    t = coordinates[order, 0]
    exact = exact[order]
    raw = raw[order]
    projected = projected[order]

    raw_momentum_residual = angular_momentum(raw) - momentum0
    projected_momentum_residual = angular_momentum(projected) - momentum0

    figure, axes = plt.subplots(5, 1, figsize=(9.0, 13.0), constrained_layout=True)
    axes[0].plot(t, exact[:, 0], label="exact q1")
    axes[0].plot(t, raw[:, 0], "--", label="raw q1")
    axes[0].plot(t, projected[:, 0], ":", label="projected q1")
    axes[0].plot(t, exact[:, 1], label="exact q2")
    axes[0].plot(t, projected[:, 1], ":", label="projected q2")
    axes[0].set_ylabel("positions")
    axes[0].legend(ncol=3)

    axes[1].plot(t, exact[:, 2], label="exact p1")
    axes[1].plot(t, raw[:, 2], "--", label="raw p1")
    axes[1].plot(t, projected[:, 2], ":", label="projected p1")
    axes[1].plot(t, exact[:, 3], label="exact p2")
    axes[1].plot(t, projected[:, 3], ":", label="projected p2")
    axes[1].set_ylabel("momenta")
    axes[1].legend(ncol=3)

    axes[2].plot(exact[:, 0], exact[:, 1], label="exact orbit")
    axes[2].plot(projected[:, 0], projected[:, 1], "--", label="projected orbit")
    axes[2].scatter([0.0], [0.0], marker="*", s=70, label="central body")
    axes[2].set_xlabel("q1")
    axes[2].set_ylabel("q2")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend()

    axes[3].plot(raw[:, 0], raw[:, 3], "--", label="raw q1 vs p2")
    axes[3].plot(projected[:, 0], projected[:, 3], ":", label="projected q1 vs p2")
    axes[3].set_xlabel("q1")
    axes[3].set_ylabel("p2")
    axes[3].legend()

    axes[4].plot(t, raw_momentum_residual, "--", label="raw")
    axes[4].plot(t, projected_momentum_residual, ":", label="projected")
    axes[4].set_xlabel("time")
    axes[4].set_ylabel(r"$q_1p_2-q_2p_1-L_0$")
    axes[4].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[4].legend()

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
            "Generate planar two-body data and train DAE-HardNet with a hard "
            "angular-momentum invariant."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "03_two_body" / "model" / "new_runs",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument(
        "--initial-condition",
        nargs=4,
        type=float,
        metavar=("Q1_0", "Q2_0", "P1_0", "P2_0"),
        default=(1.0, 0.0, 0.0, 1.0),
    )
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--horizon", type=float, default=2.0 * np.pi)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--minimum-radius", type=float, default=5.0e-2)
    parser.add_argument("--invariant-tolerance", type=float, default=1.0e-8)
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=150)
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
    initial = np.asarray(args.initial_condition, dtype=np.float64)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_condition must contain four finite values")
    initial_radius = float(np.linalg.norm(initial[:2]))
    if initial_radius <= float(args.minimum_radius):
        raise ValueError("initial position is too close to the collision singularity")
    initial_momentum = float(angular_momentum(initial[None, :])[0])
    if abs(initial_momentum) <= 1.0e-8:
        raise ValueError(
            "the hard angular-momentum projection requires a nondegenerate "
            "initial angular momentum"
        )
    if not np.isfinite(args.mu) or args.mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    if not np.isfinite(args.horizon) or args.horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if args.samples < 20:
        raise ValueError("samples must be at least 20")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if args.minimum_radius <= 0.0 or args.invariant_tolerance <= 0.0:
        raise ValueError("minimum_radius and invariant_tolerance must be positive")
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
    if args.violation_weight < 0.0:
        raise ValueError("lambda_physics must be nonnegative")


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

    residual_radius_floor = max(0.25 * dataset.minimum_radius, 1.0e-2)
    physical_system = make_two_body_system(
        dataset.horizon,
        dataset.state_scales,
        projection_only=False,
        radius_floor=residual_radius_floor,
    )
    projection_system = make_two_body_system(
        dataset.horizon,
        dataset.state_scales,
        projection_only=True,
        radius_floor=residual_radius_floor,
    )
    discretization = make_discretization(float(args.delta))
    compiled_unscaled = compile_dae(
        projection_system,
        discretization=discretization,
        scaling=ScalingConfig(residual_mode="none"),
    )

    momentum_scale = max(abs(dataset.angular_momentum0), 1.0e-3)
    projection_row_scales = jnp.asarray((momentum_scale,), dtype=jnp.float64)
    raw_equalities = compiled_unscaled.raw_equalities
    compiled = replace(
        compiled_unscaled,
        residual_scales=projection_row_scales,
        scaled_equalities=lambda context, variables: (
            raw_equalities(context, variables) / projection_row_scales
        ),
    )

    data_column = {
        "position_q1": 0,
        "position_q2": 1,
        "momentum_p1": 2,
        "momentum_p2": 3,
    }
    derivative_column_order = np.asarray(
        [data_column[spec.state.name] for spec in compiled.discretization.derivatives],
        dtype=np.int64,
    )
    derivative_targets = np.asarray(data.derivatives, dtype=np.float64)[
        :, derivative_column_order
    ]
    derivative_name_map = {
        "position_q1": "dq1_dt",
        "position_q2": "dq2_dt",
        "momentum_p1": "dp1_dt",
        "momentum_p2": "dp2_dt",
    }
    compiled_derivative_names = tuple(
        derivative_name_map[spec.state.name]
        for spec in compiled.discretization.derivatives
    )

    print(f"  compiled derivative order: {compiled_derivative_names}")
    print(f"  Taylor projection step: {args.delta:.8e}")
    print(f"  full-model derivative step: {args.derivative_step:.8e}")
    print(f"  projection angular-momentum scale: {momentum_scale:.8e}")
    print(f"  residual radius floor: {residual_radius_floor:.8e}")

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
        state_scales=dataset.state_scales,
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
        example=13,
        system_name=physical_system.name,
        model=args.model,
        trainer_cls=TwoBodyTrainer,
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
