#!/usr/bin/env python
r"""Generate torque-free rigid-body data and train DAE-HardNet.

Physical system (Euler equations in principal axes)
---------------------------------------------------
    dω1/dt = ((I2-I3)/I1) ω2 ω3
    dω2/dt = ((I3-I1)/I2) ω3 ω1
    dω3/dt = ((I1-I2)/I3) ω1 ω2

Conserved quantities
--------------------
    E(ω)  = 0.5 * (I1*ω1² + I2*ω2² + I3*ω3²)
    M²(ω) = (I1*ω1)² + (I2*ω2)² + (I3*ω3)²

The projected phase hard-projects E=E0 and M²=M0². The three Euler
residuals are evaluated with derivatives of the complete
NN -> projection -> reconstruction map. Training is split into raw
pretraining and projected DAE-HardNet training with a fresh, smaller-step
Adam optimizer.

PowerShell
----------
    python .\examples\16_rigid_body_daehn.py `
        --data-dir data\example_16_rigid_body `
        --output-dir results\example_16_rigid_body_daehn `
        --plot

Quick pipeline check
--------------------
    python .\examples\16_rigid_body_daehn.py --smoke
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

STATE_NAMES = ("omega_1", "omega_2", "omega_3")
DATA_DERIVATIVE_NAMES = ("domega1_dt", "domega2_dt", "domega3_dt")


def _empty(_coordinates, _states, _derivatives, _parameters):
    return jnp.empty((0,), dtype=jnp.float64)


@dataclass(frozen=True)
class GeneratedDataset:
    data: DAEData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    inertias: tuple[float, float, float]
    energy0: float
    momentum_squared0: float
    horizon: float
    time_step: float
    state_scales: tuple[float, float, float]
    minimum_invariant_singular_value: float


@dataclass(frozen=True)
class TimeScaledMLPBackbone:
    """MLP on normalized time with physical component output scales."""

    hidden_sizes: tuple[int, ...]
    time_scale: float
    state_scales: tuple[float, float, float]
    activation: Any = jax.nn.tanh

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_scale) or self.time_scale <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        scales = np.asarray(self.state_scales, dtype=np.float64)
        if scales.shape != (3,):
            raise ValueError("state_scales must contain three entries")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("state_scales must be finite and positive")

    def init(self, key, input_dim, output_dim, dtype=jnp.float64):
        if int(output_dim) != 3:
            raise ValueError(f"rigid-body backbone expects three outputs, got {output_dim}")
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
        return raw * jnp.asarray(self.state_scales, dtype=raw.dtype)


class RigidBodyTrainer(WeightedLossTrainerMixin, DAETrainer):
    """Project the invariants while training/reporting the physical Euler ODE."""

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


def rigid_body_rhs(
    _time: float,
    state: np.ndarray,
    inertias: tuple[float, float, float],
) -> np.ndarray:
    omega1, omega2, omega3 = np.asarray(state, dtype=np.float64)
    inertia1, inertia2, inertia3 = inertias
    return np.asarray(
        (
            ((inertia2 - inertia3) / inertia1) * omega2 * omega3,
            ((inertia3 - inertia1) / inertia2) * omega3 * omega1,
            ((inertia1 - inertia2) / inertia3) * omega1 * omega2,
        ),
        dtype=np.float64,
    )


def kinetic_energy(
    states: np.ndarray,
    inertias: tuple[float, float, float],
) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    inertia = np.asarray(inertias, dtype=np.float64)
    return 0.5 * np.sum(inertia * values * values, axis=-1)


def momentum_squared(
    states: np.ndarray,
    inertias: tuple[float, float, float],
) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    inertia = np.asarray(inertias, dtype=np.float64)
    return np.sum((inertia * values) ** 2, axis=-1)


def invariant_jacobian_singular_values(
    states: np.ndarray,
    inertias: tuple[float, float, float],
) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    inertia = np.asarray(inertias, dtype=np.float64)
    rows = []
    for omega in values:
        jacobian = np.vstack(
            (
                inertia * omega,
                2.0 * inertia * inertia * omega,
            )
        )
        rows.append(np.linalg.svd(jacobian, compute_uv=False))
    return np.asarray(rows, dtype=np.float64)


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
    inertias = tuple(float(value) for value in args.inertias)
    time = np.linspace(0.0, float(args.horizon), int(args.samples), dtype=np.float64)
    solution = solve_ivp(
        fun=lambda t, state: rigid_body_rhs(t, state, inertias),
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
    if states.shape != (time.size, 3) or not np.all(np.isfinite(states)):
        raise RuntimeError("generated rigid-body trajectory is invalid")

    derivatives = np.stack(
        [rigid_body_rhs(t, state, inertias) for t, state in zip(time, states)],
        axis=0,
    )
    energy_values = kinetic_energy(states, inertias)
    momentum_values = momentum_squared(states, inertias)
    energy0 = float(kinetic_energy(initial[None, :], inertias)[0])
    momentum_squared0 = float(momentum_squared(initial[None, :], inertias)[0])
    energy_residual = energy_values - energy0
    momentum_residual = momentum_values - momentum_squared0
    max_energy_residual = float(np.max(np.abs(energy_residual)))
    max_momentum_residual = float(np.max(np.abs(momentum_residual)))
    if max(max_energy_residual, max_momentum_residual) > float(args.invariant_tolerance):
        raise RuntimeError(
            "invariant drift exceeded tolerance: "
            f"max |E-E0|={max_energy_residual:.6e}, "
            f"max |M²-M0²|={max_momentum_residual:.6e}"
        )

    invariant_singular_values = invariant_jacobian_singular_values(states, inertias)
    minimum_invariant_singular_value = float(
        np.min(invariant_singular_values[:, -1])
    )
    if minimum_invariant_singular_value <= float(args.minimum_invariant_singular_value):
        raise RuntimeError(
            "the energy and angular-momentum constraints became nearly dependent: "
            f"minimum singular value={minimum_invariant_singular_value:.6e}"
        )

    train, validation, test = make_splits(
        len(time),
        float(args.train_fraction),
        float(args.validation_fraction),
        int(args.split_seed),
        str(args.split_mode),
    )
    parameter_vector = np.asarray(
        (*inertias, energy0, momentum_squared0), dtype=np.float64
    )
    parameters = np.broadcast_to(parameter_vector, (len(time), 5)).copy()
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
        kinetic_energy=energy_values,
        kinetic_energy_residual=energy_residual,
        angular_momentum_squared=momentum_values,
        angular_momentum_squared_residual=momentum_residual,
        energy0=np.asarray(energy0),
        momentum_squared0=np.asarray(momentum_squared0),
        inertias=np.asarray(inertias),
        invariant_singular_values=invariant_singular_values,
        initial_condition=initial,
    )
    np.savetxt(
        data_dir / "trajectory.csv",
        np.column_stack(
            (
                time,
                states,
                derivatives,
                energy_values,
                energy_residual,
                momentum_values,
                momentum_residual,
                invariant_singular_values[:, -1],
            )
        ),
        delimiter=",",
        comments="",
        header=(
            "time,omega1,omega2,omega3,domega1_dt,domega2_dt,domega3_dt,"
            "kinetic_energy,kinetic_energy_residual,angular_momentum_squared,"
            "angular_momentum_squared_residual,"
            "minimum_invariant_jacobian_singular_value"
        ),
        fmt="%.17e",
    )

    state_scales = tuple(
        float(value)
        for value in np.maximum(1.0, 1.1 * np.max(np.abs(states), axis=0))
    )
    metadata = {
        "problem": "torque_free_rigid_body_two_invariants",
        "equations": {
            "domega1_dt": "((I2-I3)/I1)*omega2*omega3",
            "domega2_dt": "((I3-I1)/I2)*omega3*omega1",
            "domega3_dt": "((I1-I2)/I3)*omega1*omega2",
            "kinetic_energy": "0.5*(I1*omega1^2+I2*omega2^2+I3*omega3^2)",
            "angular_momentum_squared": "(I1*omega1)^2+(I2*omega2)^2+(I3*omega3)^2",
        },
        "initial_condition": initial.tolist(),
        "inertias": list(inertias),
        "energy0": energy0,
        "momentum_squared0": momentum_squared0,
        "horizon": float(args.horizon),
        "samples": int(args.samples),
        "time_step": float(time[1] - time[0]),
        "minimum_invariant_jacobian_singular_value": minimum_invariant_singular_value,
        "maximum_energy_residual": max_energy_residual,
        "maximum_momentum_squared_residual": max_momentum_residual,
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Rigid-body data generation finished.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {len(time)}")
    print(f"  initial angular velocity: {initial.tolist()}")
    print(f"  inertias: {list(inertias)}")
    print(f"  kinetic energy E0: {energy0:.12e}")
    print(f"  momentum squared M0²: {momentum_squared0:.12e}")
    print(
        "  minimum invariant-Jacobian singular value: "
        f"{minimum_invariant_singular_value:.8e}"
    )
    print(f"  max |E-E0|: {max_energy_residual:.6e}")
    print(f"  max |M²-M0²|: {max_momentum_residual:.6e}")
    print(
        "  split counts: "
        f"train={train.size}, validation={validation.size}, test={test.size}"
    )

    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        inertias=inertias,
        energy0=energy0,
        momentum_squared0=momentum_squared0,
        horizon=float(args.horizon),
        time_step=float(time[1] - time[0]),
        state_scales=state_scales,
        minimum_invariant_singular_value=minimum_invariant_singular_value,
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
    if states.shape != (count, 3):
        raise ValueError("rigid-body states must have shape (N, 3)")
    if coordinates.shape != (count, 1):
        raise ValueError("coordinates must have shape (N, 1)")
    if parameters.shape != (count, 5):
        raise ValueError("parameters must have shape (N, 5)")
    if derivatives.shape != (count, 3):
        raise ValueError("derivatives must have shape (N, 3)")
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
    if not np.allclose(parameters, parameters[:1], atol=1e-12, rtol=0.0):
        raise ValueError("all parameter rows must be identical")

    inertias = tuple(float(value) for value in parameters[0, :3])
    energy0 = float(parameters[0, 3])
    momentum_squared0 = float(parameters[0, 4])
    energy_residual = kinetic_energy(states, inertias) - energy0
    momentum_residual = momentum_squared(states, inertias) - momentum_squared0
    if max(np.max(np.abs(energy_residual)), np.max(np.abs(momentum_residual))) > 1e-8:
        raise ValueError("stored trajectory does not satisfy the two invariants")

    invariant_singular_values = invariant_jacobian_singular_values(states, inertias)
    minimum_invariant_singular_value = float(
        np.min(invariant_singular_values[:, -1])
    )
    horizon = float(coordinates[-1, 0] - coordinates[0, 0])
    time_step = float(np.median(np.diff(coordinates[:, 0])))
    state_scales = tuple(
        float(value)
        for value in np.maximum(1.0, 1.1 * np.max(np.abs(states), axis=0))
    )

    print("Loaded existing rigid-body dataset.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {count}")
    print(f"  inertias: {list(inertias)}")
    print(f"  kinetic energy E0: {energy0:.12e}")
    print(f"  momentum squared M0²: {momentum_squared0:.12e}")
    print(f"  max |E-E0|: {np.max(np.abs(energy_residual)):.6e}")
    print(f"  max |M²-M0²|: {np.max(np.abs(momentum_residual)):.6e}")
    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        inertias=inertias,
        energy0=energy0,
        momentum_squared0=momentum_squared0,
        horizon=horizon,
        time_step=time_step,
        state_scales=state_scales,
        minimum_invariant_singular_value=minimum_invariant_singular_value,
    )


def make_rigid_body_system(
    horizon: float,
    state_scales: tuple[float, float, float],
    *,
    projection_only: bool,
) -> DAESystem:
    time = Coordinate("time", 0.0, float(horizon))
    omega1_state = DifferentialState("omega_1", scale=float(state_scales[0]))
    omega2_state = DifferentialState("omega_2", scale=float(state_scales[1]))
    omega3_state = DifferentialState("omega_3", scale=float(state_scales[2]))
    inertia1 = Parameter("inertia_1")
    inertia2 = Parameter("inertia_2")
    inertia3 = Parameter("inertia_3")
    energy0 = Parameter("energy0")
    momentum_squared0 = Parameter("momentum_squared0")

    derivatives = (
        DerivativeSpec(omega1_state, (1,)),
        DerivativeSpec(omega2_state, (1,)),
        DerivativeSpec(omega3_state, (1,)),
    )

    def differential_equations(_coordinates, states, derivative_values, parameters):
        omega1, omega2, omega3 = states
        inertia_1, inertia_2, inertia_3, _energy0, _momentum0 = parameters
        derivative_map = {
            spec.state.name: derivative_values[index]
            for index, spec in enumerate(derivatives)
        }
        domega1 = derivative_map["omega_1"]
        domega2 = derivative_map["omega_2"]
        domega3 = derivative_map["omega_3"]
        rhs1 = ((inertia_2 - inertia_3) / inertia_1) * omega2 * omega3
        rhs2 = ((inertia_3 - inertia_1) / inertia_2) * omega3 * omega1
        rhs3 = ((inertia_1 - inertia_2) / inertia_3) * omega1 * omega2
        return jnp.asarray((domega1 - rhs1, domega2 - rhs2, domega3 - rhs3))

    def invariant_equations(_coordinates, states, _derivative_values, parameters):
        omega1, omega2, omega3 = states
        inertia_1, inertia_2, inertia_3, energy_target, momentum_target = parameters
        energy_value = 0.5 * (
            inertia_1 * omega1 * omega1
            + inertia_2 * omega2 * omega2
            + inertia_3 * omega3 * omega3
        )
        momentum_value = (
            (inertia_1 * omega1) ** 2
            + (inertia_2 * omega2) ** 2
            + (inertia_3 * omega3) ** 2
        )
        return jnp.asarray(
            (energy_value - energy_target, momentum_value - momentum_target)
        )

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
            size=3,
            affine_in_projection_variables=False,
            name="rigid_body_euler_equations",
        )
    )

    return DAESystem(
        coordinates=(time,),
        differential_states=(omega1_state, omega2_state, omega3_state),
        algebraic_states=(),
        parameters=(inertia1, inertia2, inertia3, energy0, momentum_squared0),
        derivatives=derivatives,
        differential_equations=differential_block,
        algebraic_equalities=EquationBlock(
            invariant_equations,
            size=2,
            affine_in_projection_variables=False,
            name="kinetic_energy_and_momentum_squared_invariants",
        ),
        algebraic_inequalities=EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="no_inequalities",
        ),
        name=(
            "rigid_body_invariant_projection"
            if projection_only
            else "rigid_body_physical_system"
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
    inertias: tuple[float, float, float],
    energy0: float,
    momentum_squared0: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(coordinates[:, 0])
    time = coordinates[order, 0]
    exact = exact[order]
    raw = raw[order]
    projected = projected[order]

    raw_energy_residual = kinetic_energy(raw, inertias) - energy0
    projected_energy_residual = kinetic_energy(projected, inertias) - energy0
    raw_momentum_residual = momentum_squared(raw, inertias) - momentum_squared0
    projected_momentum_residual = (
        momentum_squared(projected, inertias) - momentum_squared0
    )

    figure, axes = plt.subplots(6, 1, figsize=(9.0, 15.5), constrained_layout=True)
    labels = (r"$\omega_1$", r"$\omega_2$", r"$\omega_3$")
    for index, label in enumerate(labels):
        axes[index].plot(time, exact[:, index], label=f"exact {label}")
        axes[index].plot(time, raw[:, index], "--", label=f"raw {label}")
        axes[index].plot(
            time, projected[:, index], ":", label=f"projected {label}"
        )
        axes[index].set_ylabel(label)
        axes[index].legend(ncol=3)

    axes[3].plot(exact[:, 0], exact[:, 1], label="exact")
    axes[3].plot(projected[:, 0], projected[:, 1], "--", label="projected")
    axes[3].set_xlabel(r"$\omega_1$")
    axes[3].set_ylabel(r"$\omega_2$")
    axes[3].legend()

    axes[4].plot(time, raw_energy_residual, "--", label="raw")
    axes[4].plot(time, projected_energy_residual, ":", label="projected")
    axes[4].set_ylabel(r"$E-E_0$")
    axes[4].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[4].legend()

    axes[5].plot(time, raw_momentum_residual, "--", label="raw")
    axes[5].plot(time, projected_momentum_residual, ":", label="projected")
    axes[5].set_xlabel("time")
    axes[5].set_ylabel(r"$M^2-M_0^2$")
    axes[5].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[5].legend()

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
            "Generate torque-free rigid-body data and train DAE-HardNet with "
            "hard kinetic-energy and angular-momentum invariants."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "05_rigid_body" / "model" / "new_runs",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument(
        "--initial-condition",
        nargs=3,
        type=float,
        metavar=("OMEGA1", "OMEGA2", "OMEGA3"),
        default=(0.8, 0.6, 0.4),
    )
    parser.add_argument(
        "--inertias",
        nargs=3,
        type=float,
        metavar=("I1", "I2", "I3"),
        default=(1.0, 2.0, 3.0),
    )
    parser.add_argument("--horizon", type=float, default=12.0)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--invariant-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--minimum-invariant-singular-value", type=float, default=1.0e-3
    )
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
        args.horizon = 4.0
        args.samples = 240
        args.epochs = 100
        args.pretrain_epochs = 80
        args.batch_size = 16
        args.hidden_size = 32
        args.hidden_layers = 3
        args.print_every = 10
        args.train_limit = 128
        args.validation_limit = 32
        args.test_limit = 32

    initial = np.asarray(args.initial_condition, dtype=np.float64)
    inertias = np.asarray(args.inertias, dtype=np.float64)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_condition must contain three finite values")
    if inertias.shape != (3,) or not np.all(np.isfinite(inertias)):
        raise ValueError("inertias must contain three finite values")
    if np.any(inertias <= 0.0):
        raise ValueError("principal inertias must be positive")
    if np.min(np.abs(np.subtract.outer(inertias, inertias) + np.eye(3))) < 1.0e-8:
        raise ValueError("use three distinct principal inertias")
    initial_singular_values = invariant_jacobian_singular_values(
        initial[None, :], tuple(float(value) for value in inertias)
    )[0]
    if initial_singular_values[-1] <= float(args.minimum_invariant_singular_value):
        raise ValueError(
            "initial condition makes the two invariant gradients nearly dependent; "
            "use a non-axis-aligned angular velocity"
        )
    if not np.isfinite(args.horizon) or args.horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if args.samples < 20:
        raise ValueError("samples must be at least 20")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if args.invariant_tolerance <= 0.0:
        raise ValueError("invariant_tolerance must be positive")
    if args.minimum_invariant_singular_value <= 0.0:
        raise ValueError("minimum_invariant_singular_value must be positive")
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

    dataset = load_dataset(args) if args.reuse_data else generate_dataset(args)
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
            print(
                f"  removed {removed} {name} samples near boundaries "
                f"(margin={margin:.6g})"
            )
        return filtered

    train_indices = keep_interior(train_indices, "training")
    validation_indices = keep_interior(validation_indices, "validation")
    test_indices = keep_interior(test_indices, "test")

    physical_system = make_rigid_body_system(
        dataset.horizon, dataset.state_scales, projection_only=False
    )
    projection_system = make_rigid_body_system(
        dataset.horizon, dataset.state_scales, projection_only=True
    )
    discretization = make_discretization(float(args.delta))
    compiled_unscaled = compile_dae(
        projection_system,
        discretization=discretization,
        scaling=ScalingConfig(residual_mode="none"),
    )

    energy_scale = max(abs(dataset.energy0), 1.0e-3)
    momentum_scale = max(abs(dataset.momentum_squared0), 1.0e-3)
    projection_row_scales = jnp.asarray(
        (energy_scale, momentum_scale), dtype=jnp.float64
    )
    raw_equalities = compiled_unscaled.raw_equalities
    compiled = replace(
        compiled_unscaled,
        residual_scales=projection_row_scales,
        scaled_equalities=lambda context, variables: (
            raw_equalities(context, variables) / projection_row_scales
        ),
    )

    data_column = {"omega_1": 0, "omega_2": 1, "omega_3": 2}
    derivative_column_order = np.asarray(
        [data_column[spec.state.name] for spec in compiled.discretization.derivatives],
        dtype=np.int64,
    )
    derivative_targets = np.asarray(data.derivatives, dtype=np.float64)[
        :, derivative_column_order
    ]
    derivative_name_map = {
        "omega_1": "domega1_dt",
        "omega_2": "domega2_dt",
        "omega_3": "domega3_dt",
    }
    compiled_derivative_names = tuple(
        derivative_name_map[spec.state.name]
        for spec in compiled.discretization.derivatives
    )

    print(f"  compiled derivative order: {compiled_derivative_names}")
    print(f"  Taylor projection step: {args.delta:.8e}")
    print(f"  full-model derivative step: {args.derivative_step:.8e}")
    print(f"  projection energy scale: {energy_scale:.8e}")
    print(f"  projection momentum-squared scale: {momentum_scale:.8e}")

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
        example=16,
        system_name=physical_system.name,
        model=args.model,
        trainer_cls=RigidBodyTrainer,
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
