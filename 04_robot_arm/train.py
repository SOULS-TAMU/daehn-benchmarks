#!/usr/bin/env python
r"""Generate robot-arm data and train DAE-HardNet.

Benchmark system
----------------
For three unit-length planar links, define the end-effector map

    e(theta) = [sum_i cos(theta_i), sum_i sin(theta_i)]^T.

The prescribed path is

    p(t) = e(theta(0)) - [sin(2*pi*t)/(2*pi), 0]^T,

and the minimum-norm joint velocity is

    dtheta/dt = J(theta)^T [J(theta) J(theta)^T]^{-1} dp/dt,

where J(theta) = de/dtheta.  The time-dependent algebraic constraints are

    C(theta,t) = e(theta) - p(t) = 0.

The projected phase hard-projects the two kinematic constraints.  The three
original differential equations are retained in the training loss and their
residual is evaluated using derivatives of the complete
NN -> projection -> reconstruction map.  Training uses two optimizer stages:
raw pretraining followed by projected DAE-HardNet training with a fresh,
smaller-step Adam optimizer.

The default initial configuration [0.2, 1.4, -1.0] is nonsingular along the
full one-period path.  The script checks the smallest singular value of J and
refuses trajectories that approach a kinematic singularity.

PowerShell
----------
    python .\examples\15_robot_arm_daehn.py `
        --data-dir data\example_15_robot_arm `
        --output-dir results\example_15_robot_arm_daehn `
        --plot

Quick pipeline check
--------------------
    python .\examples\15_robot_arm_daehn.py --smoke
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

_TWO_PI = 2.0 * np.pi
STATE_NAMES = ("theta1", "theta2", "theta3")
DATA_DERIVATIVE_NAMES = ("dtheta1_dt", "dtheta2_dt", "dtheta3_dt")


def _empty(_coordinates, _states, _derivatives, _parameters):
    return jnp.empty((0,), dtype=jnp.float64)


@dataclass(frozen=True)
class GeneratedDataset:
    data: DAEData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    end_effector0: tuple[float, float]
    horizon: float
    time_step: float
    state_scales: tuple[float, float, float]
    minimum_jacobian_singular_value: float
    maximum_constraint_residual: float


@dataclass(frozen=True)
class TimeScaledMLPBackbone:
    """MLP on normalized time with componentwise physical angle scales."""

    hidden_sizes: tuple[int, ...]
    time_scale: float
    state_scales: tuple[float, ...]
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
            raise ValueError(f"robot-arm backbone expects three outputs, got {output_dim}")
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


class RobotArmTrainer(WeightedLossTrainerMixin, DAETrainer):
    """Project the kinematic constraints; train/report the physical ODE."""

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


def end_effector(theta: np.ndarray) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    return np.stack(
        (np.sum(np.cos(values), axis=-1), np.sum(np.sin(values), axis=-1)),
        axis=-1,
    )


def end_effector_jacobian(theta: np.ndarray) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    return np.stack((-np.sin(values), np.cos(values)), axis=0)


def target_path(time: np.ndarray | float, end_effector0: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=np.float64)
    x = float(end_effector0[0]) - np.sin(_TWO_PI * t) / _TWO_PI
    y = np.full_like(t, float(end_effector0[1]), dtype=np.float64)
    return np.stack((x, y), axis=-1)


def target_velocity(time: float) -> np.ndarray:
    return np.asarray((-np.cos(_TWO_PI * float(time)), 0.0), dtype=np.float64)


def robot_arm_rhs(time: float, theta: np.ndarray) -> np.ndarray:
    jacobian = end_effector_jacobian(theta)
    gram = jacobian @ jacobian.T
    velocity_multiplier = np.linalg.solve(gram, target_velocity(time))
    return np.asarray(jacobian.T @ velocity_multiplier, dtype=np.float64)


def minimum_jacobian_singular_values(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    return np.asarray(
        [
            np.linalg.svd(end_effector_jacobian(theta), compute_uv=False)[-1]
            for theta in values
        ],
        dtype=np.float64,
    )


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
    end_effector0 = end_effector(initial)
    time = np.linspace(0.0, float(args.horizon), int(args.samples), dtype=np.float64)
    solution = solve_ivp(
        fun=robot_arm_rhs,
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
        raise RuntimeError("generated robot-arm trajectory is invalid")

    derivatives = np.stack(
        [robot_arm_rhs(t, state) for t, state in zip(time, states)], axis=0
    )
    end_effector_values = end_effector(states)
    path_values = target_path(time, end_effector0)
    constraint_residual = end_effector_values - path_values
    maximum_constraint_residual = float(np.max(np.abs(constraint_residual)))
    if maximum_constraint_residual > float(args.constraint_tolerance):
        raise RuntimeError(
            "kinematic-constraint drift exceeded tolerance: "
            f"max |e(theta)-p(t)|={maximum_constraint_residual:.6e}"
        )

    singular_values = minimum_jacobian_singular_values(states)
    minimum_singular_value = float(np.min(singular_values))
    if minimum_singular_value <= float(args.minimum_jacobian_singular_value):
        raise RuntimeError(
            "robot-arm trajectory approached a kinematic singularity: "
            f"minimum singular value={minimum_singular_value:.6e}"
        )

    train, validation, test = make_splits(
        len(time),
        float(args.train_fraction),
        float(args.validation_fraction),
        int(args.split_seed),
        str(args.split_mode),
    )
    parameter_vector = np.asarray(end_effector0, dtype=np.float64)
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
        end_effector=end_effector_values,
        target_path=path_values,
        constraint_residual=constraint_residual,
        minimum_jacobian_singular_value=singular_values,
        end_effector0=end_effector0,
        initial_condition=initial,
    )
    np.savetxt(
        data_dir / "trajectory.csv",
        np.column_stack(
            (
                time,
                states,
                derivatives,
                end_effector_values,
                path_values,
                constraint_residual,
                singular_values,
            )
        ),
        delimiter=",",
        comments="",
        header=(
            "time,theta1,theta2,theta3,dtheta1_dt,dtheta2_dt,dtheta3_dt,"
            "end_effector_x,end_effector_y,target_x,target_y,"
            "constraint_x,constraint_y,minimum_jacobian_singular_value"
        ),
        fmt="%.17e",
    )

    state_scales = tuple(
        float(value)
        for value in np.maximum(1.0, 1.1 * np.max(np.abs(states), axis=0))
    )
    metadata = {
        "problem": "three_link_robot_arm_kinematic_constraint",
        "equations": {
            "end_effector_x": "cos(theta1)+cos(theta2)+cos(theta3)",
            "end_effector_y": "sin(theta1)+sin(theta2)+sin(theta3)",
            "path_x": "e0_x-sin(2*pi*t)/(2*pi)",
            "path_y": "e0_y",
            "joint_velocity": "J(theta)^T [J(theta)J(theta)^T]^{-1} p_dot(t)",
            "constraint": "e(theta)-p(t)=0",
        },
        "initial_condition": initial.tolist(),
        "end_effector0": end_effector0.tolist(),
        "horizon": float(args.horizon),
        "samples": int(args.samples),
        "time_step": float(time[1] - time[0]),
        "minimum_jacobian_singular_value": minimum_singular_value,
        "maximum_constraint_residual": maximum_constraint_residual,
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

    print("Robot-arm data generation finished.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {len(time)}")
    print(f"  initial configuration: {initial.tolist()}")
    print(f"  initial end effector: {end_effector0.tolist()}")
    print(f"  minimum Jacobian singular value: {minimum_singular_value:.8e}")
    print(f"  max |e(theta)-p(t)|: {maximum_constraint_residual:.6e}")
    print(
        "  split counts: "
        f"train={train.size}, validation={validation.size}, test={test.size}"
    )

    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        end_effector0=(float(end_effector0[0]), float(end_effector0[1])),
        horizon=float(args.horizon),
        time_step=float(time[1] - time[0]),
        state_scales=state_scales,
        minimum_jacobian_singular_value=minimum_singular_value,
        maximum_constraint_residual=maximum_constraint_residual,
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
        raise ValueError("robot-arm states must have shape (N, 3)")
    if coordinates.shape != (count, 1):
        raise ValueError("coordinates must have shape (N, 1)")
    if parameters.shape != (count, 2):
        raise ValueError("parameters must have shape (N, 2)")
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

    end_effector0 = parameters[0]
    path_values = target_path(coordinates[:, 0], end_effector0)
    constraint_residual = end_effector(states) - path_values
    maximum_constraint_residual = float(np.max(np.abs(constraint_residual)))
    if maximum_constraint_residual > 1e-8:
        raise ValueError("stored trajectory does not satisfy e(theta)=p(t)")

    singular_values = minimum_jacobian_singular_values(states)
    minimum_singular_value = float(np.min(singular_values))
    horizon = float(coordinates[-1, 0] - coordinates[0, 0])
    time_step = float(np.median(np.diff(coordinates[:, 0])))
    state_scales = tuple(
        float(value)
        for value in np.maximum(1.0, 1.1 * np.max(np.abs(states), axis=0))
    )

    print("Loaded existing robot-arm dataset.")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {count}")
    print(f"  initial end effector: {end_effector0.tolist()}")
    print(f"  minimum Jacobian singular value: {minimum_singular_value:.8e}")
    print(f"  max |e(theta)-p(t)|: {maximum_constraint_residual:.6e}")
    return GeneratedDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        end_effector0=(float(end_effector0[0]), float(end_effector0[1])),
        horizon=horizon,
        time_step=time_step,
        state_scales=state_scales,
        minimum_jacobian_singular_value=minimum_singular_value,
        maximum_constraint_residual=maximum_constraint_residual,
    )


def make_robot_arm_system(
    horizon: float,
    state_scales: tuple[float, float, float],
    *,
    projection_only: bool,
) -> DAESystem:
    time = Coordinate("time", 0.0, float(horizon))
    theta1 = DifferentialState("theta1", scale=float(state_scales[0]))
    theta2 = DifferentialState("theta2", scale=float(state_scales[1]))
    theta3 = DifferentialState("theta3", scale=float(state_scales[2]))
    end_effector_x0 = Parameter("end_effector_x0")
    end_effector_y0 = Parameter("end_effector_y0")

    derivatives = (
        DerivativeSpec(theta1, (1,)),
        DerivativeSpec(theta2, (1,)),
        DerivativeSpec(theta3, (1,)),
    )

    def desired_joint_velocity(coordinates, states):
        t = coordinates[0]
        angles = jnp.asarray(states)
        sine = jnp.sin(angles)
        cosine = jnp.cos(angles)
        gram00 = jnp.sum(sine * sine)
        gram01 = -jnp.sum(sine * cosine)
        gram11 = jnp.sum(cosine * cosine)
        determinant = gram00 * gram11 - gram01 * gram01
        # This continuation is inactive on the validated nonsingular trajectory.
        determinant_safe = jnp.maximum(
            determinant, jnp.asarray(1.0e-8, dtype=determinant.dtype)
        )
        path_dx = -jnp.cos(jnp.asarray(_TWO_PI, dtype=t.dtype) * t)
        multiplier0 = gram11 * path_dx / determinant_safe
        multiplier1 = -gram01 * path_dx / determinant_safe
        return -sine * multiplier0 + cosine * multiplier1

    def differential_equations(coordinates, states, derivative_values, _parameters):
        derivative_map = {
            spec.state.name: derivative_values[index]
            for index, spec in enumerate(derivatives)
        }
        actual = jnp.asarray(
            (
                derivative_map["theta1"],
                derivative_map["theta2"],
                derivative_map["theta3"],
            )
        )
        return actual - desired_joint_velocity(coordinates, states)

    def kinematic_constraints(coordinates, states, _derivative_values, parameters):
        t = coordinates[0]
        angle1, angle2, angle3 = states
        e0_x, e0_y = parameters
        end_x = jnp.cos(angle1) + jnp.cos(angle2) + jnp.cos(angle3)
        end_y = jnp.sin(angle1) + jnp.sin(angle2) + jnp.sin(angle3)
        target_x = e0_x - jnp.sin(jnp.asarray(_TWO_PI, dtype=t.dtype) * t) / jnp.asarray(
            _TWO_PI, dtype=t.dtype
        )
        target_y = e0_y
        return jnp.asarray((end_x - target_x, end_y - target_y))

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
            name="robot_arm_joint_velocity_equations",
        )
    )

    return DAESystem(
        coordinates=(time,),
        differential_states=(theta1, theta2, theta3),
        algebraic_states=(),
        parameters=(end_effector_x0, end_effector_y0),
        derivatives=derivatives,
        differential_equations=differential_block,
        algebraic_equalities=EquationBlock(
            kinematic_constraints,
            size=2,
            affine_in_projection_variables=False,
            name="robot_arm_end_effector_constraints",
        ),
        algebraic_inequalities=EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="no_inequalities",
        ),
        name=(
            "robot_arm_kinematic_projection"
            if projection_only
            else "robot_arm_physical_system"
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
    end_effector0: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(coordinates[:, 0])
    t = coordinates[order, 0]
    exact = exact[order]
    raw = raw[order]
    projected = projected[order]
    path_values = target_path(t, np.asarray(end_effector0))
    exact_end = end_effector(exact)
    raw_end = end_effector(raw)
    projected_end = end_effector(projected)
    raw_residual = raw_end - path_values
    projected_residual = projected_end - path_values

    figure, axes = plt.subplots(5, 1, figsize=(9.0, 13.5), constrained_layout=True)
    for index, name in enumerate((r"$\theta_1$", r"$\theta_2$", r"$\theta_3$")):
        axes[0].plot(t, exact[:, index], label=f"exact {name}")
        axes[0].plot(t, projected[:, index], ":", label=f"projected {name}")
    axes[0].set_ylabel("joint angles")
    axes[0].legend(ncol=3)

    for index, name in enumerate((r"$\theta_1$", r"$\theta_2$", r"$\theta_3$")):
        axes[1].plot(t, raw[:, index] - exact[:, index], label=f"raw {name}")
        axes[1].plot(
            t,
            projected[:, index] - exact[:, index],
            ":",
            label=f"projected {name}",
        )
    axes[1].set_ylabel("angle error")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(ncol=3)

    axes[2].plot(path_values[:, 0], path_values[:, 1], label="target path")
    axes[2].plot(raw_end[:, 0], raw_end[:, 1], "--", label="raw end effector")
    axes[2].plot(
        projected_end[:, 0], projected_end[:, 1], ":", label="projected end effector"
    )
    axes[2].set_xlabel("end-effector x")
    axes[2].set_ylabel("end-effector y")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend()

    axes[3].plot(t, raw_residual[:, 0], "--", label="raw x")
    axes[3].plot(t, projected_residual[:, 0], ":", label="projected x")
    axes[3].set_ylabel(r"$e_x-p_x$")
    axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[3].legend()

    axes[4].plot(t, raw_residual[:, 1], "--", label="raw y")
    axes[4].plot(t, projected_residual[:, 1], ":", label="projected y")
    axes[4].set_xlabel("time")
    axes[4].set_ylabel(r"$e_y-p_y$")
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
            "Generate a three-link robot-arm trajectory and train DAE-HardNet "
            "with hard end-effector kinematic constraints."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "04_robot_arm" / "model" / "new_runs",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument(
        "--initial-condition",
        nargs=3,
        type=float,
        metavar=("THETA1_0", "THETA2_0", "THETA3_0"),
        default=(0.2, 1.4, -1.0),
    )
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--constraint-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--minimum-jacobian-singular-value", type=float, default=0.1)
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
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--derivative-step", type=float, default=0.005)
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
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_condition must contain three finite values")
    initial_sigma = np.linalg.svd(
        end_effector_jacobian(initial), compute_uv=False
    )[-1]
    if initial_sigma <= float(args.minimum_jacobian_singular_value):
        raise ValueError(
            "initial condition is too close to a kinematic singularity; "
            f"smallest singular value={initial_sigma:.6e}"
        )
    if not np.isfinite(args.horizon) or args.horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if args.samples < 20:
        raise ValueError("samples must be at least 20")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if args.constraint_tolerance <= 0.0:
        raise ValueError("constraint_tolerance must be positive")
    if args.minimum_jacobian_singular_value <= 0.0:
        raise ValueError("minimum_jacobian_singular_value must be positive")
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

    physical_system = make_robot_arm_system(
        dataset.horizon, dataset.state_scales, projection_only=False
    )
    projection_system = make_robot_arm_system(
        dataset.horizon, dataset.state_scales, projection_only=True
    )
    discretization = make_discretization(float(args.delta))
    compiled_unscaled = compile_dae(
        projection_system,
        discretization=discretization,
        scaling=ScalingConfig(residual_mode="none"),
    )

    # Both projection rows are Cartesian position residuals with unit link lengths.
    projection_row_scales = jnp.asarray((1.0, 1.0), dtype=jnp.float64)
    raw_equalities = compiled_unscaled.raw_equalities
    compiled = replace(
        compiled_unscaled,
        residual_scales=projection_row_scales,
        scaled_equalities=lambda context, variables: (
            raw_equalities(context, variables) / projection_row_scales
        ),
    )

    data_column = {"theta1": 0, "theta2": 1, "theta3": 2}
    derivative_column_order = np.asarray(
        [data_column[spec.state.name] for spec in compiled.discretization.derivatives],
        dtype=np.int64,
    )
    derivative_targets = np.asarray(data.derivatives, dtype=np.float64)[
        :, derivative_column_order
    ]
    derivative_name_map = {
        "theta1": "dtheta1_dt",
        "theta2": "dtheta2_dt",
        "theta3": "dtheta3_dt",
    }
    compiled_derivative_names = tuple(
        derivative_name_map[spec.state.name]
        for spec in compiled.discretization.derivatives
    )

    print(f"  compiled derivative order: {compiled_derivative_names}")
    print(f"  Taylor projection step: {args.delta:.8e}")
    print(f"  full-model derivative step: {args.derivative_step:.8e}")

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
        example=15,
        system_name=physical_system.name,
        model=args.model,
        trainer_cls=RobotArmTrainer,
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
