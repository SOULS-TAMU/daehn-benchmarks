"""Generate Lotka--Volterra data for the constrained DAEHN benchmark.

Physical system
---------------
    dx/dt = alpha*x - beta*x*y
    dy/dt = -gamma*y + delta*x*y

For positive x and y, the trajectory preserves

    V(x, y) = delta*x - gamma*log(x) + beta*y - alpha*log(y).

The algebraic residual used by the later training script is

    g(x, y) = V(x, y) - V(x0, y0) = 0.

This script only generates and validates data; it does not train a model. Run ``11_lotka_volterra_daehn_training.py --model mlp|pinn|daehn|all`` to train the benchmark models on the generated data.

Outputs
-------
trajectory.csv
    Human-readable trajectory and diagnostic columns.
trajectory.npz
    Clean physical trajectory arrays.
data.npz
    DAEHN-compatible dataset. Its state array is [x, y, lambda], where the
    auxiliary algebraic multiplier lambda is identically zero in the exact
    trajectory. The multiplier will make the later projection system square.
splits.npz
    Train, validation, and test indices.
metadata.json
    Parameters, initial condition, solver settings, and validation metrics.
trajectory.png
    Generated only when --plot is supplied.

Example
-------
python .\\examples\\dae\\10_lotka_volterra_data_generator.py `
    --initial-condition 1.0 1.0 `
    --alpha 1.5 --beta 1.0 --gamma 3.0 --delta 1.0 `
    --horizon 7.0 --samples 2000 `
    --output-dir data\\example_10_lotka_volterra `
    --plot
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LotkaVolterraParameters:
    """Positive parameters of the predator--prey model."""

    alpha: float
    beta: float
    gamma: float
    delta: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; received {value!r}")


def invariant(states: np.ndarray, parameters: LotkaVolterraParameters) -> np.ndarray:
    """Evaluate the conserved Lotka--Volterra quantity row-wise."""

    values = np.asarray(states, dtype=np.float64)
    if values.shape[-1] != 2:
        raise ValueError(f"states must have final dimension 2; received {values.shape}")
    x = values[..., 0]
    y = values[..., 1]
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("the logarithmic invariant requires strictly positive x and y")
    p = parameters
    return p.delta * x - p.gamma * np.log(x) + p.beta * y - p.alpha * np.log(y)


def rhs(
    _time: float,
    state: np.ndarray,
    parameters: LotkaVolterraParameters,
) -> np.ndarray:
    """Lotka--Volterra right-hand side."""

    x, y = np.asarray(state, dtype=np.float64)
    p = parameters
    return np.asarray(
        (
            p.alpha * x - p.beta * x * y,
            -p.gamma * y + p.delta * x * y,
        ),
        dtype=np.float64,
    )


def make_splits(
    count: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic non-overlapping train/validation/test indices."""

    if count < 3:
        raise ValueError("at least three samples are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than one")

    n_train = int(np.floor(train_fraction * count))
    n_validation = int(np.floor(validation_fraction * count))
    n_train = max(1, n_train)
    n_validation = max(1, n_validation)
    if n_train + n_validation >= count:
        n_train = count - 2
        n_validation = 1

    indices = np.arange(count, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
    elif mode != "chronological":
        raise ValueError(f"unsupported split mode: {mode!r}")

    train = np.sort(indices[:n_train])
    validation = np.sort(indices[n_train : n_train + n_validation])
    test = np.sort(indices[n_train + n_validation :])
    return train, validation, test


def save_daehn_data(
    path: Path,
    *,
    time: np.ndarray,
    states: np.ndarray,
    derivatives: np.ndarray,
    parameters: LotkaVolterraParameters,
    invariant0: float,
) -> None:
    """Write the exact array layout consumed by ``daehn.dae.DAEData.load``."""

    count = int(time.size)
    parameter_vector = np.asarray(
        (
            parameters.alpha,
            parameters.beta,
            parameters.gamma,
            parameters.delta,
            invariant0,
        ),
        dtype=np.float64,
    )
    parameter_rows = np.broadcast_to(parameter_vector, (count, 5)).copy()

    # The later DAEHN formulation introduces one algebraic Lagrange multiplier.
    # It is exactly zero on the physical Lotka--Volterra trajectory.
    lambda_invariant = np.zeros((count, 1), dtype=np.float64)
    full_states = np.concatenate((states, lambda_invariant), axis=1)
    empty = np.asarray([], dtype=np.float64)

    np.savez(
        path,
        coordinates=time[:, None],
        parameters=parameter_rows,
        states=full_states,
        derivatives=derivatives,
        initial_coordinates=np.asarray(((time[0],),), dtype=np.float64),
        initial_targets=full_states[:1],
        boundary_coordinates=empty,
        boundary_targets=empty,
        initial_mask=empty,
        boundary_mask=empty,
    )


def save_plot(path: Path, time: np.ndarray, states: np.ndarray, residual: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    """Save a compact trajectory and invariant-drift diagnostic figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), constrained_layout=True)
    axes[0].plot(time, states[:, 0], label="prey x")
    axes[0].plot(time, states[:, 1], label="predator y")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("population")
    axes[0].legend()

    axes[1].plot(states[:, 0], states[:, 1])
    axes[1].scatter(states[0, 0], states[0, 1], marker="o", label="initial state")
    axes[1].set_xlabel("prey x")
    axes[1].set_ylabel("predator y")
    axes[1].legend()

    axes[2].plot(time, residual)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel(r"$V(x,y)-V_0$")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a high-accuracy Lotka--Volterra trajectory and the "
            "corresponding conserved-invariant residual."
        )
    )
    parser.add_argument(
        "--initial-condition",
        nargs=2,
        type=float,
        metavar=("X0", "Y0"),
        default=(1.0, 1.0),
        help="positive initial prey and predator populations",
    )
    parser.add_argument("--alpha", type=float, default=1.5, help="prey growth rate")
    parser.add_argument("--beta", type=float, default=1.0, help="predation coefficient")
    parser.add_argument("--gamma", type=float, default=3.0, help="predator death rate")
    parser.add_argument("--delta", type=float, default=1.0, help="predator conversion coefficient")
    parser.add_argument(
        "--horizon",
        type=float,
        default=7.0,
        help="final physical time; 7.0 matches the paper's training horizon",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="number of uniformly spaced samples including the initial and final points",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--invariant-tolerance",
        type=float,
        default=1.0e-8,
        help="maximum permitted absolute drift in V(x,y)-V0",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=("random", "chronological"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--plot", action="store_true", help="save trajectory.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = LotkaVolterraParameters(
        alpha=float(args.alpha),
        beta=float(args.beta),
        gamma=float(args.gamma),
        delta=float(args.delta),
    )
    parameters.validate()

    initial = np.asarray(args.initial_condition, dtype=np.float64)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("--initial-condition must contain two finite values")
    if np.any(initial <= 0.0):
        raise ValueError("Lotka--Volterra initial populations must be strictly positive")
    if not np.isfinite(args.horizon) or args.horizon <= 0.0:
        raise ValueError("--horizon must be finite and positive")
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("--rtol and --atol must be positive")
    if args.invariant_tolerance <= 0.0:
        raise ValueError("--invariant-tolerance must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    time = np.linspace(0.0, float(args.horizon), int(args.samples), dtype=np.float64)

    solution = solve_ivp(
        fun=lambda t, state: rhs(t, state, parameters),
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
    if states.shape != (args.samples, 2):
        raise RuntimeError(f"unexpected solution shape: {states.shape}")
    if not np.all(np.isfinite(states)):
        raise RuntimeError("the generated trajectory contains non-finite states")
    if np.any(states <= 0.0):
        minimum = float(np.min(states))
        raise RuntimeError(f"the numerical trajectory left the positive domain: min={minimum:.6e}")

    derivatives = np.stack(
        [rhs(t, state, parameters) for t, state in zip(time, states)],
        axis=0,
    )
    invariant_values = invariant(states, parameters)
    invariant0 = float(invariant(initial[None, :], parameters)[0])
    invariant_residual = invariant_values - invariant0
    max_abs_invariant_residual = float(np.max(np.abs(invariant_residual)))
    if max_abs_invariant_residual > float(args.invariant_tolerance):
        raise RuntimeError(
            "invariant drift exceeded tolerance: "
            f"{max_abs_invariant_residual:.6e} > {args.invariant_tolerance:.6e}. "
            "Tighten --rtol/--atol or relax --invariant-tolerance."
        )

    train_indices, validation_indices, test_indices = make_splits(
        int(args.samples),
        float(args.train_fraction),
        float(args.validation_fraction),
        int(args.seed),
        str(args.split_mode),
    )

    lambda_invariant = np.zeros_like(time)
    csv_matrix = np.column_stack(
        (
            time,
            states[:, 0],
            states[:, 1],
            derivatives[:, 0],
            derivatives[:, 1],
            invariant_values,
            invariant_residual,
            lambda_invariant,
        )
    )
    np.savetxt(
        output_dir / "trajectory.csv",
        csv_matrix,
        delimiter=",",
        header=(
            "time,prey_x,predator_y,dprey_dt,dpredator_dt,"
            "invariant,invariant_residual,lambda_invariant"
        ),
        comments="",
        fmt="%.17e",
    )

    np.savez(
        output_dir / "trajectory.npz",
        time=time,
        states=states,
        derivatives=derivatives,
        invariant=invariant_values,
        invariant_residual=invariant_residual,
        invariant0=np.asarray(invariant0, dtype=np.float64),
        parameters=np.asarray(
            (parameters.alpha, parameters.beta, parameters.gamma, parameters.delta),
            dtype=np.float64,
        ),
        initial_condition=initial,
    )
    save_daehn_data(
        output_dir / "data.npz",
        time=time,
        states=states,
        derivatives=derivatives,
        parameters=parameters,
        invariant0=invariant0,
    )
    np.savez(
        output_dir / "splits.npz",
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    metadata = {
        "problem": "lotka_volterra_invariant",
        "equations": {
            "dx_dt": "alpha*x - beta*x*y",
            "dy_dt": "-gamma*y + delta*x*y",
            "invariant": "delta*x - gamma*log(x) + beta*y - alpha*log(y)",
            "algebraic_residual": "invariant - invariant0",
        },
        "parameters": asdict(parameters),
        "initial_condition": {"x0": float(initial[0]), "y0": float(initial[1])},
        "invariant0": invariant0,
        "horizon": float(args.horizon),
        "samples": int(args.samples),
        "time_step": float(time[1] - time[0]),
        "solver": {
            "method": "DOP853",
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "successful": bool(solution.success),
            "function_evaluations": int(solution.nfev),
        },
        "validation": {
            "max_abs_invariant_residual": max_abs_invariant_residual,
            "rms_invariant_residual": float(np.sqrt(np.mean(invariant_residual**2))),
            "minimum_prey_x": float(np.min(states[:, 0])),
            "minimum_predator_y": float(np.min(states[:, 1])),
            "all_values_finite": bool(
                np.all(np.isfinite(states)) and np.all(np.isfinite(derivatives))
            ),
        },
        "splits": {
            "mode": str(args.split_mode),
            "seed": int(args.seed),
            "train_count": int(train_indices.size),
            "validation_count": int(validation_indices.size),
            "test_count": int(test_indices.size),
        },
        "daehn_state_order": ["prey_x", "predator_y", "lambda_invariant"],
        "daehn_parameter_order": ["alpha", "beta", "gamma", "delta", "invariant0"],
        "derivative_order": ["dprey_dt", "dpredator_dt"],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    if args.plot:
        save_plot(Path(__file__).resolve().parent.parent / "results" / "01_lotka_volterra" / "output" / "data_generation" / "trajectory.png", time, states, invariant_residual)

    print("Lotka--Volterra data generation finished.")
    print(f"  output directory: {output_dir}")
    print(f"  samples: {args.samples}")
    print(f"  initial condition: x0={initial[0]:.8g}, y0={initial[1]:.8g}")
    print(
        "  parameters: "
        f"alpha={parameters.alpha:.8g}, beta={parameters.beta:.8g}, "
        f"gamma={parameters.gamma:.8g}, delta={parameters.delta:.8g}"
    )
    print(f"  invariant0: {invariant0:.12e}")
    print(f"  max |V-V0|: {max_abs_invariant_residual:.6e}")
    print(
        "  split counts: "
        f"train={train_indices.size}, validation={validation_indices.size}, "
        f"test={test_indices.size}"
    )
    print("  wrote: trajectory.csv, trajectory.npz, data.npz, splits.npz, metadata.json")
    if args.plot:
        print("  wrote: trajectory.png")


if __name__ == "__main__":
    main()
