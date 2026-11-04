"""Run the CSTR multiple-reaction DAE example from a source checkout."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Direct execution (python examples/<script>.py) places only the
# script directory on sys.path. Add the repository and src roots so both the
# examples package and an uninstalled source checkout of daehn are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from project_paths import find_project_root, result_path

_REPOSITORY_ROOT = find_project_root(Path(__file__).parent)
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from daehn.dae import (
    AlgebraicState,
    Coordinate,
    DAEData,
    DAETrainer,
    DAETrainingConfig,
    DAESystem,
    DerivativeSpec,
    DifferentialState,
    EquationBlock,
    Parameter,
    ScalingConfig,
    TaylorNeighborhood,
    compile_dae,
)
from daehn.opt import ImplicitGradientConfig, ProjectionConfig
from daehn.train.projected_neural import MLPBackbone
import case_study_utils as _case_study_utils
from case_study_utils import (
    WeightedDAETrainer,
    same_horizon_indices,
    run_model_suite,
    set_reproducible_seed,
)

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "cstr.csv"


@dataclass(frozen=True)
class CSTRDataBundle:
    data: DAEData
    all_times: np.ndarray
    all_states: np.ndarray


@dataclass(frozen=True)
class CSTRPeriodicBackbone:
    """MLP with deterministic normalized-time and Fourier input features.

    The feature map changes only the neural representation.  The physical
    coordinate, DAE equations, Taylor reconstruction, and projection problem
    remain unchanged.
    """

    hidden_sizes: tuple[int, ...]
    time_start: float
    time_end: float
    forcing_period: float = 20.0
    harmonics: int = 2
    activation: object = jax.nn.tanh

    def _features(self, coordinates):
        t = jnp.asarray(coordinates)[..., 0]
        span = max(float(self.time_end) - float(self.time_start), 1.0e-12)
        tau = 2.0 * (t - float(self.time_start)) / span - 1.0
        omega = 2.0 * jnp.pi / float(self.forcing_period)
        values = [tau]
        for harmonic in range(1, int(self.harmonics) + 1):
            phase = harmonic * omega * t
            values.extend((jnp.sin(phase), jnp.cos(phase)))
        return jnp.stack(values, axis=-1)

    def init(self, key, input_dim, output_dim, dtype=jnp.float64):
        del input_dim
        feature_dim = 1 + 2 * int(self.harmonics)
        base = MLPBackbone(hidden_sizes=self.hidden_sizes, activation=self.activation)
        return base.init(key, feature_dim, output_dim, dtype=dtype)

    def apply(self, parameters, coordinates):
        base = MLPBackbone(hidden_sizes=self.hidden_sizes, activation=self.activation)
        return base.apply(parameters, self._features(coordinates))


def reaction_rates(states):
    c_a, c_b, c_c, theta = states
    r1 = 0.55 * np.exp(0.25 * theta) * c_a
    r2 = 0.08 * np.exp(0.20 * theta) * c_b
    r3 = 0.05 * np.exp(0.15 * theta) * c_c
    return np.asarray((r1, r2, r3), dtype=np.float64)


def input_profile(t):
    t = np.asarray(t, dtype=np.float64)
    phi = np.where(t >= 20.0, 1.10, 1.0)
    q = np.where(t >= 40.0, 0.02, -0.0134)
    return phi, q


def disturbance_profile(t):
    t = np.asarray(t, dtype=np.float64)
    omega = 2.0 * np.pi / 20.0
    active = t >= 25.0
    wave = np.sin(omega * (t - 25.0))
    c_ai = np.where(active, 1.0 + 0.05 * wave, 1.0)
    theta_i = np.where(active, 0.20 * wave, 0.0)
    return c_ai, theta_i


def cstr_rhs_from_values(states, phi, q, c_ai, theta_i):
    c_a, c_b, c_c, theta = states
    r1, r2, r3 = reaction_rates(states)
    return np.asarray(
        (
            0.2 * phi * (c_ai - c_a) - r1,
            -0.2 * phi * c_b + r1 - r2,
            -0.2 * phi * c_c + r2 - r3,
            0.2 * phi * (theta_i - theta)
            + 0.06 * r1
            + 0.04 * r2
            + 0.02 * r3
            + q,
        ),
        dtype=np.float64,
    )


def cstr_rhs(t, states):
    phi, q = input_profile(t)
    c_ai, theta_i = disturbance_profile(t)
    return cstr_rhs_from_values(states, float(phi), float(q), float(c_ai), float(theta_i))


def generate_trajectory(count: int):
    x0 = np.asarray((0.2667, 0.5238, 0.1676, 0.0), dtype=np.float64)
    times = np.linspace(0.0, 80.0, int(count), dtype=np.float64)
    sol = solve_ivp(
        cstr_rhs,
        (0.0, 80.0),
        x0,
        t_eval=times,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError("CSTR integration failed: " + sol.message)
    states = sol.y.T.astype(np.float64)
    return times, states


def load_csv_trajectory(path: Path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    times = np.asarray(table["time_min"], dtype=np.float64)
    states = np.stack(
        (
            np.asarray(table["c_A"], dtype=np.float64),
            np.asarray(table["c_B"], dtype=np.float64),
            np.asarray(table["c_C"], dtype=np.float64),
            np.asarray(table["theta"], dtype=np.float64),
        ),
        axis=-1,
    )
    return times, states


def _empty(_coords, _states, _derivatives, _parameters):
    return jnp.empty((0,), dtype=jnp.float64)


def make_cstr_system(include_state_bounds: bool = False) -> DAESystem:
    time = Coordinate("time_min", 0.0, 80.0)
    c_a = DifferentialState("c_A")
    c_b = DifferentialState("c_B")
    c_c = DifferentialState("c_C")
    theta = DifferentialState("theta")
    r1 = AlgebraicState("r1")
    r2 = AlgebraicState("r2")
    r3 = AlgebraicState("r3")
    phi = Parameter("phi")
    q = Parameter("q")
    c_ai = Parameter("c_Ai")
    theta_i = Parameter("theta_i")
    derivatives = (
        DerivativeSpec(c_a, (1,)),
        DerivativeSpec(c_b, (1,)),
        DerivativeSpec(c_c, (1,)),
        DerivativeSpec(theta, (1,)),
    )

    def differential(_coords, states, derivs, params):
        ca, cb, cc, th, rr1, rr2, rr3 = states
        ph, heat, inlet_ca, inlet_th = params
        return jnp.asarray(
            (
                derivs[0] - (0.2 * ph * (inlet_ca - ca) - rr1),
                derivs[1] - (-0.2 * ph * cb + rr1 - rr2),
                derivs[2] - (-0.2 * ph * cc + rr2 - rr3),
                derivs[3]
                - (
                    0.2 * ph * (inlet_th - th)
                    + 0.06 * rr1
                    + 0.04 * rr2
                    + 0.02 * rr3
                    + heat
                ),
            )
        )

    def algebraic(_coords, states, _derivs, _params):
        ca, cb, cc, th, rr1, rr2, rr3 = states
        return jnp.asarray(
            (
                rr1 - 0.55 * jnp.exp(0.25 * th) * ca,
                rr2 - 0.08 * jnp.exp(0.20 * th) * cb,
                rr3 - 0.05 * jnp.exp(0.15 * th) * cc,
            )
        )

    if include_state_bounds:

        def inequalities(_coords, states, _derivs, params):
            ca, cb, cc, th, *_rates = states
            ph, heat, _inlet_ca, _inlet_th = params
            return jnp.asarray(
                (
                    -ca,
                    ca - 1.2,
                    -cb,
                    cb - 1.2,
                    -cc,
                    cc - 1.2,
                    -2.0 - th,
                    th - 2.0,
                    0.4 - ph,
                    ph - 2.0,
                    -0.6 - heat,
                    heat - 0.6,
                )
            )

        inequality_block = EquationBlock(
            inequalities,
            12,
            affine_in_projection_variables=False,
            name="state_and_input_bounds",
        )
    else:
        inequality_block = EquationBlock(
            _empty, 0, affine_in_projection_variables=True, name="bounds"
        )

    return DAESystem(
        (time,),
        (c_a, c_b, c_c, theta),
        (r1, r2, r3),
        (phi, q, c_ai, theta_i),
        derivatives,
        EquationBlock(
            differential,
            4,
            affine_in_projection_variables=True,
            name="cstr_differential",
        ),
        EquationBlock(
            algebraic,
            3,
            affine_in_projection_variables=False,
            name="reaction_rates",
        ),
        inequality_block,
        name="cstr_multiple_reaction",
    )


def build_bundle(
    *,
    samples: int,
    delta: float,
    data_source: str,
    seed: int,
    include_state_bounds: bool,
    time_start: float,
    time_end: float,
) -> CSTRDataBundle:
    if data_source not in {"auto", "csv", "generate"}:
        raise ValueError("data_source must be auto, csv, or generate")
    use_csv = data_source == "csv" or (
        data_source == "auto" and CSV_PATH.exists()
    )
    if use_csv:
        times, differential_states = load_csv_trajectory(CSV_PATH)
        if len(times) < samples and data_source == "auto":
            times, differential_states = generate_trajectory(max(samples + 128, 2200))
    else:
        times, differential_states = generate_trajectory(max(samples + 128, 2200))

    phi, heat = input_profile(times)
    c_ai, theta_i = disturbance_profile(times)
    rates = np.stack([reaction_rates(row) for row in differential_states], axis=0)
    states = np.concatenate((differential_states, rates), axis=-1)
    derivatives = np.stack(
        [
            cstr_rhs_from_values(row, ph, qq, inlet_ca, inlet_th)
            for row, ph, qq, inlet_ca, inlet_th in zip(
                differential_states, phi, heat, c_ai, theta_i
            )
        ],
        axis=0,
    )
    parameters = np.stack((phi, heat, c_ai, theta_i), axis=-1)

    margin = max(2.0 * float(delta), 1e-6)
    if not float(time_start) < float(time_end):
        raise ValueError("time_start must be smaller than time_end")
    if float(time_start) < float(times[0]) or float(time_end) > float(times[-1]):
        raise ValueError(
            f"requested learning window [{time_start}, {time_end}] lies outside "
            f"the generated trajectory [{times[0]}, {times[-1]}]"
        )
    keep = (times >= float(time_start) + margin) & (times <= float(time_end) - margin)
    for discontinuity in (20.0, 25.0, 40.0):
        keep &= np.abs(times - discontinuity) > margin
    if include_state_bounds:
        keep &= np.all(differential_states[:, :3] >= 0.0, axis=1)
        keep &= np.all(differential_states[:, :3] <= 1.2, axis=1)
        keep &= (differential_states[:, 3] >= -2.0) & (differential_states[:, 3] <= 2.0)

    valid = np.flatnonzero(keep)
    if valid.size < 2:
        raise ValueError("not enough CSTR samples remain after interior filtering")
    count = min(int(samples), valid.size)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(valid, size=count, replace=False))

    initial_coordinates = np.asarray([[float(time_start)]], dtype=np.float64)
    initial_target = interpolate_states(
        np.asarray([float(time_start)], dtype=np.float64), times, states
    )
    data = DAEData(
        coordinates=times[selected, None],
        parameters=parameters[selected],
        states=states[selected],
        derivatives=derivatives[selected],
        initial_coordinates=initial_coordinates,
        initial_targets=initial_target,
    )
    return CSTRDataBundle(data, times, states)


def split_indices(count, train_fraction, validation_fraction, seed):
    if count < 2:
        raise ValueError("at least two samples are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    if count == 2:
        return order[:1], order[1:], order[1:].copy()
    train_count = max(1, int(round(train_fraction * count)))
    validation_count = max(1, int(round(validation_fraction * count)))
    if train_count + validation_count >= count:
        train_count = max(1, count - 2)
        validation_count = 1
    train = order[:train_count]
    validation = order[train_count : train_count + validation_count]
    test = order[train_count + validation_count :]
    return train, validation, test


def _json_ready(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def interpolate_states(times, all_times, all_states):
    columns = [
        np.interp(times, all_times, all_states[:, index])
        for index in range(all_states.shape[1])
    ]
    return np.stack(columns, axis=-1)


def write_learning_plot(output: Path, histories: dict[str, tuple[dict, ...]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["validation_data_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row["validation_violation"] for row in history], label=label)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MSE")
    axes[1].set_ylabel("Validation violation")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.savefig(output / "all_models_rmse_and_violation.png", dpi=200)
    plt.close(fig)



_CSTR_PLOT_LABELS = (
    r"$C_{A}$",
    r"$C_{B}$",
    r"$C_{C}$",
    r"$\theta$",
    r"$r_{1}$",
    r"$r_{2}$",
    r"$r_{3}$",
)


def _save_cstr_state_comparison_plot(
    output_dir: Path,
    coordinates: np.ndarray,
    exact: np.ndarray,
    predictions,
    state_names,
) -> None:
    """Save the seven CSTR states with a centered 4+3 manuscript layout."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
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
    if exact.ndim != 2 or exact.shape[1] != 7:
        raise ValueError("the CSTR prediction plot expects exactly seven states")

    # Use the caller-provided labels when available, but keep the CSTR-specific
    # mathtext labels as the stable default.
    labels = tuple(state_names)
    if len(labels) != 7:
        labels = _CSTR_PLOT_LABELS

    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams.update({
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

    color_map = {"mlp": "g", "pinn": "b", "daehn": "r"}
    display_map = {
        "mlp": "MLP",
        "pinn": "PINN",
        "daehn": "DAE-HardNet",
    }

    # Eight equal grid columns let the three lower panels occupy [1:3],
    # [3:5], and [5:7], centering them beneath the four upper panels.
    figure = plt.figure(figsize=(28, 12))
    grid = figure.add_gridspec(
        2,
        8,
        left=0.055,
        right=0.99,
        bottom=0.17,
        top=0.97,
        wspace=0.75,
        hspace=0.45,
    )
    spans = (
        (0, 0, 2),
        (0, 2, 4),
        (0, 4, 6),
        (0, 6, 8),
        (1, 1, 3),
        (1, 3, 5),
        (1, 5, 7),
    )
    axes = [
        figure.add_subplot(grid[row, start:stop])
        for row, start, stop in spans
    ]

    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(
            coordinate,
            exact[:, index],
            color="k",
            linewidth=1.2,
            label="Ground Truth",
        )
        for name in ("mlp", "pinn", "daehn"):
            if name not in ordered_predictions:
                continue
            axis.plot(
                coordinate,
                ordered_predictions[name][:, index],
                color=color_map[name],
                linewidth=1.0,
                label=display_map[name],
            )
        axis.set_xlabel(r"$t$", fontsize=24)
        axis.set_ylabel(label, fontsize=24)
        axis.tick_params(axis="both", which="major", labelsize=18)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            fontsize=24,
            ncol=max(1, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / "all_models_prediction_comparison.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "all_models_prediction_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def run_cstr(
    *,
    output_dir: str,
    samples: int = 512,
    time_start: float = 40.5,
    time_end: float = 60.5,
    fourier_harmonics: int = 2,
    delta: float = 1e-2,
    projection: str = "fb",
    lambda_data: float = 1.0,
    projected_learning_rate: float = 5.0e-5,
    data_source: str = "auto",
    discretization: str = "taylor",
    model: str = "daehn",
    include_state_bounds: bool = False,
    noise: float = 0.0,
    seed: int = 0,
    data_seed: int = 31,
    split_seed: int = 31,
    epochs: int = 600,
    pretrain_epochs: int = 500,
    batch_size: int = 128,
    hidden_size: int = 64,
    hidden_layers: int = 3,
    learning_rate: float = 1e-3,
    derivative_weight: float = 1.0,
    pinn_reg_factor: float = 1.0,
    hardnet_reg_factor: float = 1.0,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    print_every: int = 20,
    validation_every: int = 20,
    split_mode: str = "random",
    tolerance: float = 1e-8,
    max_iterations: int = 8,
    backtracking: bool = False,
    taylor_order: int = 1,
    plots: bool = True,
):
    set_reproducible_seed(int(seed))
    if discretization != "taylor":
        raise ValueError("the CSTR example currently supports only Taylor discretization")
    output = result_path(output_dir, __file__)
    output.mkdir(parents=True, exist_ok=True)
    system = make_cstr_system(include_state_bounds=include_state_bounds)
    compiled = compile_dae(
        system,
        discretization=TaylorNeighborhood(
            deltas=(float(delta),),
            order=int(taylor_order),
            radii=(1.0,),
            aggregation="equal_average",
        ),
        scaling=ScalingConfig(),
    )
    bundle = build_bundle(
        samples=samples,
        delta=delta,
        data_source=data_source,
        seed=data_seed,
        include_state_bounds=include_state_bounds,
        time_start=time_start,
        time_end=time_end,
    )
    bundle.data.save(output / "data.npz")
    data = DAEData.load(output / "data.npz")

    if noise > 0.0:
        rng = np.random.default_rng(seed + 17)
        noisy_states = np.asarray(data.states) + rng.normal(scale=noise, size=data.states.shape)
    else:
        noisy_states = np.asarray(data.states)

    if int(epochs) <= 0:
        offsets = np.asarray(compiled.discretization.offsets, dtype=np.float64)[:, 0]
        shifted_times = data.coordinates[:, 0, None] + offsets[None, :]
        neighbor_states = interpolate_states(
            shifted_times.reshape(-1), bundle.all_times, bundle.all_states
        ).reshape((data.coordinates.shape[0], len(offsets), -1))[:, :, :4]
        normalized_derivatives = (
            np.asarray(data.derivatives)
            / np.asarray(compiled.discretization.derivative_physical_scales)
        )
        context = compiled.pack_context(
            jnp.asarray(data.coordinates),
            jnp.asarray(data.parameters),
            jnp.asarray(noisy_states[:, :4]),
            jnp.asarray(neighbor_states),
        )
        target = compiled.build_target(
            jnp.asarray(noisy_states),
            jnp.asarray(normalized_derivatives),
        )
        projection_config = ProjectionConfig(
            tolerance=tolerance,
            max_iterations=int(max_iterations),
            backtracking=bool(backtracking),
            implicit_gradient=ImplicitGradientConfig(
                differentiate_problem_parameters=False,
                policy="residual_checked",
                inexact_tolerance=max(tolerance, 1e-7),
                adjoint_tolerance=max(tolerance, 1e-7),
            ),
        )
        prepared = compiled.prepare_projection(
            batch_size=data.coordinates.shape[0],
            method="fb",
            config=projection_config,
            dtype=jnp.float64,
            backend="cpu",
        )
        projected, info = prepared.with_info(target, context)
        decoded = compiled.decode(context, projected)
        raw_eq = jax.vmap(compiled.raw_equalities)(context, target)
        projected_eq = jax.vmap(compiled.raw_equalities)(context, projected)
        result = {
            "example": 8,
            "system": system.name,
            "projection": prepared.method,
            "discretization": compiled.discretization.name,
            "samples": int(data.coordinates.shape[0]),
            "raw_violation_inf": float(jnp.max(jnp.abs(raw_eq))),
            "projected_violation_inf": float(jnp.max(jnp.abs(projected_eq))),
            "state_mse": float(jnp.mean((decoded["states"] - jnp.asarray(data.states)) ** 2)),
            "converged_fraction": float(jnp.mean(info.forward_converged.astype(jnp.float64))),
            "eligible_fraction": float(jnp.mean(info.gradient_eligible.astype(jnp.float64))),
            "max_iterations": int(jnp.max(info.forward_iterations)),
            "training_epochs": 0,
            "final_training_loss": None,
        }
        result = {key: _json_ready(value) for key, value in result.items()}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    all_valid = np.arange(len(data.coordinates), dtype=np.int64)
    train_index, validation_index, test_index, split_info = same_horizon_indices(
        np.asarray(data.coordinates)[:, 0],
        all_valid,
        train_fraction=float(train_fraction),
        validation_fraction=float(validation_fraction),
        split_mode=str(split_mode),
        seed=int(split_seed),
    )
    print(
        f"  common train/validation/test horizon: "
        f"[{split_info['horizon_start']:.6g}, {split_info['horizon_end']:.6g}] "
        f"({split_info['split_mode']} split)"
    )

    projection_config = ProjectionConfig(
        tolerance=tolerance,
        max_iterations=int(max_iterations),
        backtracking=bool(backtracking),
        implicit_gradient=ImplicitGradientConfig(
            differentiate_problem_parameters=False,
            policy="residual_checked",
            inexact_tolerance=max(tolerance, 1e-7),
            adjoint_tolerance=max(tolerance, 1e-7),
            invalid_batch_policy="use_valid",
        ),
    )
    prepared = compiled.prepare_projection(
        batch_size=batch_size,
        method="fb",
        config=projection_config,
        dtype=jnp.float64,
        backend="cpu",
    )
    backbone = CSTRPeriodicBackbone(
        hidden_sizes=(hidden_size,) * hidden_layers,
        time_start=float(time_start),
        time_end=float(time_end),
        harmonics=int(fourier_harmonics),
        activation=jax.nn.tanh,
    )
    trainer_kwargs = dict(
        compiled=compiled,
        prepared_projection=prepared,
        backbone=backbone,
        initial_coordinates=data.initial_coordinates,
        initial_targets=data.initial_targets,
    )
    train_targets = noisy_states[train_index].copy()
    run_config = {
        "samples": samples,
        "time_start": time_start,
        "time_end": time_end,
        "fourier_harmonics": fourier_harmonics,
        "delta": delta,
        "projection": "fb",
        "data_source": data_source,
        "model": model,
        "noise": noise,
        "seed": seed,
        "data_seed": data_seed,
        "split_seed": split_seed,
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "hidden_layers": hidden_layers,
        "learning_rate": learning_rate,
        "projected_learning_rate": projected_learning_rate,
        "lambda_data": lambda_data,
        "lambda_derivative": derivative_weight,
        "lambda_physics": hardnet_reg_factor,
        "taylor_order": taylor_order,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "backtracking": backtracking,
        "state_bounds": include_state_bounds,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "split_mode": split_mode,
    }
    if plots:
        # Replace only this process-local plotting callback.  The shared
        # runtime and all other case studies remain unchanged.
        _case_study_utils.save_state_comparison_plot = (
            _save_cstr_state_comparison_plot
        )

    return run_model_suite(
        example=8,
        system_name=system.name,
        model=model,
        trainer_cls=WeightedDAETrainer,
        trainer_kwargs=trainer_kwargs,
        train_coordinates=data.coordinates[train_index],
        train_parameters=data.parameters[train_index],
        train_targets=train_targets,
        train_derivative_targets=data.derivatives[train_index],
        validation_coordinates=data.coordinates[validation_index],
        validation_parameters=data.parameters[validation_index],
        validation_targets=data.states[validation_index],
        validation_derivative_targets=data.derivatives[validation_index],
        evaluation_coordinates=data.coordinates[test_index],
        evaluation_parameters=data.parameters[test_index],
        evaluation_targets=data.states[test_index],
        evaluation_derivative_targets=data.derivatives[test_index],
        output_dir=output,
        hidden_sizes=(hidden_size,) * hidden_layers,
        batch_size=batch_size,
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
        learning_rate=learning_rate,
        projected_learning_rate=projected_learning_rate,
        lambda_data=lambda_data,
        lambda_derivative=derivative_weight,
        lambda_physics=hardnet_reg_factor,
        derivative_steps=(delta,),
        seed=seed,
        print_every=print_every,
        validation_every=validation_every,
        state_names=system.state_names,
        split_info=split_info,
        run_config=run_config,
        plot=plots,
        plot_state_names=_CSTR_PLOT_LABELS,
        trajectory_coordinates=data.coordinates[all_valid],
        trajectory_parameters=data.parameters[all_valid],
        trajectory_targets=data.states[all_valid],
        trajectory_derivative_targets=data.derivatives[all_valid],
        case_script=Path(__file__).name,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the CSTR multiple-reaction DAE example with Fischer-"
            "Burmeister projection. Use --model all to compare MLP, PINN, "
            "and DAE-HardNet."
        )
    )
    parser.add_argument("--output-dir", default=str(_REPOSITORY_ROOT / "results" / "06_cstr" / "model" / "new_runs"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--time-start", type=float, default=40.5)
    parser.add_argument("--time-end", type=float, default=60.5)
    parser.add_argument("--fourier-harmonics", type=int, default=2)
    parser.add_argument("--delta", type=float, default=1e-2)
    parser.add_argument(
        "--projection", choices=("fb", "fischer_burmeister"), default="fb"
    )
    parser.add_argument("--discretization", choices=("taylor",), default="taylor")
    parser.add_argument("--data-source", choices=("auto", "csv", "generate"), default="auto")
    parser.add_argument("--model", choices=("daehn", "mlp", "pinn", "all"), default="daehn")
    parser.add_argument("--state-bounds", action="store_true")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="model/training seed")
    parser.add_argument("--data-seed", type=int, default=31, help="fixed CSTR sample-selection seed")
    parser.add_argument("--split-seed", type=int, default=31, help="fixed train/validation/test split seed")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-derivative", "--derivative-weight", dest="derivative_weight", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=1.0)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--projected-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-mode", choices=("random", "chronological"), default="random")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--backtracking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--taylor-order", type=int, default=1)
    parser.add_argument("--plot", "--plots", dest="plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir = result_path(args.output_dir, __file__)

    if args.smoke:
        args.samples = 8
        args.epochs = 2
        args.pretrain_epochs = 1
        args.batch_size = 2
        args.hidden_size = 8
        args.hidden_layers = 1
        args.print_every = 1
        args.validation_every = 1
        args.max_iterations = min(args.max_iterations, 8)
        args.plots = False

    run_cstr(
        output_dir=args.output_dir,
        samples=args.samples,
        time_start=args.time_start,
        time_end=args.time_end,
        fourier_harmonics=args.fourier_harmonics,
        delta=args.delta,
        projection="fb",
        lambda_data=args.lambda_data,
        projected_learning_rate=args.projected_learning_rate,
        data_source=args.data_source,
        discretization=args.discretization,
        model=args.model,
        include_state_bounds=args.state_bounds,
        noise=args.noise,
        seed=args.seed,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        derivative_weight=args.derivative_weight,
        pinn_reg_factor=args.lambda_physics,
        hardnet_reg_factor=args.lambda_physics,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        print_every=args.print_every,
        validation_every=args.validation_every,
        split_mode=args.split_mode,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        backtracking=args.backtracking,
        taylor_order=args.taylor_order,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
