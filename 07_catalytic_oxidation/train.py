#!/usr/bin/env python
r"""Generate catalysis DAE data and train DAE-HardNet.

Source benchmark
----------------
The benchmark in SOULS-TAMU/DAE-HardNet defines

    -dy1/dx1 = 0.00588*y2
    y2       = 4.6195*y1*y3
    y2 + y3  = 1

and stores observations in ``catalysis_dae_data.csv`` with columns ``x1,y1``.

Numerically balanced state representation
-----------------------------------------
The physical concentration y1 spans many orders of magnitude.  Internally this
script trains

    s = log(y1),  y2,  y3,

for which the equivalent differential equation is

    ds/dx1 = -a*k*y3,

while the original physical differential residual reported in the logs is

    -exp(s)*ds/dx1 - a*y2.

The two algebraic equations are hard-projected.  The physical DE residual uses
derivatives of the complete NN -> projection -> reconstruction mapping.

Data modes
----------
1. Default: generate a high-accuracy, internally consistent trajectory.  No
   external CSV is needed.
2. Exact repository observations: pass

       --source-csv data/example_17_catalysis_dae/catalysis_dae_data.csv

   The script reads x1,y1 and reconstructs y2,y3 and derivative targets.

PowerShell
----------
    python .\examples\17_catalysis_dae_daehn.py `
        --data-dir data\example_17_catalysis_dae `
        --output-dir results\example_17_catalysis_dae_daehn `
        --plot

Quick pipeline check
--------------------
    python .\examples\17_catalysis_dae_daehn.py --smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
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
    AlgebraicState,
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

INTERNAL_STATE_NAMES = ("log_y1", "y2", "y3")
PHYSICAL_STATE_NAMES = ("y1", "y2", "y3")
DATA_DERIVATIVE_NAMES = ("dlog_y1_dx1",)


def _empty(_coordinates, _states, _derivatives, _parameters):
    return jnp.empty((0,), dtype=jnp.float64)


def _safe_exp(value):
    """Exponentiation over the physically relevant log-concentration range."""

    value = jnp.asarray(value)
    return jnp.exp(jnp.clip(value, -60.0, 5.0))


@dataclass(frozen=True)
class CatalysisDataset:
    data: DAEData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    rate_coefficient: float
    equilibrium_coefficient: float
    horizon: float
    time_step: float
    state_scales: tuple[float, float, float]
    source_kind: str


@dataclass(frozen=True)
class TimeScaledMLPBackbone:
    """MLP on normalized x1 with componentwise internal-state scales."""

    hidden_sizes: tuple[int, ...]
    coordinate_min: float
    coordinate_scale: float
    state_scales: tuple[float, float, float]
    activation: Any = jax.nn.tanh

    def __post_init__(self) -> None:
        if not np.isfinite(self.coordinate_scale) or self.coordinate_scale <= 0.0:
            raise ValueError("coordinate_scale must be finite and positive")
        scales = np.asarray(self.state_scales, dtype=np.float64)
        if scales.shape != (3,):
            raise ValueError("state_scales must contain three entries")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("state_scales must be finite and positive")

    def init(self, key, input_dim, output_dim, dtype=jnp.float64):
        if int(output_dim) != 3:
            raise ValueError(f"catalysis backbone expects 3 outputs, got {output_dim}")
        return MLPBackbone(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).init(key, input_dim, output_dim, dtype)

    def apply(self, parameters, inputs):
        values = jnp.asarray(inputs)
        normalized = (
            values - jnp.asarray(self.coordinate_min, dtype=values.dtype)
        ) / jnp.asarray(self.coordinate_scale, dtype=values.dtype)
        raw = MLPBackbone(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).apply(parameters, normalized)
        return raw * jnp.asarray(self.state_scales, dtype=raw.dtype)


class CatalysisTrainer(WeightedLossTrainerMixin, DAETrainer):
    """Hard-project algebraic equations and train/report the original DAE."""

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


def _load_source_observations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"source CSV does not exist: {path}")
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    names = tuple(table.dtype.names or ())
    if "x1" not in names or "y1" not in names:
        raise ValueError(
            f"source CSV must contain columns x1,y1; received columns {names}"
        )
    coordinate = np.asarray(table["x1"], dtype=np.float64).reshape(-1)
    y1 = np.asarray(table["y1"], dtype=np.float64).reshape(-1)
    if coordinate.size < 20:
        raise ValueError("source CSV must contain at least 20 observations")
    if coordinate.shape != y1.shape:
        raise ValueError("x1 and y1 columns have inconsistent lengths")
    if not np.all(np.isfinite(coordinate)) or not np.all(np.isfinite(y1)):
        raise ValueError("source CSV contains non-finite values")
    if not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("x1 values must be strictly increasing")
    if np.any(y1 <= 0.0):
        raise ValueError("all y1 observations must be strictly positive")
    return coordinate, y1


def _generate_consistent_observations(
    horizon: float,
    samples: int,
    initial_y1: float,
    rate_coefficient: float,
    equilibrium_coefficient: float,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    coordinate = np.linspace(0.0, horizon, samples, dtype=np.float64)
    initial_log = float(np.log(initial_y1))
    a = float(rate_coefficient)
    k = float(equilibrium_coefficient)

    def log_rhs(_x1, log_state):
        y1 = np.exp(np.clip(log_state[0], -60.0, 5.0))
        y3 = 1.0 / (1.0 + k * y1)
        return np.asarray((-a * k * y3,), dtype=np.float64)

    solution = solve_ivp(
        log_rhs,
        (0.0, horizon),
        np.asarray((initial_log,), dtype=np.float64),
        t_eval=coordinate,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"catalysis integration failed: {solution.message}")
    log_y1 = np.asarray(solution.y[0], dtype=np.float64)
    y1 = np.exp(log_y1)
    if not np.all(np.isfinite(y1)) or np.any(y1 <= 0.0):
        raise RuntimeError("generated y1 trajectory is invalid")
    return coordinate, y1, int(solution.nfev)


def _assemble_dataset(
    args: argparse.Namespace,
    coordinate: np.ndarray,
    y1: np.ndarray,
    source_kind: str,
    solver_nfev: int | None,
) -> CatalysisDataset:
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    a = float(args.rate_coefficient)
    k = float(args.equilibrium_coefficient)
    log_y1 = np.log(y1)
    y3 = 1.0 / (1.0 + k * y1)
    y2 = 1.0 - y3
    dlog_y1 = -a * k * y3
    dy1 = y1 * dlog_y1

    algebraic_1 = y2 - k * y1 * y3
    algebraic_2 = y2 + y3 - 1.0
    differential_residual = -dy1 - a * y2

    max_de = float(np.max(np.abs(differential_residual)))
    max_alg = float(
        max(np.max(np.abs(algebraic_1)), np.max(np.abs(algebraic_2)))
    )
    if max_de > 1.0e-11 or max_alg > 1.0e-11:
        raise RuntimeError(
            "assembled catalysis data are inconsistent: "
            f"max DE residual={max_de:.6e}, max algebraic residual={max_alg:.6e}"
        )

    internal_states = np.column_stack((log_y1, y2, y3))
    derivatives = dlog_y1[:, None]
    parameters = np.broadcast_to(np.asarray((a, k)), (len(coordinate), 2)).copy()
    train, validation, test = make_splits(
        len(coordinate),
        float(args.train_fraction),
        float(args.validation_fraction),
        int(args.split_seed),
        str(args.split_mode),
    )
    data = DAEData(
        coordinates=coordinate[:, None],
        parameters=parameters,
        states=internal_states,
        derivatives=derivatives,
        initial_coordinates=np.asarray([[coordinate[0]]], dtype=np.float64),
        initial_targets=internal_states[:1],
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
        x1=coordinate,
        y1=y1,
        log_y1=log_y1,
        y2=y2,
        y3=y3,
        dy1_dx1=dy1,
        dlog_y1_dx1=dlog_y1,
        differential_residual=differential_residual,
        algebraic_residual_1=algebraic_1,
        algebraic_residual_2=algebraic_2,
        parameters=np.asarray((a, k)),
    )
    np.savetxt(
        data_dir / "trajectory.csv",
        np.column_stack(
            (
                coordinate,
                y1,
                log_y1,
                y2,
                y3,
                dy1,
                dlog_y1,
                differential_residual,
                algebraic_1,
                algebraic_2,
            )
        ),
        delimiter=",",
        comments="",
        header=(
            "x1,y1,log_y1,y2,y3,dy1_dx1,dlog_y1_dx1,"
            "differential_residual,algebraic_residual_1,algebraic_residual_2"
        ),
        fmt="%.17e",
    )

    state_scales = (
        max(1.0, 1.1 * float(np.max(np.abs(log_y1)))),
        1.0,
        1.0,
    )
    metadata = {
        "problem": "catalysis_dae",
        "source_kind": source_kind,
        "equations": {
            "physical_differential": "-dy1/dx1 = a*y2",
            "algebraic_1": "y2 = k*y1*y3",
            "algebraic_2": "y2 + y3 = 1",
            "internal_state": "log_y1 = log(y1)",
            "internal_differential": "dlog_y1/dx1 = -a*k*y3",
        },
        "rate_coefficient": a,
        "equilibrium_coefficient": k,
        "initial_y1": float(y1[0]),
        "horizon": float(coordinate[-1] - coordinate[0]),
        "coordinate_min": float(coordinate[0]),
        "coordinate_max": float(coordinate[-1]),
        "samples": int(len(coordinate)),
        "time_step": float(np.median(np.diff(coordinate))),
        "minimum_y1": float(np.min(y1)),
        "maximum_y1": float(np.max(y1)),
        "maximum_differential_residual": max_de,
        "maximum_algebraic_residual": max_alg,
        "state_scales": list(state_scales),
        "solver": {
            "method": None if solver_nfev is None else "DOP853 on log(y1)",
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "function_evaluations": solver_nfev,
        },
        "splits": {
            "mode": str(args.split_mode),
            "seed": int(args.split_seed),
            "split_seed": int(args.split_seed),
            "train_count": int(train.size),
            "validation_count": int(validation.size),
            "test_count": int(test.size),
        },
        "internal_state_order": list(INTERNAL_STATE_NAMES),
        "physical_state_order": list(PHYSICAL_STATE_NAMES),
        "derivative_order": list(DATA_DERIVATIVE_NAMES),
    }
    (data_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Catalysis DAE data generation finished.")
    print(f"  source: {source_kind}")
    print(f"  data directory: {data_dir}")
    print(f"  samples: {len(coordinate)}")
    print(f"  x1 range: [{coordinate[0]:.8g}, {coordinate[-1]:.8g}]")
    print(f"  y1 range: [{np.min(y1):.8e}, {np.max(y1):.8e}]")
    print(f"  max physical DE residual: {max_de:.6e}")
    print(f"  max algebraic residual: {max_alg:.6e}")
    print(
        "  split counts: "
        f"train={train.size}, validation={validation.size}, test={test.size}"
    )

    return CatalysisDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        rate_coefficient=a,
        equilibrium_coefficient=k,
        horizon=float(coordinate[-1] - coordinate[0]),
        time_step=float(np.median(np.diff(coordinate))),
        state_scales=state_scales,
        source_kind=source_kind,
    )


def generate_dataset(args: argparse.Namespace) -> CatalysisDataset:
    if args.source_csv is not None:
        source_path = args.source_csv.resolve()
        coordinate, y1 = _load_source_observations(source_path)
        data_dir = args.data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        destination = data_dir / "catalysis_dae_data.csv"
        if source_path != destination:
            shutil.copy2(source_path, destination)
        return _assemble_dataset(
            args,
            coordinate,
            y1,
            source_kind=f"repository_csv:{source_path}",
            solver_nfev=None,
        )

    coordinate, y1, nfev = _generate_consistent_observations(
        float(args.horizon),
        int(args.samples),
        float(args.initial_y1),
        float(args.rate_coefficient),
        float(args.equilibrium_coefficient),
        float(args.rtol),
        float(args.atol),
    )
    return _assemble_dataset(
        args,
        coordinate,
        y1,
        source_kind="generated_consistent_trajectory",
        solver_nfev=nfev,
    )


def load_dataset(args: argparse.Namespace) -> CatalysisDataset:
    data_dir = args.data_dir.resolve()
    data = DAEData.load(data_dir / "data.npz")
    with np.load(data_dir / "splits.npz") as archive:
        train = np.asarray(archive["train_indices"], dtype=np.int64)
        validation = np.asarray(archive["validation_indices"], dtype=np.int64)
        test = np.asarray(archive["test_indices"], dtype=np.int64)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    states = np.asarray(data.states, dtype=np.float64)
    coordinates = np.asarray(data.coordinates, dtype=np.float64)
    parameters = np.asarray(data.parameters, dtype=np.float64)
    derivatives = np.asarray(data.derivatives, dtype=np.float64)
    count = states.shape[0]
    if states.shape != (count, 3):
        raise ValueError("stored catalysis states must have shape (N, 3)")
    if coordinates.shape != (count, 1):
        raise ValueError("stored coordinates must have shape (N, 1)")
    if parameters.shape != (count, 2):
        raise ValueError("stored parameters must have shape (N, 2)")
    if derivatives.shape != (count, 1):
        raise ValueError("stored derivatives must have shape (N, 1)")
    for name, values in (
        ("states", states),
        ("coordinates", coordinates),
        ("parameters", parameters),
        ("derivatives", derivatives),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.diff(coordinates[:, 0]) > 0.0):
        raise ValueError("stored x1 coordinates must be strictly increasing")
    if not np.allclose(parameters, parameters[:1], atol=1e-12, rtol=0.0):
        raise ValueError("all stored parameter rows must be identical")

    state_scales = tuple(float(value) for value in metadata["state_scales"])
    print("Loaded existing catalysis DAE dataset.")
    print(f"  data directory: {data_dir}")
    print(f"  source: {metadata.get('source_kind', 'unknown')}")
    print(f"  samples: {count}")
    return CatalysisDataset(
        data=data,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        rate_coefficient=float(parameters[0, 0]),
        equilibrium_coefficient=float(parameters[0, 1]),
        horizon=float(coordinates[-1, 0] - coordinates[0, 0]),
        time_step=float(np.median(np.diff(coordinates[:, 0]))),
        state_scales=state_scales,
        source_kind=str(metadata.get("source_kind", "unknown")),
    )


def make_catalysis_system(
    coordinate_min: float,
    coordinate_max: float,
    state_scales: tuple[float, float, float],
    *,
    projection_only: bool,
) -> DAESystem:
    x1 = Coordinate("x1", float(coordinate_min), float(coordinate_max))
    log_y1 = DifferentialState("log_y1", scale=float(state_scales[0]))
    y2 = AlgebraicState("y2", scale=float(state_scales[1]))
    y3 = AlgebraicState("y3", scale=float(state_scales[2]))
    rate = Parameter("rate_coefficient")
    equilibrium = Parameter("equilibrium_coefficient")
    derivatives = (DerivativeSpec(log_y1, (1,)),)

    def differential_equations(_coordinates, states, derivative_values, parameters):
        log_concentration, algebraic_y2, _algebraic_y3 = states
        a, _k = parameters
        dlog_dx = derivative_values[0]
        physical_y1 = _safe_exp(log_concentration)
        # Original repository equation: -dy1/dx1 = a*y2,
        # with dy1/dx1 = y1*dlog(y1)/dx1.
        return jnp.asarray((-physical_y1 * dlog_dx - a * algebraic_y2,))

    def algebraic_equations(_coordinates, states, _derivative_values, parameters):
        log_concentration, algebraic_y2, algebraic_y3 = states
        _a, k = parameters
        physical_y1 = _safe_exp(log_concentration)
        return jnp.asarray(
            (
                algebraic_y2 - k * physical_y1 * algebraic_y3,
                algebraic_y2 + algebraic_y3 - 1.0,
            )
        )

    differential_block = (
        EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="physical_differential_equation_not_in_projection",
        )
        if projection_only
        else EquationBlock(
            differential_equations,
            size=1,
            affine_in_projection_variables=False,
            name="catalysis_physical_differential_equation",
        )
    )

    return DAESystem(
        coordinates=(x1,),
        differential_states=(log_y1,),
        algebraic_states=(y2, y3),
        parameters=(rate, equilibrium),
        derivatives=derivatives,
        differential_equations=differential_block,
        algebraic_equalities=EquationBlock(
            algebraic_equations,
            size=2,
            affine_in_projection_variables=False,
            name="catalysis_algebraic_equations",
        ),
        algebraic_inequalities=EquationBlock(
            _empty,
            size=0,
            affine_in_projection_variables=True,
            name="no_inequalities",
        ),
        name=(
            "catalysis_algebraic_projection"
            if projection_only
            else "catalysis_physical_dae"
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


def decode_internal_states(values: np.ndarray) -> np.ndarray:
    internal = np.asarray(values, dtype=np.float64)
    physical = internal.copy()
    physical[..., 0] = np.exp(np.clip(internal[..., 0], -60.0, 5.0))
    return physical


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or load catalysis DAE data and train DAE-HardNet with "
            "hard algebraic equations."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "results" / "07_catalytic_oxidation" / "model" / "new_runs",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="optional original x1,y1 CSV; if omitted, data are generated",
    )
    parser.add_argument("--a", "--rate-coefficient", dest="rate_coefficient", type=float, default=0.00588)
    parser.add_argument("--k", "--equilibrium-coefficient", dest="equilibrium_coefficient", type=float, default=4.6195)
    parser.add_argument("--initial-y1", type=float, default=1.0)
    parser.add_argument("--horizon", type=float, default=1000.0)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-12)
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
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--derivative-step", type=float, default=1.0)
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
        if args.source_csv is None:
            args.horizon = 100.0
            args.samples = 240
        args.delta = 1.0
        args.derivative_step = 0.5
        args.epochs = 80
        args.pretrain_epochs = 60
        args.batch_size = 16
        args.hidden_size = 32
        args.hidden_layers = 3
        args.print_every = 10
        args.train_limit = 128
        args.validation_limit = 32
        args.test_limit = 32

    for name in ("rate_coefficient", "equilibrium_coefficient", "initial_y1"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.horizon) or args.horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if args.samples < 20:
        raise ValueError("samples must be at least 20")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if args.model in ("daehn", "all") and not 0 < args.pretrain_epochs < args.epochs:
        raise ValueError("pretrain_epochs must lie strictly between 0 and epochs for DAEHN")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.hidden_size < 1 or args.hidden_layers < 1:
        raise ValueError("network dimensions must be positive")
    if args.delta <= 0.0 or args.derivative_step <= 0.0:
        raise ValueError("delta and derivative_step must be positive")
    if args.source_csv is None and args.delta + args.derivative_step >= 0.25 * args.horizon:
        raise ValueError("projection and derivative stencils are too large")
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

    coordinates = np.asarray(data.coordinates[:, 0], dtype=np.float64)
    coordinate_min = float(coordinates[0])
    coordinate_max = float(coordinates[-1])
    margin = float(args.delta) + float(args.derivative_step)

    def keep_interior(indices: np.ndarray, name: str) -> np.ndarray:
        selected = coordinates[indices]
        valid = (
            (selected >= coordinate_min + margin)
            & (selected <= coordinate_max - margin)
        )
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

    physical_system = make_catalysis_system(
        coordinate_min,
        coordinate_max,
        dataset.state_scales,
        projection_only=False,
    )
    projection_system = make_catalysis_system(
        coordinate_min,
        coordinate_max,
        dataset.state_scales,
        projection_only=True,
    )
    compiled = compile_dae(
        projection_system,
        discretization=make_discretization(float(args.delta)),
        scaling=ScalingConfig(residual_mode="none"),
    )

    compiled_derivative_names = tuple(
        f"d{spec.state.name}_dx1" for spec in compiled.discretization.derivatives
    )
    derivative_targets = np.asarray(data.derivatives, dtype=np.float64)
    print(f"  compiled derivative order: {compiled_derivative_names}")
    print(f"  Taylor projection step: {args.delta:.8e}")
    print(f"  full-model derivative step: {args.derivative_step:.8e}")
    print(f"  internal state scales: {dataset.state_scales}")

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
    all_valid_indices = keep_interior(np.arange(len(coordinates), dtype=np.int64), "all")
    train_indices, validation_indices, test_indices, split_info = same_horizon_indices(
        coordinates,
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
        coordinate_min=coordinate_min,
        coordinate_scale=dataset.horizon,
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
        rng = np.random.default_rng(int(args.seed) + 17)
        train_targets[:, 0] += rng.normal(
            scale=float(args.noise), size=train_targets.shape[0]
        )

    run_model_suite(
        example=17,
        system_name=physical_system.name,
        model=args.model,
        trainer_cls=CatalysisTrainer,
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
        state_names=INTERNAL_STATE_NAMES,
        split_info=split_info,
        run_config=vars(args),
        plot=bool(args.plot),
        plot_state_names=PHYSICAL_STATE_NAMES,
        plot_state_transform=decode_internal_states,
        trajectory_coordinates=data.coordinates[all_valid_indices],
        trajectory_parameters=data.parameters[all_valid_indices],
        trajectory_targets=data.states[all_valid_indices],
        trajectory_derivative_targets=derivative_targets[all_valid_indices],
        case_script=Path(__file__).name,
    )


if __name__ == "__main__":
    main()
