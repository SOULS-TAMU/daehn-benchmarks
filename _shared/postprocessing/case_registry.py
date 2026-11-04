"""Case registry and exact inference reconstruction for saved case-study runs."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from project_paths import find_project_root

PROJECT_ROOT = find_project_root(EXAMPLES_ROOT)
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from case_study_utils import make_inference_trainer


@dataclass(frozen=True)
class CaseSpec:
    example: int
    latex_labels: tuple[str, ...]
    physical_state_names: tuple[str, ...]
    transform: str = "identity"
    layout: str = "auto"


CASE_SPECS: dict[int, CaseSpec] = {
    8: CaseSpec(8, (r"$C_{A}$", r"$C_{B}$", r"$C_{C}$", r"$\theta$", r"$r_{1}$", r"$r_{2}$", r"$r_{3}$"), ("c_A", "c_B", "c_C", "theta", "r1", "r2", "r3"), layout="centered_4_3"),
    11: CaseSpec(11, (r"$x$", r"$y$"), ("prey_x", "predator_y")),
    12: CaseSpec(12, (r"$x$", r"$v$"), ("position_x", "velocity_v")),
    13: CaseSpec(13, (r"$q_{1}$", r"$q_{2}$", r"$p_{1}$", r"$p_{2}$"), ("q1", "q2", "p1", "p2")),
    14: CaseSpec(14, (r"$x$", r"$y$", r"$u$", r"$v$"), ("x", "y", "u", "v")),
    15: CaseSpec(15, (r"$\theta_{1}$", r"$\theta_{2}$", r"$\theta_{3}$"), ("theta1", "theta2", "theta3")),
    16: CaseSpec(16, (r"$\omega_{1}$", r"$\omega_{2}$", r"$\omega_{3}$"), ("omega1", "omega2", "omega3")),
    17: CaseSpec(17, (r"$y_{1}$", r"$y_{2}$", r"$y_{3}$"), ("y1", "y2", "y3"), transform="catalysis_physical"),
}


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "checkpoint_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_run_config(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_data(run_dir: Path) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "evaluation_data.npz"
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_case_module(manifest: dict[str, Any]):
    filename = manifest.get("case_script")
    if not filename:
        raise ValueError("checkpoint manifest does not contain case_script")
    study_names = {11: '01_lotka_volterra', 12: '02_mass_spring', 13: '03_two_body', 15: '04_robot_arm', 16: '05_rigid_body', 8: '06_cstr', 17: '07_catalytic_oxidation'}
    path = PROJECT_ROOT / study_names[int(manifest["example"])] / "train.py"
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"daehn_saved_case_{manifest['example']}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _projection_config(module, config: dict[str, Any], *, differentiate_parameters: bool):
    tolerance = float(config.get("tolerance", 1.0e-8))
    return module.ProjectionConfig(
        tolerance=tolerance,
        max_iterations=int(config.get("max_iterations", 20)),
        backtracking=bool(config.get("backtracking", False)),
        implicit_gradient=module.ImplicitGradientConfig(
            differentiate_problem_parameters=differentiate_parameters,
            policy="residual_checked",
            inexact_tolerance=max(tolerance, 1.0e-7),
            adjoint_tolerance=max(tolerance, 1.0e-7),
            invalid_batch_policy="use_valid",
        ),
    )


def _scaled_compiled(module, compiled_unscaled, scales):
    row_scales = jnp.asarray(scales, dtype=jnp.float64)
    raw_equalities = compiled_unscaled.raw_equalities
    return module.replace(
        compiled_unscaled,
        residual_scales=row_scales,
        scaled_equalities=lambda context, variables: raw_equalities(context, variables) / row_scales,
    )


def _backbone(module, manifest: dict[str, Any]):
    info = manifest["backbone"]
    cls = getattr(module, info["class"])
    kwargs = dict(info.get("kwargs", {}))
    kwargs.pop("activation", None)
    for key, value in list(kwargs.items()):
        if key in {"hidden_sizes", "state_scales"} and isinstance(value, list):
            kwargs[key] = tuple(value)
    return cls(**kwargs, activation=jax.nn.tanh)


def build_inference_trainer(run_dir: Path):
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    config = read_run_config(run_dir)
    data = load_evaluation_data(run_dir)
    module = load_case_module(manifest)
    example = int(manifest["example"])
    batch_size = int(manifest.get("batch_size", config.get("batch_size", 32)))
    projected = bool(manifest.get("projected", manifest.get("model") == "daehn"))
    physical_parameters = np.asarray(data["physical_parameters"], dtype=np.float64)
    backbone = _backbone(module, manifest)
    hidden_sizes = tuple(int(v) for v in manifest["hidden_sizes"])
    derivative_steps = tuple(float(v) for v in manifest["derivative_steps"])

    if example == 8:
        system = module.make_cstr_system(include_state_bounds=bool(config.get("state_bounds", False)))
        compiled = module.compile_dae(
            system,
            discretization=module.TaylorNeighborhood(
                deltas=(float(config["delta"]),),
                order=int(config.get("taylor_order", 1)),
                radii=(1.0,),
                aggregation="equal_average",
            ),
            scaling=module.ScalingConfig(),
        )
        prepared = compiled.prepare_projection(
            batch_size=batch_size,
            method="fb",
            config=_projection_config(module, config, differentiate_parameters=False),
            dtype=jnp.float64,
            backend="cpu",
        )
        trainer_cls = module.WeightedDAETrainer
        trainer_kwargs = dict(
            compiled=compiled,
            prepared_projection=prepared,
            backbone=backbone,
            initial_coordinates=data["initial_coordinates"],
            initial_targets=data["initial_targets"],
        )

    elif example == 11:
        horizon = float(manifest["backbone"]["kwargs"]["time_scale"])
        physical_system = module.make_lotka_volterra_system(horizon, projection_only=False)
        projection_system = module.make_lotka_volterra_system(horizon, projection_only=True)
        unscaled = module.compile_dae(
            projection_system,
            discretization=module.make_discretization(float(config["delta"])),
            scaling=module.ScalingConfig(residual_mode="none"),
        )
        compiled = _scaled_compiled(module, unscaled, (module.INVARIANT_RESIDUAL_SCALE,))
        prepared = compiled.prepare_projection(
            batch_size=batch_size,
            method="fb",
            config=_projection_config(module, config, differentiate_parameters=True),
            dtype=jnp.float64,
            backend="cpu",
        )
        trainer_cls = module.LotkaVolterraTrainer
        trainer_kwargs = dict(
            physical_system=physical_system,
            compiled=compiled,
            prepared_projection=prepared,
            backbone=backbone,
            initial_coordinates=data["initial_coordinates"],
            initial_targets=data["initial_targets"],
        )

    elif example == 12:
        kwargs = manifest["backbone"]["kwargs"]
        horizon = float(kwargs["time_scale"])
        state_scale = float(kwargs["state_scale"])
        physical_system = module.make_mass_spring_system(horizon, state_scale, projection_only=False)
        projection_system = module.make_mass_spring_system(horizon, state_scale, projection_only=True)
        unscaled = module.compile_dae(projection_system, discretization=module.make_discretization(float(config["delta"])), scaling=module.ScalingConfig(residual_mode="none"))
        energy_scale = max(abs(float(physical_parameters[0, 0])), 1.0e-3)
        compiled = _scaled_compiled(module, unscaled, (energy_scale,))
        prepared = compiled.prepare_projection(batch_size=batch_size, method="fb", config=_projection_config(module, config, differentiate_parameters=True), dtype=jnp.float64, backend="cpu")
        trainer_cls = module.MassSpringTrainer
        trainer_kwargs = dict(physical_system=physical_system, compiled=compiled, prepared_projection=prepared, backbone=backbone, initial_coordinates=data["initial_coordinates"], initial_targets=data["initial_targets"])

    elif example == 13:
        kwargs = manifest["backbone"]["kwargs"]
        horizon = float(kwargs["time_scale"])
        state_scales = tuple(float(v) for v in kwargs["state_scales"])
        positions = np.asarray(data["targets"], dtype=np.float64)[:, :2]
        minimum_radius = float(np.min(np.linalg.norm(positions, axis=1)))
        radius_floor = max(0.25 * minimum_radius, 1.0e-2)
        physical_system = module.make_two_body_system(horizon, state_scales, projection_only=False, radius_floor=radius_floor)
        projection_system = module.make_two_body_system(horizon, state_scales, projection_only=True, radius_floor=radius_floor)
        unscaled = module.compile_dae(projection_system, discretization=module.make_discretization(float(config["delta"])), scaling=module.ScalingConfig(residual_mode="none"))
        momentum_scale = max(abs(float(physical_parameters[0, 1])), 1.0e-3)
        compiled = _scaled_compiled(module, unscaled, (momentum_scale,))
        prepared = compiled.prepare_projection(batch_size=batch_size, method="fb", config=_projection_config(module, config, differentiate_parameters=True), dtype=jnp.float64, backend="cpu")
        trainer_cls = module.TwoBodyTrainer
        trainer_kwargs = dict(physical_system=physical_system, compiled=compiled, prepared_projection=prepared, backbone=backbone, initial_coordinates=data["initial_coordinates"], initial_targets=data["initial_targets"])

    elif example == 14:
        kwargs = manifest["backbone"]["kwargs"]
        horizon = float(kwargs["time_scale"])
        state_scales = tuple(float(v) for v in kwargs["state_scales"])
        physical_system = module.make_nonlinear_spring_system(horizon, state_scales, projection_only=False)
        projection_system = module.make_nonlinear_spring_system(horizon, state_scales, projection_only=True)
        unscaled = module.compile_dae(projection_system, discretization=module.make_discretization(float(config["delta"])), scaling=module.ScalingConfig(residual_mode="none"))
        compiled = _scaled_compiled(module, unscaled, (max(abs(float(physical_parameters[0, 0])), 1.0e-3), max(abs(float(physical_parameters[0, 1])), 1.0e-3)))
        prepared = compiled.prepare_projection(batch_size=batch_size, method="fb", config=_projection_config(module, config, differentiate_parameters=True), dtype=jnp.float64, backend="cpu")
        trainer_cls = module.NonlinearSpringTrainer
        trainer_kwargs = dict(physical_system=physical_system, compiled=compiled, prepared_projection=prepared, backbone=backbone, initial_coordinates=data["initial_coordinates"], initial_targets=data["initial_targets"])

    elif example == 15:
        kwargs = manifest["backbone"]["kwargs"]
        horizon = float(kwargs["time_scale"])
        state_scales = tuple(float(v) for v in kwargs["state_scales"])
        physical_system = module.make_robot_arm_system(horizon, state_scales, projection_only=False)
        projection_system = module.make_robot_arm_system(horizon, state_scales, projection_only=True)
        unscaled = module.compile_dae(projection_system, discretization=module.make_discretization(float(config["delta"])), scaling=module.ScalingConfig(residual_mode="none"))
        compiled = _scaled_compiled(module, unscaled, (1.0, 1.0))
        prepared = compiled.prepare_projection(batch_size=batch_size, method="fb", config=_projection_config(module, config, differentiate_parameters=True), dtype=jnp.float64, backend="cpu")
        trainer_cls = module.RobotArmTrainer
        trainer_kwargs = dict(physical_system=physical_system, compiled=compiled, prepared_projection=prepared, backbone=backbone, initial_coordinates=data["initial_coordinates"], initial_targets=data["initial_targets"])

    elif example == 16:
        kwargs = manifest["backbone"]["kwargs"]
        horizon = float(kwargs["time_scale"])
        state_scales = tuple(float(v) for v in kwargs["state_scales"])
        physical_system = module.make_rigid_body_system(horizon, state_scales, projection_only=False)
        projection_system = module.make_rigid_body_system(horizon, state_scales, projection_only=True)
        unscaled = module.compile_dae(projection_system, discretization=module.make_discretization(float(config["delta"])), scaling=module.ScalingConfig(residual_mode="none"))
        compiled = _scaled_compiled(module, unscaled, (max(abs(float(physical_parameters[0, 3])), 1.0e-3), max(abs(float(physical_parameters[0, 4])), 1.0e-3)))
        prepared = compiled.prepare_projection(batch_size=batch_size, method="fb", config=_projection_config(module, config, differentiate_parameters=True), dtype=jnp.float64, backend="cpu")
        trainer_cls = module.RigidBodyTrainer
        trainer_kwargs = dict(physical_system=physical_system, compiled=compiled, prepared_projection=prepared, backbone=backbone, initial_coordinates=data["initial_coordinates"], initial_targets=data["initial_targets"])

    elif example == 17:
        kwargs = manifest["backbone"]["kwargs"]
        coordinate_min = float(kwargs["coordinate_min"])
        coordinate_max = coordinate_min + float(kwargs["coordinate_scale"])
        state_scales = tuple(float(v) for v in kwargs["state_scales"])
        physical_system = module.make_catalysis_system(coordinate_min, coordinate_max, state_scales, projection_only=False)
        projection_system = module.make_catalysis_system(coordinate_min, coordinate_max, state_scales, projection_only=True)
        compiled = module.compile_dae(projection_system, discretization=module.make_discretization(float(config["delta"])), scaling=module.ScalingConfig(residual_mode="none"))
        prepared = compiled.prepare_projection(batch_size=batch_size, method="fb", config=_projection_config(module, config, differentiate_parameters=True), dtype=jnp.float64, backend="cpu")
        trainer_cls = module.CatalysisTrainer
        trainer_kwargs = dict(physical_system=physical_system, compiled=compiled, prepared_projection=prepared, backbone=backbone, initial_coordinates=data["initial_coordinates"], initial_targets=data["initial_targets"])

    else:
        raise KeyError(f"no inference builder registered for example {example}")

    trainer = make_inference_trainer(
        trainer_cls,
        trainer_kwargs,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        derivative_steps=derivative_steps,
        seed=int(manifest["seed"]),
        lambda_data=float(config.get("lambda_data", 1.0)),
        lambda_derivative=float(config.get("lambda_derivative", 1.0)),
        lambda_physics=float(config.get("lambda_physics", 1.0)),
    )
    return trainer, projected, manifest, config, data, module


def transform_states(example: int, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if int(example) == 17:
        result = np.empty_like(array)
        result[..., 0] = np.exp(array[..., 0])
        result[..., 1:] = array[..., 1:]
        return result
    return array
