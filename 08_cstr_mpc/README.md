# CSTR control

A state/control/disturbance-conditioned CSTR flow surrogate used inside a receding-horizon controller.

[Install and set up](../README.md#installation) · [All studies](../README.md#case-studies)

## Files

| Script | Purpose |
| --- | --- |
| `analysis.py` | Evaluate and analyze saved runs. |
| `compare_prediction_models.py` | Compare learned and reference prediction models in control. |
| `control.py` | Run the CSTR closed-loop controller. |
| `generate_data.py` | Generate or validate the reference dataset. |
| `run_ten_hour_study.py` | Run the longer irregular-setpoint controller study. |
| `train.py` | Train a new model using the included data. |
| `train_pinn.py` | Train the PINN flow-model comparison. |

## Data and saved runs

Inputs are included in `data/`. Preserved trained runs are in `models/`
(8 checkpoints, including the available model families/seeds).
Saved study branches: `smoke`, `flow`.

Run setup from the repository root before using saved-run analyses:

```shell
python prepare_models.py --case 08_cstr_mpc
```

This stages verified working copies in `results/08_cstr_mpc/model/`; analysis writes
under `results/08_cstr_mpc/output/`. The versioned model snapshots remain unchanged.

## Train and validate

```shell
python 08_cstr_mpc/train.py --help
python 08_cstr_mpc/train.py --smoke
python 08_cstr_mpc/analysis.py verify --reference-checks 4
```

The smoke run checks execution rather than convergence. New training writes under
`results/08_cstr_mpc/model/new_runs/` and refuses to replace an existing checkpoint.

## Closed-loop control

```shell
python 08_cstr_mpc/control.py --steps 4 --save
python 08_cstr_mpc/run_ten_hour_study.py --help
python 08_cstr_mpc/compare_prediction_models.py --help
```

The controller defaults to the included `flow/daehn/seed=11` checkpoint after setup.
Pass `--model-dir` to select another compatible checkpoint. The flow model anchors
the initial state and reconstructs algebraic reaction rates; differential dynamics
are enforced through training losses. The controller and plant simulation retain
their existing formulations. Short control checks and the full ten-hour study
have very different execution costs.
