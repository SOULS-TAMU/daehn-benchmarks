# Catalytic oxidation

Differential–algebraic prediction for the catalytic-oxidation problem, including parameter-pair and discretization-step studies.

[Install and set up](../README.md#installation) · [All studies](../README.md#case-studies)

## Files

| Script | Purpose |
| --- | --- |
| `analysis.py` | Evaluate and analyze saved runs. |
| `physical_case_analysis.py` | Audit saved predictions against case-specific physical quantities. |
| `train.py` | Train a new model using the included data. |

## Data and saved runs

Inputs are included in `data/`. Preserved trained runs are in `models/`
(60 checkpoints, including the available model families/seeds).
Saved study branches: `parameter_pairs`, `delta_sweep`.

Run setup from the repository root before using saved-run analyses:

```shell
python prepare_models.py --case 07_catalytic_oxidation
```

This stages verified working copies in `results/07_catalytic_oxidation/model/`; analysis writes
under `results/07_catalytic_oxidation/output/`. The versioned model snapshots remain unchanged.

## Train

```shell
python 07_catalytic_oxidation/train.py --help
python 07_catalytic_oxidation/train.py --model daehn
```

The full configuration is exposed by `--help`. The paper cases support the
MLP, PINN and DAEHN comparison modes. Keep the default `new_runs` output location
for new training; use a fresh output directory for repeated experiments.

## Analyze saved models

```shell
python 07_catalytic_oxidation/analysis.py predict --study parameter_pairs --models daehn --seeds 11
python 07_catalytic_oxidation/analysis.py compare --study parameter_pairs --models daehn
python 07_catalytic_oxidation/analysis.py plot --study parameter_pairs --models daehn
```

`predict` reconstructs the case and evaluates its stored checkpoint on included
evaluation inputs. `compare` builds tables, while `plot` renders trajectories.
Omit the model filter to compare all available model families. Saved manifests
record each run's actual configuration; CLI training defaults can differ from those
used for the archived paper runs.

## Discretization study

```shell
python 07_catalytic_oxidation/analysis.py sweep-metrics --study delta_sweep
python 07_catalytic_oxidation/analysis.py sweep-trajectories --study delta_sweep
```

The sweep analyses use the included saved runs; no retraining is required.
