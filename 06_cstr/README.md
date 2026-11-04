# Multireaction CSTR benchmark

Constrained differential–algebraic modeling of multireaction CSTR states and reaction rates.

[Install and set up](../README.md#installation) · [All studies](../README.md#case-studies)

## Files

| Script | Purpose |
| --- | --- |
| `analysis.py` | Evaluate and analyze saved runs. |
| `physical_case_analysis.py` | Audit saved predictions against case-specific physical quantities. |
| `train.py` | Train a new model using the included data. |

## Data and saved runs

Inputs are included in `data/`. Preserved trained runs are in `models/`
(30 checkpoints, including the available model families/seeds).
Saved study branches: `reference`, `delta_sweep`.

Run setup from the repository root before using saved-run analyses:

```shell
python prepare_models.py --case 06_cstr
```

This stages verified working copies in `results/06_cstr/model/`; analysis writes
under `results/06_cstr/output/`. The versioned model snapshots remain unchanged.

## Train

```shell
python 06_cstr/train.py --help
python 06_cstr/train.py --model daehn
```

The full configuration is exposed by `--help`. The paper cases support the
MLP, PINN and DAEHN comparison modes. Keep the default `new_runs` output location
for new training; use a fresh output directory for repeated experiments.

## Analyze saved models

```shell
python 06_cstr/analysis.py predict --study reference --models daehn --seeds 11
python 06_cstr/analysis.py compare --study reference --models daehn
python 06_cstr/analysis.py plot --study reference --models daehn
```

`predict` reconstructs the case and evaluates its stored checkpoint on included
evaluation inputs. `compare` builds tables, while `plot` renders trajectories.
Omit the model filter to compare all available model families. Saved manifests
record each run's actual configuration; CLI training defaults can differ from those
used for the archived paper runs.

## Discretization study

```shell
python 06_cstr/analysis.py sweep-metrics --study delta_sweep
python 06_cstr/analysis.py sweep-trajectories --study delta_sweep
```

The sweep analyses use the included saved runs; no retraining is required.
