# Lotka–Volterra

Invariant-constrained prediction of predator–prey trajectories, with an additional inverse-parameter-estimation workflow.

[Install and set up](../README.md#installation) · [All studies](../README.md#case-studies)

## Files

| Script | Purpose |
| --- | --- |
| `analysis.py` | Evaluate and analyze saved runs. |
| `generate_data.py` | Generate or validate the reference dataset. |
| `inverse.py` | Train or evaluate the inverse-parameter-estimation study. |
| `inverse_physical_analysis.py` | Audit the inverse-study physical behavior. |
| `physical_case_analysis.py` | Audit saved predictions against case-specific physical quantities. |
| `train.py` | Train a new model using the included data. |

## Data and saved runs

Inputs are included in `data/`. Preserved trained runs are in `models/`
(16 checkpoints, including the available model families/seeds).
Saved study branches: `inverse`, `reference`.

Run setup from the repository root before using saved-run analyses:

```shell
python prepare_models.py --case 01_lotka_volterra
```

This stages verified working copies in `results/01_lotka_volterra/model/`; analysis writes
under `results/01_lotka_volterra/output/`. The versioned model snapshots remain unchanged.

## Train

```shell
python 01_lotka_volterra/train.py --help
python 01_lotka_volterra/train.py --model daehn
```

The full configuration is exposed by `--help`. The paper cases support the
MLP, PINN and DAEHN comparison modes. Keep the default `new_runs` output location
for new training; use a fresh output directory for repeated experiments.

## Analyze saved models

```shell
python 01_lotka_volterra/analysis.py predict --study reference --models daehn --seeds 11
python 01_lotka_volterra/analysis.py compare --study reference --models daehn
python 01_lotka_volterra/analysis.py plot --study reference --models daehn
```

`predict` reconstructs the case and evaluates its stored checkpoint on included
evaluation inputs. `compare` builds tables, while `plot` renders trajectories.
Omit the model filter to compare all available model families. Saved manifests
record each run's actual configuration; CLI training defaults can differ from those
used for the archived paper runs.

## Inverse study

```shell
python 01_lotka_volterra/inverse.py --action predict
python 01_lotka_volterra/inverse.py --action report
```

The inverse checkpoint is included separately in `models/inverse/`. Use the
script's `--help` for training and parameter-initialization options.
