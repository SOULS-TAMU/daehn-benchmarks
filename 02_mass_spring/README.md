# Mass–spring system

Trajectory prediction constrained by the conserved mechanical energy.

[Install and set up](../README.md#installation) · [All studies](../README.md#case-studies)

## Files

| Script | Purpose |
| --- | --- |
| `analysis.py` | Evaluate and analyze saved runs. |
| `physical_case_analysis.py` | Audit saved predictions against case-specific physical quantities. |
| `train.py` | Train a new model using the included data. |

## Data and saved runs

Inputs are included in `data/`. Preserved trained runs are in `models/`
(15 checkpoints, including the available model families/seeds).
Saved study branches: `reference`.

Run setup from the repository root before using saved-run analyses:

```shell
python prepare_models.py --case 02_mass_spring
```

This stages verified working copies in `results/02_mass_spring/model/`; analysis writes
under `results/02_mass_spring/output/`. The versioned model snapshots remain unchanged.

## Train

```shell
python 02_mass_spring/train.py --help
python 02_mass_spring/train.py --model daehn
```

The full configuration is exposed by `--help`. The paper cases support the
MLP, PINN and DAEHN comparison modes. Keep the default `new_runs` output location
for new training; use a fresh output directory for repeated experiments.

## Analyze saved models

```shell
python 02_mass_spring/analysis.py predict --study reference --models daehn --seeds 11
python 02_mass_spring/analysis.py compare --study reference --models daehn
python 02_mass_spring/analysis.py plot --study reference --models daehn
```

`predict` reconstructs the case and evaluates its stored checkpoint on included
evaluation inputs. `compare` builds tables, while `plot` renders trajectories.
Omit the model filter to compare all available model families. Saved manifests
record each run's actual configuration; CLI training defaults can differ from those
used for the archived paper runs.
