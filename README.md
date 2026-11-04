# DAE-HardNet benchmarks

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Repository checks](https://github.com/SOULS-TAMU/daehn-benchmarks/actions/workflows/checks.yml/badge.svg)](https://github.com/SOULS-TAMU/daehn-benchmarks/actions/workflows/checks.yml)

Train, evaluate and analyze constrained differential–algebraic neural models with
**DAE-HardNet**. This repository contains the seven paper benchmark systems,
a multireaction CSTR control study and a battery voltage-forecasting study.
Each case includes its scripts, input data and preserved trained models.

Built on [DAEHN](https://github.com/SOULS-TAMU/dae-hardnet), shared execution in
[OHAGI](https://github.com/SOULS-TAMU/Ohagi), and projection algorithms in
[PURIN](https://github.com/SOULS-TAMU/Purin).

## Case studies

| Study | Focus | Guide |
| --- | --- | --- |
| Lotka–Volterra | Invariant-constrained trajectories and inverse estimation | [01_lotka_volterra](01_lotka_volterra/README.md) |
| Mass–spring | Energy-constrained trajectories | [02_mass_spring](02_mass_spring/README.md) |
| Two-body | Constrained orbital dynamics | [03_two_body](03_two_body/README.md) |
| Robot arm | Kinematic and differential consistency | [04_robot_arm](04_robot_arm/README.md) |
| Rigid body | Conserved quantities in rotation | [05_rigid_body](05_rigid_body/README.md) |
| Multireaction CSTR | Differential–algebraic process modeling | [06_cstr](06_cstr/README.md) |
| Catalytic oxidation | Parameter and discretization studies | [07_catalytic_oxidation](07_catalytic_oxidation/README.md) |
| CSTR control | Flow surrogate, MPC and closed-loop analysis | [08_cstr_mpc](08_cstr_mpc/README.md) |
| Battery forecasting | Voltage forecasts and causal state estimation | [09_battery_dae](09_battery_dae/README.md) |

## Installation

Use Python **3.10 or newer** and Git. Python 3.11 is a convenient starting point;
JAX availability depends on your Python version, operating system and hardware.
CPU is the default. Clone and create an environment:

```shell
git clone https://github.com/SOULS-TAMU/daehn-benchmarks.git
cd daehn-benchmarks
python -m venv .venv
```

Activate it using the command for your platform:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```shell
# Linux / macOS
source .venv/bin/activate
```

Install DAEHN and its shared libraries, then stage the included trained models:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "from daehn import DAEHardNet; print('DAEHN is ready')"
python prepare_models.py
```

Requirements select explicit Git revisions of DAEHN, OHAGI and PURIN rather than
relying on similarly named PyPI packages. PyTorch is optional and needed only for
the upstream battery `literature/` baselines:

```shell
python -m pip install -r requirements-literature.txt
```

## Saved models and setup

Each `<case>/models/` directory contains the preserved checkpoint tree and its
configuration, manifests and evaluation inputs. Model families, seeds and sweep
branches remain together. [MODEL_MANIFEST.json](MODEL_MANIFEST.json) and
[DATA_MANIFEST.json](DATA_MANIFEST.json) record SHA-256 checksums.

The existing benchmark scripts expect working models under
`results/<case>/model/`. `prepare_models.py` copies the bundled snapshots there,
checks source bytes, and refuses to overwrite differing files. Repeating setup
with identical files is safe. To set up only one case or verify bundled assets:

```shell
python prepare_models.py --case 08_cstr_mpc
python prepare_models.py --check
```

**These benchmark checkpoints are parameter-tree snapshots reconstructed by the
included case loaders.** They are not OHAGI `metadata.json` inference bundles.
Use the analysis/control/forecasting commands below for these saved models.
The library's `DAEHardNet().load('metadata.json')` API is for separately exported
OHAGI-format models. No checkpoint conversion or retraining is performed by setup.

## Paper benchmarks

All seven paper systems expose `train.py`, `analysis.py` and physical-analysis
scripts. From the repository root:

```shell
python 01_lotka_volterra/train.py --help
python 01_lotka_volterra/train.py --model daehn
python 01_lotka_volterra/analysis.py predict --study reference --models daehn --seeds 11
python 01_lotka_volterra/analysis.py compare --study reference --models daehn
python 01_lotka_volterra/analysis.py plot --study reference --models daehn
```

Change the case prefix for other systems. Catalytic oxidation uses
`--study parameter_pairs`; CSTR and catalysis also include `delta_sweep` runs.
Use each script's `--help` for the supported model families and training options.
Saved manifests record the exact configuration of each archived run; rerunning
training defaults alone is not a claim to reproduce the paper's full experiment.

A tiny execution check can use a fresh output folder:

```shell
python 01_lotka_volterra/train.py --model daehn --epochs 2 --pretrain-epochs 1 --batch-size 2 --hidden-size 4 --hidden-layers 1 --train-limit 4 --validation-limit 4 --test-limit 4 --output-dir results/01_lotka_volterra/model/quickstart
```

Short checks validate execution, not converged benchmark accuracy. Preserve the
included reference studies and select fresh names for new experiments.

## CSTR control

```shell
python 08_cstr_mpc/train.py --help
python 08_cstr_mpc/control.py --steps 4 --save
python 08_cstr_mpc/run_ten_hour_study.py --help
```

After model setup, control defaults to the included `flow/daehn/seed=11` model.
The case also retains dataset verification, PINN training and model-comparison
workflows. The longer simulation is optional; details are in the case guide.

## Battery forecasting

```shell
python 09_battery_dae/train.py --help
python 09_battery_dae/forecasting.py --help
python 09_battery_dae/run_case_study.py --tasks forecasting
```

Select another compatible saved run with `--model-dir`. The preserved default is
`smoke_100_full/daehn/seed=42__59620738988d`; longer-trained models and comparison
variants are also included. Forecasts use specified current scenarios. Report
voltage error by horizon and protocol; supplied future current is not a prediction
of unknown demand. See the case guide for formulation, initialization and data details.

## Repository layout

```text
01_lotka_volterra/     training, inverse study, analysis, data/, models/
02_mass_spring/        training, analysis, data/, models/
03_two_body/           training, analysis, data/, models/
04_robot_arm/          training, analysis, data/, models/
05_rigid_body/         training, analysis, data/, models/
06_cstr/               training, analysis, data/, models/
07_catalytic_oxidation/ training, analysis, data/, models/
08_cstr_mpc/           training, control, analysis, _impl/, data/, models/
09_battery_dae/        training, forecasting, _impl/, literature/, data/, models/
_shared/              common paper-case and postprocessing helpers
tests/                repository and artifact integrity checks
results/              staged working models and generated outputs; Git-ignored
```

New training writes to `results/<case>/model/` under its selected study; default
new training uses `new_runs`. Predictions, metrics and figures are written under
`results/<case>/output/`. Only the nine selected cases and repository support
files are tracked; unrelated local folders, environments and caches are ignored.

## Validation and contribution

```shell
python -m unittest discover -s tests -v
```

GitHub checks run repository validation on Windows, Linux and macOS with Python
3.10 and 3.13. They verify syntax, setup behavior, artifact hashes and checkpoint
references; they do not run full numerical training or certify convergence.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Citation, data and license

Copy the BibTeX entries from [CITATION.bib](CITATION.bib); GitHub's citation menu
uses [CITATION.cff](CITATION.cff). Cite the framework and relevant data sources.
[DATA_SOURCES.md](DATA_SOURCES.md) records the battery dataset DOI, authors,
preparation and third-party attribution.

Project code is [Apache-2.0](LICENSE). Included third-party battery measurements,
OCV data and literature baselines retain their source terms; see [NOTICE](NOTICE)
and [DATA_SOURCES.md](DATA_SOURCES.md).
