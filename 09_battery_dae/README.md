# Battery voltage forecasting

Physically structured battery voltage prediction, causal state estimation and rolling forecasts under specified current scenarios.

[Install and set up](../README.md#installation) · [All studies](../README.md#case-studies)

## Files

| Script | Purpose |
| --- | --- |
| `analysis.py` | Evaluate and analyze saved runs. |
| `forecasting.py` | Evaluate rolling battery voltage forecasts. |
| `prepare_data.py` | Download/prepare the source battery measurements. |
| `run_case_study.py` | Run selected battery forecasting and physical-analysis tasks. |
| `train.py` | Train a new model using the included data. |
| `train_suite.py` | Train matched battery formulation and baseline campaigns. |

## Data and saved runs

Inputs are included in `data/`. Preserved trained runs are in `models/`
(45 checkpoints, including the available model families/seeds).
Saved study branches: `battery_pinn_tanh_128x3_lr1e4_b64_w128_alg1`, `pinn_smoke_relu_v1`, `pinn_zero_weight_check_v1`, `pinn_smoke_tanh_v1`, `battery_full_tanh_128x3_lr1e4_b64_w128`, `comparison_print_interval_smoke`, `comparison_relu_smoke_v2`, `comparison_smoke_v2`, `smoke_100_full`, `smoke_100`.

Run setup from the repository root before using saved-run analyses:

```shell
python prepare_models.py --case 09_battery_dae
```

This stages verified working copies in `results/09_battery_dae/model/`; analysis writes
under `results/09_battery_dae/output/`. The versioned model snapshots remain unchanged.

## Train

```shell
python 09_battery_dae/train.py --help
python 09_battery_dae/train.py --epochs 1 --steps-per-epoch 1 --batch-size 2 --window-length 3 --hidden-size 4 --hidden-layers 1 --study quickstart
```

This tiny configuration checks execution, not forecast quality. Choose a fresh
`--study` name for subsequent runs. Full training and formulation sweeps remain
available through `train.py` and `train_suite.py`.

## Forecast and analyze

```shell
python 09_battery_dae/forecasting.py --help
python 09_battery_dae/run_case_study.py --tasks forecasting
```

The default saved model is `smoke_100_full/daehn/seed=42__59620738988d`. This is the
preserved default, not a claim that this short-trained checkpoint is the best model.
Longer-trained models and comparisons are also included; select them with
`--model-dir results/09_battery_dae/model/<study>/<variant>/<seed-folder>`.
Additional campaign tasks cover reliability, state estimation, discharge energy
and identifiability. Results remain under the selected ignored output branch.

## Measurement data and interpretation

The CALB LFP measurements, source record and prepared dataset are included.
See [DATA_SOURCES.md](../DATA_SOURCES.md) for the source authors, DOI, license and
OCV-data attribution. `prepare_data.py` downloads/prepares them again if needed;
regeneration is not necessary to use the bundled models.

Known-current forecasts receive a supplied future current scenario. They do not
forecast unknown electricity demand. Real SOC accuracy is unverified; report
voltage error by horizon/protocol and retain the existing domain-exit diagnostics.

The `literature/` scripts reproduce the upstream PyTorch baseline and require
`requirements-literature.txt`. They expect their CSV inputs in the working
directory; run them from a separate scratch directory populated from `data/raw/`,
starting with the static model and then the complete model. The main DAEHN
training/forecasting pathway does not require PyTorch.
