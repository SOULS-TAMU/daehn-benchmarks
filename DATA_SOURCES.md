# Data sources and attribution

## Paper systems and CSTR control

The synthetic reference datasets in cases 01–08 accompany the benchmark problem
definitions. Their included metadata, parameter files and generator scripts
record the system, splits and preparation settings. Saved trained-run manifests
record their actual evaluation inputs and training configuration.

## Battery measurements

Source: [Zenodo record 6323623](https://doi.org/10.5281/zenodo.6323623),
*Code and measurement data – Neural ordinary differential equations for grey-box
modelling of lithium-ion batteries on the basis of an equivalent circuit model*,
Jennifer Brucker, René Behmann, Wolfgang G. Bessler and Rainer Gasper (2022).
Related article: [10.3390/en15072661](https://doi.org/10.3390/en15072661).

The original record and raw measurements are retained under
`09_battery_dae/data/raw/`. The source record identifies an MIT license; the
literature scripts retain their complete MIT notice, copyright Jennifer Brucker
(2021). These third-party terms are separate from this repository's Apache-2.0
license. Original measurements are kept unchanged.

The source specifically attributes the OCV-SOC curve to Yagci et al. (2021),
*Electrical and Structural Characterization of Large-Format Lithium Iron Phosphate
Cells used in Home-Storage Systems*,
[10.1002/ente.20200091](https://doi.org/10.1002/ente.20200091), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The source also attributes its load profile to Weißhar and Bessler (2017),
[10.1016/j.est.2017.10.002](https://doi.org/10.1016/j.est.2017.10.002).

The prepared `data.npz` is derived by the included `prepare_data.py`: it cleans and
orders observations, normalizes protocol time origins, compresses the OCV curve,
computes charge integrals and assigns protocol-level splits. The preparation
metadata records source checksums and transformations. Units are seconds,
amperes, volts and ampere-hours. Cite the source work when reusing these data.
