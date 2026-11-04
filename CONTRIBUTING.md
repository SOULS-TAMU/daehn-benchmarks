# Contributing

Open an issue with the case, command, Python/library versions, and observed
behavior. Keep each change focused and preserve script APIs and numerical
problem definitions unless a numerical change is explicit and documented.

1. Follow the installation and model setup instructions in README.md.
2. Run `python -m unittest discover -s tests -v`.
3. Run a relevant short training, prediction or analysis check. Compare numerical
   outputs whenever model execution changes.
4. Describe validation and expected numerical differences in the pull request.

Do not commit generated results, local environments or unrelated studies. Bundled
`models/` snapshots and `data/` inputs are checksum-protected; do not replace them
in a refactoring change. A new snapshot needs its configuration, source provenance
and hashes. Retain third-party data attribution and notices. Contributions use
Apache-2.0, subject to the existing third-party terms described in DATA_SOURCES.md.
