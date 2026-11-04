# Preserved trained models

Saved artifacts retain their original bytes and relative run layout. Each
`parameter_tree.pkl` belongs with its `checkpoint_manifest.json`, `run_config.json`
and any accompanying evaluation inputs. Use this repository's case loader.
These benchmark checkpoints are not OHAGI `metadata.json` export bundles.

From the repository root, run `python prepare_models.py` once to stage working
copies under ignored `results/<case>/model/` paths expected by the existing scripts.
The command validates source hashes and refuses to overwrite differing files.
The versioned snapshots in this directory stay unchanged.
