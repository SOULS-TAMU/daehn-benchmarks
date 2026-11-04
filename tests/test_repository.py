"""Repository and preservation checks using only the Python standard library."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_models', ROOT / 'prepare_models.py')
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class RepositoryTests(unittest.TestCase):
    def test_model_and_dataset_checksums(self):
        for name in ('MODEL_MANIFEST.json', 'DATA_MANIFEST.json'):
            for item in json.loads((ROOT / name).read_text())['files']:
                with self.subTest(path=item['path']):
                    path = (ROOT / item['path']).resolve()
                    self.assertTrue(path.is_relative_to(ROOT))
                    self.assertEqual(SETUP.sha256(path), item['sha256'])

    def test_checkpoint_sidecars_and_hashes(self):
        for case in SETUP.CASES:
            checkpoints = list((ROOT / case / 'models').rglob('parameter_tree.pkl'))
            self.assertTrue(checkpoints, case)
            for checkpoint in checkpoints:
                with self.subTest(path=str(checkpoint.relative_to(ROOT))):
                    folder = checkpoint.parent
                    manifest = json.loads((folder / 'checkpoint_manifest.json').read_text())
                    self.assertTrue((folder / 'run_config.json').is_file())
                    self.assertEqual(SETUP.sha256(checkpoint), manifest['checkpoint_sha256'])
                    for file_key, hash_key in [('evaluation_data_file', 'evaluation_data_sha256'),
                                               ('trajectory_data_file', 'trajectory_data_sha256')]:
                        if file_key in manifest:
                            artifact = folder / manifest[file_key]
                            self.assertTrue(artifact.is_file())
                            # Run manifests hash logical array content, not NPZ bytes.
                            # Byte integrity is checked by MODEL_MANIFEST.json above.

    def test_case_structure_and_syntax(self):
        for case in SETUP.CASES:
            self.assertTrue((ROOT / case / 'README.md').is_file())
            self.assertTrue((ROOT / case / 'train.py').is_file())
            self.assertTrue((ROOT / case / 'data/data.npz').is_file())
        paths = [*ROOT.glob('*.py')]
        for folder in (*SETUP.CASES, '_shared', 'tests'):
            paths.extend((ROOT / folder).rglob('*.py'))
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 10))

    def test_setup_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = SETUP.CASES[0]
            relative = Path(case) / 'models/reference/daehn/seed=11/parameter_tree.pkl'
            source = root / relative
            source.parent.mkdir(parents=True)
            source.write_bytes(b'preserved-checkpoint')
            working = Path('results') / case / 'model/reference/daehn/seed=11/parameter_tree.pkl'
            record = {'path': relative.as_posix(), 'working_path': working.as_posix(),
                      'sha256': SETUP.sha256(source)}
            (root / 'MODEL_MANIFEST.json').write_text(json.dumps({'files': [record]}))
            self.assertEqual(SETUP.prepare(root)['copied_files'], 1)
            self.assertEqual((root / working).read_bytes(), source.read_bytes())
            self.assertEqual(SETUP.prepare(root)['existing_identical_files'], 1)
            (root / working).write_bytes(b'new-training-result')
            with self.assertRaises(FileExistsError):
                SETUP.prepare(root)
            self.assertEqual((root / working).read_bytes(), b'new-training-result')
            self.assertEqual(source.read_bytes(), b'preserved-checkpoint')
            source.write_bytes(b'changed-source')
            with self.assertRaises(ValueError):
                SETUP.prepare(root, check_only=True)


if __name__ == '__main__':
    unittest.main()
