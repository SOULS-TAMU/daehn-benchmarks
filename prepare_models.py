"""Stage verified saved models without changing benchmark script APIs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CASES = (
    '01_lotka_volterra', '02_mass_spring', '03_two_body', '04_robot_arm',
    '05_rigid_body', '06_cstr', '07_catalytic_oxidation', '08_cstr_mpc',
    '09_battery_dae',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: Path = ROOT, cases=None, *, check_only=False) -> dict:
    """Validate all selected sources before copying; never overwrite a conflict."""
    root = Path(root).resolve()
    selected = set(CASES if cases is None else cases)
    unknown = selected.difference(CASES)
    if unknown:
        raise ValueError('Unknown cases: ' + ', '.join(sorted(unknown)))
    manifest = json.loads((root / 'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
    entries = [item for item in manifest['files'] if Path(item['path']).parts[0] in selected]
    pending = []
    unchanged = 0
    for item in entries:
        relative = Path(item['path'])
        case = relative.parts[0]
        source = (root / relative).resolve()
        target = (root / item['working_path']).resolve()
        if not source.is_relative_to((root / case / 'models').resolve()):
            raise ValueError('Source must remain inside its case models directory')
        expected_target = (root / 'results' / case / 'model' / Path(*relative.parts[2:])).resolve()
        if target != expected_target or not target.is_relative_to((root / 'results').resolve()):
            raise ValueError('Invalid working model path')
        if sha256(source) != item['sha256']:
            raise ValueError('Bundled model hash mismatch: ' + item['path'])
        if check_only:
            continue
        if target.exists():
            if sha256(target) != item['sha256']:
                raise FileExistsError('Refusing to overwrite differing working model: ' + str(target))
            unchanged += 1
        else:
            pending.append((source, target, item['sha256']))
    for source, target, expected in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects a file created after preflight.
        with source.open('rb') as original, target.open('xb') as destination:
            shutil.copyfileobj(original, destination)
        if sha256(target) != expected:
            raise ValueError('Working-copy hash mismatch: ' + str(target))
    return {'verified_files': len(entries), 'copied_files': len(pending),
            'existing_identical_files': unchanged, 'check_only': check_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', dest='cases', choices=CASES, action='append',
                        help='Set up only this case; may be repeated')
    parser.add_argument('--check', action='store_true', help='Verify bundled models without copying')
    args = parser.parse_args()
    try:
        report = prepare(cases=args.cases, check_only=args.check)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
