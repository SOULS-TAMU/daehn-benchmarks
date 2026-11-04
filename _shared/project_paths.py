"""Stable study paths and centralized result output policy."""
from pathlib import Path

def find_project_root(start=None):
    current=Path(start or __file__).resolve()
    if current.is_file(): current=current.parent
    for p in (current,*current.parents):
        if (p/"01_lotka_volterra").is_dir() and (p/"results").is_dir(): return p
    raise RuntimeError("Cannot locate daehn-benchmarks")

def result_path(path, script):
    """Resolve a case result path, including calls from shared postprocessing."""
    script=Path(script).resolve()
    root=find_project_root(script)
    case=script.parent
    path=Path(path)
    if not (case/'train.py').is_file():
        if not path.is_absolute():
            raise ValueError('Shared postprocessing requires an absolute results/<case>/... path')
        relative=path.resolve().relative_to((root/'results').resolve())
        case=root/relative.parts[0]
        if not (case/'train.py').is_file():raise ValueError('Unknown benchmark case: '+case.name)
    allowed=(root/'results'/case.name).resolve()
    resolved=(allowed/path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(allowed):raise ValueError('Results must remain under '+str(allowed))
    relative=resolved.relative_to(allowed)
    if not relative.parts or relative.parts[0] not in {'model','output'}:
        resolved=allowed/'model'/relative
    return resolved
