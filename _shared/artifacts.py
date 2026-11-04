"""Map matching model and output run folders without duplicate artifacts."""
from pathlib import Path
from project_paths import find_project_root

def _branch(path,branch):
    path=Path(path).resolve()
    root=find_project_root(path)/"results"
    relative=path.relative_to(root)
    if len(relative.parts)<2 or relative.parts[1] not in {"model","output"}:
        raise ValueError("Run folders must be inside results/<case>/model or output")
    return root/relative.parts[0]/branch/Path(*relative.parts[2:])

def model_path(path): return _branch(path,"model")
def output_path(path): return _branch(path,"output")
