"""Analyze every checkpoint in a saved suite, then compare matched forecasts."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,subprocess,sys
CASE=_CASE_DIR
BASE=CASE.parent/'results'/CASE.name
TASKS=['reliability','state_estimation','forecasting','discharge_energy','identifiability']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-manifest',type=Path,action='append',default=[])
    p.add_argument('--model-dir',type=Path,action='append',default=[])
    p.add_argument('--study',default='case_study_v2'); p.add_argument('--name',default='baseline_formulation_comparison')
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS); p.add_argument('--compare-only',action='store_true')
    args=p.parse_args()
    if Path(args.study).name!=args.study or Path(args.name).name!=args.name: p.error('study/name must be folder names')
    paths=list(args.model_dir)
    for path in args.suite_manifest:
        manifest=json.loads(path.read_text()); paths += [Path(r['model_dir']) for r in manifest['runs'] if 'model_dir' in r and r['status'] in ('trained','analyzed')]
    paths=list(dict.fromkeys(x.resolve() for x in paths))
    if not paths: p.error('Supply --suite-manifest or --model-dir')
    campaigns=[]
    for directory in paths:
        if not directory.is_relative_to((BASE/'model').resolve()): p.error('Models must be inside this case model branch')
        if not args.compare_only:
            subprocess.run([sys.executable,str(CASE/'run_case_study.py'),'--model-dir',str(directory),'--study',args.study,'--tasks',*args.tasks],cwd=CASE,check=True)
            if set(args.tasks)==set(TASKS): subprocess.run([sys.executable,str((_CASE_DIR / "_impl" / 'validate_case_study.py')),'--model-dir',str(directory),'--study',args.study],cwd=CASE,check=True)
        campaigns.append(BASE/'output'/args.study/directory.relative_to(BASE/'model'))
    if len(campaigns)>1 and (args.compare_only or {'reliability','forecasting','discharge_energy'}.issubset(args.tasks)):
        command=[sys.executable,str((_CASE_DIR / "_impl" / 'compare_formulations.py')),'--name',args.name]
        for directory in campaigns: command+=['--campaign-dir',str(directory)]
        subprocess.run(command,cwd=CASE,check=True)

if __name__=='__main__': main()
