"""Postprocess this case; all outputs stay in its central results folder."""
from pathlib import Path
import argparse,sys
CASE=Path(__file__).resolve().parent
sys.path.insert(0,str(CASE.parent/"_shared"))
from project_paths import find_project_root,result_path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("predict","compare","plot","sweep-metrics","sweep-trajectories"))
    parser.add_argument("--study",default="reference",help="saved study folder; use --results-dir to select other runs")
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args,extra=parser.parse_known_args()
    if args.study == 'inverse':
        import subprocess
        actions={'predict':'predict','compare':'report','plot':'plot'}
        if args.action not in actions: parser.error('Inverse study supports predict, compare and plot')
        subprocess.run([sys.executable,str(CASE/'inverse.py'),'--action',actions[args.action],*extra],check=True)
        return
    base=find_project_root(CASE)/"results"/CASE.name
    source=result_path(args.results_dir,__file__) if args.results_dir else base/"model"/args.study
    if not source.resolve().is_relative_to(base.resolve()): parser.error("Study path must remain in this case's results folder")
    output=result_path(args.output_dir,__file__) if args.output_dir else base/"output"/args.study
    modules={"predict":"predict_saved_runs","compare":"compare_saved_runs","plot":"plot_saved_runs","sweep-metrics":"plot_sweep_metrics","sweep-trajectories":"plot_sweep_trajectories"}
    import importlib
    module=importlib.import_module("postprocessing."+modules[args.action])
    sys.argv=[str(Path(module.__file__)),"--results-dir",str(source)]
    if args.action!="predict": sys.argv.extend(["--output-dir",str(output)])
    sys.argv.extend(extra)
    module.main()

if __name__=="__main__": main()
