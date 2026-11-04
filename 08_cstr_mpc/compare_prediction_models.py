"""Sequential matched MPC trials. Each predictor uses identical controller code."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,functools,hashlib
import control_experiments as suite
from control import MPCController,DEFAULT_MODEL
from prediction_models import make_predictor
ROOT=suite.ROOT.parent/'model_comparison'
suite.EXPERIMENTS={'00_local_tracking':dict(steps=128,modes=['tracking'],seeds=[101],variants=['default']),**suite.EXPERIMENTS}
original_requirements=suite.requirements
def comparison_requirements(name,k,n):
    req=original_requirements(name,k,n)
    if name=='00_local_tracking':
        req['reference']=[.52,.515,.525,.52][min(3,int(k*4/n))]
        req['band']=(.51,.53)
    return req
suite.requirements=comparison_requirements

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['all']+list(suite.EXPERIMENTS),default='03_feed_disturbances')
    p.add_argument('--predictor',choices=['all','daehn','linear','nonlinear'],default='all')
    p.add_argument('--objective',choices=['all','tracking','economic'],default='tracking')
    p.add_argument('--steps',type=int,help='Short verification run; reschedules phases across this duration')
    p.add_argument('--rk-substeps',type=int,default=8)
    p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.model_dir=args.model_dir.resolve();args.smoke=False
    if args.steps is not None and args.steps<4:p.error('steps must be at least four')
    if args.rk_substeps<1:p.error('rk-substeps must be positive')
    studies=list(suite.EXPERIMENTS) if args.experiment=='all' else [args.experiment]
    kinds=['daehn','linear','nonlinear'] if args.predictor=='all' else [args.predictor]
    modes=['tracking','economic'] if args.objective=='all' else [args.objective]
    original=suite.EXPERIMENTS.copy()
    for kind in kinds:
        print("PREDICTOR",kind,flush=True)
        build=functools.partial(make_predictor,kind=kind,substeps=args.rk_substeps)
        suite.SavedFlowModel=build
        def controller(directory,**kwargs):return MPCController(directory,prediction_model=build(directory),**kwargs)
        suite.MPCController=controller
        label='full' if args.steps is None else 'verification_'+str(args.steps)
        suite.ROOT=ROOT/label/('rk='+str(args.rk_substeps))/kind
        for study in studies:
            spec=dict(original[study]);spec['modes']=['tracking'] if study=='00_local_tracking' else modes
            if args.steps is not None:spec['steps']=args.steps
            suite.EXPERIMENTS[study]=spec;runs=[]
            for mode in spec['modes']:
                for variant in spec['variants']:
                    for seed in spec['seeds']:
                        m=suite.run(study,mode,seed,variant,args)
                        m.update(predictor=kind,rk_substeps=args.rk_substeps,comparison_protocol='Same nonlinear single-shooting SLSQP controller, objective, horizon, bounds, observer and independent plant. Fixed nominal linearization versus nominal nonlinear equations versus saved DAEHN.',verification_only=args.steps is not None)
                        destination=suite.ROOT/study/mode/variant/('seed='+str(seed))
                        (destination/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');runs.append(m)
            (suite.ROOT/study/'metrics.json').write_text(json.dumps(dict(predictor=kind,runs=runs),indent=2),encoding='utf-8')
if __name__=='__main__':main()
