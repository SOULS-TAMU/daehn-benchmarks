"""Ten-hour DAEHN MPC replay: exactly five unannounced, irregular setpoint changes."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json
import numpy as np
import control_experiments as suite
from control import MPCController,DEFAULT_MODEL

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); args.model_dir=args.model_dir.resolve(); args.smoke=False
    name='08_ten_hour_irregular_setpoints'
    boundaries=np.array([0.,90.,135.,280.,360.,505.,600.])
    references=np.array([.490,.482,.498,.485,.496,.481])
    suite.EXPERIMENTS[name]=dict(steps=2400,modes=['tracking'],seeds=[101],variants=['default'])
    folder=suite.ROOT/name/'tracking/default/seed=101'
    if folder.exists() and not args.resume:raise FileExistsError('Use --resume to reuse a finished run or choose a fresh output folder')
    def requirements(study,k,n):
        assert study==name and n==2400
        minute=k*.25; j=int(np.searchsorted(boundaries[1:],minute,side='right'))
        return dict(reference=float(references[j]),band=(.48,.50),disturbance=np.array([1.,0.]),heat_weight=.05,production_weight=1.,minimum_production=0.)
    suite.requirements=requirements
    controller=MPCController(args.model_dir)
    print('Checking steady targets for all six reference levels...',flush=True)
    feasible=[]
    for ref in references:
        x,u=controller.steady_target(np.array([1.,0.]),float(ref))
        feasible.append({'reference':float(ref),'state':x.tolist(),'control':u.tolist()})
        print('FEASIBLE',float(ref),flush=True)
    # Clear preflight target history; the actual run begins with the ordinary controller.
    controller._steady_target=None; controller._target_switches=0
    suite.MPCController=lambda *unused,**kwargs:controller
    result=suite.run(name,'tracking',101,'default',args)
    result.update(duration_requested_hours=10.,setpoint_change_times_minutes=boundaries[1:-1].tolist(),setpoint_levels=references.tolist(),schedule_preview=False,predictor_role='DAEHN is the MPC predictor; full state measured, no EKF',steady_target_preflight=feasible)
    (folder/'metrics.json').write_text(json.dumps(result,indent=2))
    if result['status']!='completed':raise RuntimeError('Ten-hour run did not finish: '+str(result['stop_reason']))
    with np.load(folder/'trajectory_predictions.npz') as z:
        assert len(z['controls'])==2400
        changes=np.flatnonzero(np.diff(z['references'])!=0)+1
        np.testing.assert_array_equal(changes*.25,boundaries[1:-1])
    print('Verified full ten-hour run and exactly five setpoint changes.',flush=True)

if __name__=='__main__':main()
