"""Sequential, reproducible nonlinear CSTR control-validation experiments.

All controller predictions use the saved DAEHN. Modified equations are used
only by the independent simulation plant, never revealed to the controller.
"""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse, hashlib, json, time, tempfile, gc
import numpy as np
from scipy.integrate import solve_ivp
from control import MPCController, DEFAULT_MODEL
from surrogate import SavedFlowModel
from model import integrate_interval, reaction_rates, state_constraint_values

CASE=_CASE_DIR
ROOT=CASE.parent/'results'/CASE.name/'output'/'experiments'
EXPERIMENTS={
 '01_long_term_requirements':dict(steps=960,modes=['fixed','tracking','economic','lqr'],seeds=[101],variants=['default']),
 '02_changing_economic_priorities':dict(steps=320,modes=['fixed','tracking','economic'],seeds=[101],variants=['default']),
 '03_feed_disturbances':dict(steps=240,modes=['fixed','tracking','economic'],seeds=[101],variants=['default']),
 '04_initial_states':dict(steps=96,modes=['tracking','economic'],seeds=[101],variants=['nominal','low_inventory','near_boundary']),
 '05_noise_and_delay':dict(steps=128,modes=['tracking','economic'],seeds=[101,202,303],variants=['default']),
 '06_model_mismatch':dict(steps=160,modes=['tracking','economic'],seeds=[101],variants=['faster','slower']),
 '07_infeasible_requests':dict(steps=64,modes=['tracking','economic'],seeds=[101],variants=['unreachable','outside_bounds']),
}

def requirements(name,k,n):
    phase=min(5,int(k*6/n)) if name.startswith('01') else min(3,int(k*4/n))
    ref=.49;band=(.48,.50);d=np.array([1.,0.]);heat=.05;production=1.;demand=0.
    if name.startswith('01'):
        ref=[.49,.485,.495,.49,.485,.49][phase]
        band=[(.48,.50),(.48,.49),(.49,.50),(.485,.495),(.48,.49),(.48,.50)][phase]
    elif name.startswith('02'):
        band=[(.48,.50),(.48,.49),(.49,.50),(.48,.50)][phase];ref=sum(band)/2
        heat=[.05,.2,.1,.05][phase];production=[1.,.7,1.2,1.][phase];demand=[.90,.85,.95,.90][phase]
    elif name.startswith('03'):
        d=np.array([[1.,0.],[.95,.1],[1.05,-.1],[.98,0.]][phase])
    elif name.startswith('04'):
        ref=.49 if k<n//2 else .495
    elif name.startswith('05'):
        ref=[.49,.485,.495,.49][phase]
    return dict(reference=ref,band=band,disturbance=d,heat_weight=heat,production_weight=production,minimum_production=demand)

def plant_step(x,u,d,c,multipliers,heat_factor):
    if np.allclose(multipliers,1) and heat_factor==1:
        return integrate_interval(x,u,d,c)['states']
    def rhs_modified(t,value):
        ca,cb,cc,theta=value;r=reaction_rates(value)*multipliers;a=.2*u[0]
        return [a*(d[0]-ca)-r[0],-a*cb+r[0]-r[1],-a*cc+r[1]-r[2],a*(d[1]-theta)+heat_factor*np.dot([.06,.04,.02],r)+u[1]]
    tau=np.linspace(0,c['sampling_interval'],c['interval_nodes'])
    s=solve_ivp(rhs_modified,(0,tau[-1]),x,t_eval=tau,method='DOP853',rtol=1e-10,atol=1e-12)
    if not s.success or not np.isfinite(s.y).all():raise RuntimeError('Independent modified-plant integration failed')
    return s.y.T

def metrics(records,c,name,mode,seed,variant,n,status,reason,model_hash,wall):
    x=np.asarray(records['states']);u=np.asarray(records['controls']).reshape((-1,2));count=len(u);ts=c['sampling_interval']
    m=dict(experiment=name,controller=mode,seed=seed,variant=variant,requested_steps=n,completed_steps=count,status=status,stop_reason=reason,
           sampling_interval_minutes=ts,horizon=40,model_sha256=model_hash,wall_seconds=wall,
           measurement_assumption='Full state; Gaussian measurement noise and one-sample actuator delay only in experiment 05.',
           noise_std=[.002,.002,.002,.01] if name.startswith('05') else [0,0,0,0],
           delay_samples=1 if name.startswith('05') else 0,
           weights=dict(production='scheduled',heat='scheduled',flow=.05,quality_penalty=1000.,quality_margin=1e-4),
           expected_infeasible=name.startswith('07'),
           infeasible_request_rejected=name.startswith('07') and status=='rejected_requirement',
           plant_config=c)
    if not count:return m
    dense=np.asarray(records['interval_states']);r=np.asarray(records['references']);bands=np.asarray(records['quality_bands'])
    error=x[1:,1]-r;lo=bands[:,0,None];hi=bands[:,1,None];qv=np.maximum(lo-dense[...,1],0)+np.maximum(dense[...,1]-hi,0)
    outside=(qv>1e-7).astype(float);move=np.diff(np.vstack((c['nominal_control'],u)),axis=0)
    b=np.asarray(c['control_bounds']);times=np.asarray(records['solve_seconds'])
    p=records['production_weights'];h=records['heat_weights']
    slack=np.maximum(lo+1e-4-dense[...,1],0)+np.maximum(dense[...,1]-(hi-1e-4),0)
    cost=-np.asarray(p)*u[:,0]*x[1:,1]+np.asarray(h)*(u[:,1]/.6)**2+.05*(u[:,0]/2)**2+1000*(slack[:,-1]/.1)**2+.1*np.sum((move/np.asarray(c['control_move_limits']))**2,axis=1)
    demand=np.asarray(records['minimum_production']);shortfall=np.maximum(demand-u[:,0]*x[1:,1],0)
    cost+=100*shortfall**2
    m.update(tracking_rmse=float(np.sqrt(np.mean(error**2))),tracking_IAE=float(ts*np.abs(error).sum()),final_B=float(x[-1,1]),final_tracking_error=float(error[-1]),
             mean_production_proxy=float(np.mean(u[:,0]*x[1:,1])),integrated_production_proxy=float(np.sum(u[:,0]*np.trapezoid(dense[...,1],dx=ts/(c['interval_nodes']-1),axis=1))),
             mean_heat_effort=float(np.mean((u[:,1]/.6)**2)),mean_flow_effort=float(np.mean((u[:,0]/2)**2)),
             mean_scheduled_proxy_cost=float(np.mean(cost)),maximum_quality_violation=float(qv.max()),quality_outside_sampled_minutes=float(np.trapezoid(outside,dx=ts/(c['interval_nodes']-1),axis=1).sum()),
             endpoint_fraction_in_quality_band=float(np.mean(qv[:,-1]<=1e-7)),
             maximum_physical_state_violation=float(np.maximum(state_constraint_values(dense,c),0).max()),
             maximum_control_bound_violation=float(max(0.,np.max(b[:,0]-u),np.max(u-b[:,1]))),
             maximum_control_move_violation=float(np.maximum(np.abs(move)-np.asarray(c['control_move_limits']),0).max()),
             control_total_variation=np.abs(move).sum(axis=0).tolist(),fallback_steps=int(sum(records['fallback_used'])),
             solver_success_steps=int(sum(records['solver_success'])),actuator_rate_limited_steps=int(sum(records['actuator_limited'])),
             solve_mean_seconds=float(times.mean()),solve_p95_seconds=float(np.percentile(times,95)),solve_max_seconds=float(times.max()),solve_over_15_seconds=int(np.sum(times>15)))
    changes=np.flatnonzero(np.r_[True,np.diff(r)!=0]|np.r_[True,np.any(np.diff(bands,axis=0)!=0,axis=1)])
    phases=[]
    for j,start in enumerate(changes):
        end=changes[j+1] if j+1<len(changes) else count;e=error[start:end]
        tail=np.maximum.accumulate(np.abs(e)[::-1])[::-1];ids=np.flatnonzero(tail<=1e-3)
        phases.append(dict(start_minute=float(start*ts),end_minute=float(end*ts),reference=float(r[start]),band=bands[start].tolist(),
                           tracking_settling_minutes_at_1e_3=float((ids[0]+1)*ts) if len(ids) else None,
                           peak_absolute_tracking_error=float(np.abs(e).max()),mean_production_proxy=float(np.mean(u[start:end,0]*x[start+1:end+1,1])),
                           quality_outside_sampled_minutes=float(np.trapezoid(outside[start:end],dx=ts/(c['interval_nodes']-1),axis=1).sum())))
    demand=np.asarray(records['minimum_production']);shortfall=np.maximum(demand-u[:,0]*x[1:,1],0)
    m['integrated_production_shortfall']=float(ts*shortfall.sum());m['maximum_production_shortfall']=float(shortfall.max())
    m['requirement_phase_metrics']=phases
    return m

def save(folder,records,m):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    if len(records['controls']):np.savez(folder/'trajectory_predictions.npz',**{k:np.asarray(v) for k,v in records.items()})

def run(name,mode,seed,variant,args):
    spec=EXPERIMENTS[name];n=4 if args.smoke else spec['steps'];model=SavedFlowModel(args.model_dir);c=model.config['plant_config']
    raw=(args.model_dir/'parameter_tree.pkl').read_bytes();model_hash=hashlib.sha256(raw).hexdigest()
    folder=(Path(tempfile.gettempdir())/'daehn_control_experiments_smoke' if args.smoke else ROOT)/name/mode/variant/f'seed={seed}'
    existing=folder/'metrics.json'
    if args.resume and existing.exists():
        previous=json.loads(existing.read_text())
        if previous['status'] in ['completed','rejected_requirement','physical_constraint_stop','controller_failure'] and previous['requested_steps']==n and previous['model_sha256']==model_hash:
            print('SKIP',name,mode,variant,seed,previous['status'],flush=True);return previous
    initial=dict(nominal=c['nominal_state'],low_inventory=[.2,.55,.2,.2],near_boundary=[.02,.5,.5,1.5])
    x=np.asarray(initial.get(variant,c['nominal_state']),float);actual=np.asarray(c['nominal_control'],float);queue=actual.copy()
    records={k:[] for k in ['states','controls','commanded_controls','measured_states','disturbances','references','quality_bands','interval_states','predicted_next_states','solve_seconds','solver_success','fallback_used','actuator_limited','production_weights','heat_weights','plant_rate_multipliers','plant_heat_multipliers','minimum_production']}
    records['states']=[x.copy()];cache={};active=None;status='running';reason=None;rng=np.random.default_rng(seed);start=time.perf_counter()
    print('START',name,mode,variant,seed,'steps',n,flush=True)
    for k in range(n):
        req=requirements(name,k,n)
        if name.startswith('07') and k>=n//2:
            req['reference']=1.3 if variant=='outside_bounds' else .71
            req['band']=(1.29,1.31) if variant=='outside_bounds' else (.70,.72)
        sigma=np.array([.002,.002,.002,.01]) if name.startswith('05') else np.zeros(4)
        measured=x+rng.normal(size=4)*sigma
        previous=actual.copy();d=req['disturbance']
        try:
            if mode=='fixed':
                command=np.asarray(c['nominal_control']);info={'solve_seconds':0.,'solver_success':True,'fallback_used':False,'predicted_next_state':model(np.r_[measured,command,d])[:4]}
            else:
                key=(mode,tuple(req['band']),req['heat_weight'],req['production_weight'],req['minimum_production']) if mode=='economic' else (mode,)
                if key not in cache:
                    cache[key]=MPCController(args.model_dir,objective_mode='economic' if mode=='economic' else 'tracking',
                                             terminal_cost='lqr' if mode=='lqr' else 'legacy',target_selection='continuous' if mode=='lqr' else 'legacy',
                                             quality_band=req['band'],heat_weight=req['heat_weight'],production_weight=req['production_weight'],minimum_production=req['minimum_production'])
                ctrl=cache[key]
                if active is not None and active is not ctrl:
                    for attr in ['plan','bias','_previous_raw_prediction','_previous_context','_steady_target','_target_switches']:
                        setattr(ctrl,attr,getattr(active,attr))
                active=ctrl
                command,info=ctrl.step(measured,previous,d,req['reference'])
        except (ValueError,RuntimeError) as error:
            status='rejected_requirement' if name.startswith('07') and k>=n//2 else 'controller_failure';reason=str(error);break
        applied=queue.copy() if name.startswith('05') else command.copy()
        queue=command.copy()
        bounds=np.asarray(c['control_bounds']);move=np.asarray(c['control_move_limits'])
        limited=np.clip(applied,np.maximum(bounds[:,0],previous-move),np.minimum(bounds[:,1],previous+move))
        hit=not np.allclose(applied,limited,atol=1e-8,rtol=0)
        if name.startswith('05'):applied=limited
        mult=np.ones(3);hf=1.
        if name.startswith('06') and k>=n//4:
            mult=np.array([1.1,.9,1.05] if variant=='faster' else [.9,1.1,.95]);hf=1.15 if variant=='faster' else .85
        dense=plant_step(x,applied,d,c,mult,hf)
        actual=applied.copy();x=dense[-1]
        values=dict(controls=actual,commanded_controls=command,measured_states=measured,disturbances=d,references=req['reference'],quality_bands=req['band'],
                    interval_states=dense,predicted_next_states=info['predicted_next_state'],solve_seconds=info['solve_seconds'],solver_success=info['solver_success'],fallback_used=info['fallback_used'],actuator_limited=hit,
                    production_weights=req['production_weight'],heat_weights=req['heat_weight'],plant_rate_multipliers=mult,plant_heat_multipliers=hf,minimum_production=req['minimum_production'])
        for key,v in values.items():records[key].append(v)
        records['states'].append(x.copy())
        if np.max(state_constraint_values(dense,c))>1e-7:
            status='physical_constraint_stop';reason='True plant exceeded a sampled physical operating bound';break
        if (k+1)%32==0 or k==0:
            m=metrics(records,c,name,mode,seed,variant,n,status,reason,model_hash,time.perf_counter()-start);save(folder,records,m)
            print('PROGRESS',name,mode,variant,seed,k+1,'/',n,'B',round(x[1],6),'fallbacks',m.get('fallback_steps'),flush=True)
    if status=='running':status='completed'
    m=metrics(records,c,name,mode,seed,variant,n,status,reason,model_hash,time.perf_counter()-start)
    if status in ['rejected_requirement','controller_failure']:m['rejected_request']={'reference':req['reference'],'band':list(req['band']),'time_minute':k*c['sampling_interval'],'minimum_production':req['minimum_production']}
    save(folder,records,m);print('DONE',name,mode,variant,seed,status,'steps',m['completed_steps'],'quality_minutes',m.get('quality_outside_sampled_minutes'),'production',m.get('mean_production_proxy'),flush=True)
    del cache,active,model;gc.collect()
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['all']+list(EXPERIMENTS),default='all')
    p.add_argument('--mode',choices=['all','fixed','tracking','economic','lqr'],default='all')
    p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.model_dir=args.model_dir.resolve()
    names=list(EXPERIMENTS) if args.experiment=='all' else [args.experiment]
    for name in names:
        spec=EXPERIMENTS[name];summaries=[]
        modes=spec['modes'] if args.mode=='all' else [args.mode]
        for mode in modes:
            for variant in spec['variants']:
                for seed in spec['seeds']:
                    summaries.append(run(name,mode,seed,variant,args))
        destination=(Path(tempfile.gettempdir())/'daehn_control_experiments_smoke' if args.smoke else ROOT)/name
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'metrics.json').write_text(json.dumps(dict(experiment=name,runs=summaries,comparison_note='Tracking targets the band midpoint; economic control optimizes within the same band. Scores use the same scheduled proxy weights and realized plant trajectories. LQR is included in experiment 01.'),indent=2),encoding='utf-8')
        print('EXPERIMENT COMPLETE',name,flush=True)

if __name__=='__main__':main()