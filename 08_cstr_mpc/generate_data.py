"""Generate state/control-conditioned continuous CSTR flow data for future MPC.

Run after reading FORMULATION.md. Outputs are input data (data.npz and metadata.json)
inside this case's data folder; no model/result files are created by this script.
"""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from scipy.stats import qmc
from model import load_config,validate_config,reaction_rates,integrate_interval,admissible_state,INPUT_NAMES,OUTPUT_NAMES,CONSTRAINT_NAMES

CASE=_CASE_DIR

def sample_states(rng,n,c):
    bounds=np.asarray(c['state_bounds']);cap=c['inventory_upper']
    x=np.column_stack((rng.dirichlet(np.ones(4),size=n)[:,:3]*cap,rng.uniform(*bounds[3],size=n)))
    for i in range(n):
        if rng.random()<0.2:
            face=int(rng.integers(0,6))
            if face<3:x[i,face]=0
            elif face==3:x[i,:3]=rng.dirichlet(np.ones(3))*cap
            else:x[i,3]=bounds[3,face-4]
        if not admissible_state(x[i],c):
            for _ in range(1000):
                value=np.r_[rng.dirichlet(np.ones(4))[:3]*cap,rng.uniform(*bounds[3])]
                if admissible_state(value,c):x[i]=value;break
            else:raise RuntimeError('Cannot sample admissible initial state with these bounds')
    return x

def group_splits(kind_by_group,rng,c):
    assigned={}
    for kind in [0,1]:
        ids=np.array([group for group,value in kind_by_group.items() if value==kind],dtype=int)
        rng.shuffle(ids);ntrain=int(c['split_fractions'][0]*len(ids));nval=int(c['split_fractions'][1]*len(ids))
        if min(ntrain,nval,len(ids)-ntrain-nval)<1:raise ValueError('Every data source must have a nonempty train/validation/test group split')
        for split,groups in enumerate([ids[:ntrain],ids[ntrain:ntrain+nval],ids[ntrain+nval:]]):
            for group in groups:assigned[int(group)]=split
    return assigned

def generate(c):
    rng=np.random.default_rng(c['seed']);records=[];kinds={}
    cb=np.asarray(c['control_bounds']);db=np.asarray(c['disturbance_bounds']);move=np.asarray(c['control_move_limits'])
    def add(x,u,d,previous,group,step,kind):
        if not admissible_state(x,c):raise ValueError('Out-of-domain initial state; never clip initial conditions silently')
        result=integrate_interval(x,u,d,c)
        records.append({'inputs':np.r_[x,u,d],'targets':np.r_[result['next_state'],result['next_rates']],'interval_states':result['states'],'interval_rates':result['rates'],'interval_derivatives':result['derivatives'],'interval_violations':result['violations'],'previous_control':previous.copy(),'trajectory_id':group,'step_index':step,'sample_kind':kind,'split':-1,'next_in_domain':bool(admissible_state(result['next_state'],c)),'interval_feasible_sampled':bool(result['violations'].max()<=1e-9),'solver_nfev':result['solver_nfev']})
        kinds[group]=kind
        return result
    n=c['independent_samples'];initial=sample_states(rng,n,c)
    operating=qmc.LatinHypercube(d=4,seed=c['seed']+1).random(n)
    boxes=np.vstack((cb,db));operating=qmc.scale(operating,boxes[:,0],boxes[:,1])
    for i in range(n):
        if i%10==0:
            column=(i//10)%4;operating[i,column]=boxes[column,(i//40)%2]
        u=operating[i,:2];d=operating[i,2:]
        previous=np.clip(u+rng.uniform(-move,move),cb[:,0],cb[:,1])
        add(initial[i],u,d,previous,i,0,0)
    initials=sample_states(rng,c['linked_trajectories'],c)
    for trajectory,x in enumerate(initials):
        previous=rng.uniform(cb[:,0],cb[:,1]);d=rng.uniform(db[:,0],db[:,1]);target=previous.copy()
        for step in range(c['steps_per_trajectory']):
            if step%16==0:target=rng.uniform(cb[:,0],cb[:,1])
            if step%8==0:d=rng.uniform(db[:,0],db[:,1])
            change=np.clip(0.3*(target-previous)+rng.uniform(-0.5*move,0.5*move),-move,move)
            u=np.clip(previous+change,cb[:,0],cb[:,1])
            result=add(x,u,d,previous,n+trajectory,step,1)
            x=result['next_state'];previous=u
            if result['violations'].max()>1e-9:break
    assignment=group_splits(kinds,np.random.default_rng(c['seed']+2),c)
    for record in records:record['split']=assignment[record['trajectory_id']]
    arrays={key:np.asarray([record[key] for record in records]) for key in records[0]}
    arrays['local_time']=np.linspace(0,c['sampling_interval'],c['interval_nodes'])
    return arrays

def summarize(a,c):
    train=a['split']==0
    mean=a['inputs'][train].mean(axis=0);std=np.maximum(a['inputs'][train].std(axis=0),1e-8)
    target_mean=a['targets'][train].mean(axis=0);target_std=np.maximum(a['targets'][train].std(axis=0),1e-8)
    counts={name:int(np.sum(a['split']==i)) for i,name in enumerate(['train','validation','test'])}
    algebraic=np.max(np.abs(a['interval_rates']-reaction_rates(a['interval_states'])))
    for group in np.unique(a['trajectory_id']):
        if len(np.unique(a['split'][a['trajectory_id']==group]))!=1:raise RuntimeError('Trajectory split leakage')
    return {'schema_version':1,'purpose':'state/control-conditioned CSTR flow-map learning for downstream B-concentration MPC','input_names':INPUT_NAMES,'output_names':OUTPUT_NAMES,'constraint_names':CONSTRAINT_NAMES,'split_codes':{'train':0,'validation':1,'test':2},'sample_kind_codes':{'independent':0,'linked_trajectory':1},'samples':len(a['inputs']),'linked_transitions':int(np.sum(a['sample_kind']==1)),'trajectory_groups':len(np.unique(a['trajectory_id'])),'split_counts':counts,'unsafe_sampled_intervals':int(np.sum(~a['interval_feasible_sampled'])),'out_of_domain_endpoints':int(np.sum(~a['next_in_domain'])),'algebraic_residual_max':float(algebraic),'input_min':a['inputs'].min(axis=0).tolist(),'input_max':a['inputs'].max(axis=0).tolist(),'target_min':a['targets'].min(axis=0).tolist(),'target_max':a['targets'].max(axis=0).tolist(),'normalization_training_only':{'input_mean':mean.tolist(),'input_scale':std.tolist(),'target_mean':target_mean.tolist(),'target_scale':target_std.tolist()},'effective_config':c,'labels':'Continuous-flow integration; controls and disturbances held constant per interval. Violations are sampled, not continuous-time certificates. No measured/noisy plant data.','model_source_sha256':hashlib.sha256(((_CASE_DIR / "_impl" / 'model.py')).read_bytes()).hexdigest(),'generator_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CASE/'config.json')
    parser.add_argument('--output-dir',type=Path,default=CASE/'data')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--independent-samples',type=int);parser.add_argument('--trajectories',type=int);parser.add_argument('--steps',type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config=load_config(args.config)
    for arg,key in [('seed','seed'),('independent_samples','independent_samples'),('trajectories','linked_trajectories'),('steps','steps_per_trajectory')]:
        if getattr(args,arg) is not None:config[key]=getattr(args,arg)
    if args.smoke:
        config.update(independent_samples=32,linked_trajectories=10,steps_per_trajectory=8)
        if args.output_dir==CASE/'data':args.output_dir=CASE/'data/smoke'
    validate_config(config);start=time.perf_counter();arrays=generate(config);meta=summarize(arrays,config)
    meta['generation_seconds']=time.perf_counter()-start
    output=(args.output_dir if args.output_dir.is_absolute() else CASE/args.output_dir).resolve()
    if not output.is_relative_to(CASE/'data'):raise ValueError('Input datasets must remain under this case data/ folder')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'data.npz',**arrays)
    meta['data_sha256']=hashlib.sha256((output/'data.npz').read_bytes()).hexdigest()
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:meta[key] for key in ['samples','linked_transitions','split_counts','unsafe_sampled_intervals','algebraic_residual_max','generation_seconds']},indent=2))

if __name__=='__main__':main()
