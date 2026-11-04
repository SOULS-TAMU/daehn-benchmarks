"""Train the conditional MPC CSTR flow model with prior CSTR hyperparameters.

Native DAEHN MLP/Adam and algebraic projection are used. The initial state and
reaction-rate equations are exact; differential dynamics are loss-enforced.
An epoch traverses all selected TRAIN transitions, sampling a dense local-time
label for each transition (half the labels are endpoints). Splits are preserved.
"""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,pickle,hashlib,time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from daehn.train.projected_neural import Adam,apply_updates
from model import reaction_rates,admissible_state
from surrogate import make_functions,rhs_jax,SavedFlowModel

CASE=_CASE_DIR
RESULTS=CASE.parent/'results'/CASE.name
_EVALUATORS={}

def save_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def load_dataset(directory):
    meta=json.loads((directory/'metadata.json').read_text())
    if hashlib.sha256((directory/'data.npz').read_bytes()).hexdigest()!=meta['data_sha256']:raise ValueError('Dataset hash mismatch')
    with np.load(directory/'data.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    return arrays,meta

def metrics(parameters,indices,a,cfg):
    key=json.dumps(cfg,sort_keys=True)
    if key not in _EVALUATORS:
        _,predict,derivative=make_functions(cfg,cfg['batch_size'])
        @jax.jit
        def evaluate(p,x,y,d):
            tau=jnp.full((len(x),1),cfg['sampling_interval'])
            pred=predict(p,x,tau);dd=derivative(p,x,tau)
            scale=jnp.asarray(cfg['target_scale']);ds=jnp.asarray(cfg['velocity_scale'])
            return pred,jnp.mean(((pred-y)/scale)**2,axis=1),jnp.mean(((dd-d)/ds)**2,axis=1),jnp.mean(((dd-rhs_jax(pred[:,:4],x))/ds)**2,axis=1)
        _EVALUATORS[key]=evaluate
    evaluate=_EVALUATORS[key]
    predictions=[];data=[];deriv=[];physics=[];b=cfg['batch_size']
    for start in range(0,len(indices),b):
        ids=indices[start:start+b];n=len(ids);padded=np.pad(ids,(0,b-n),mode='edge')
        result=evaluate(parameters,jnp.asarray(a['inputs'][padded]),jnp.asarray(a['targets'][padded]),jnp.asarray(a['interval_derivatives'][padded,-1]))
        predictions.append(np.asarray(result[0])[:n]);data.extend(np.asarray(result[1])[:n]);deriv.extend(np.asarray(result[2])[:n]);physics.extend(np.asarray(result[3])[:n])
    pred=np.concatenate(predictions)
    return {'data_loss':float(np.mean(data)),'derivative_loss':float(np.mean(deriv)),'physics_loss':float(np.mean(physics)),'mse_state':float(np.mean((pred[:,:4]-a['targets'][indices,:4])**2))},pred

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=CASE/'data')
    parser.add_argument('--epochs',type=int,default=5000,help='Total epochs, including pretraining')
    parser.add_argument('--pretrain-epochs',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--hidden-size',type=int,default=64);parser.add_argument('--hidden-layers',type=int,default=3)
    parser.add_argument('--learning-rate',type=float,default=1e-3);parser.add_argument('--projected-learning-rate',type=float,default=5e-5)
    parser.add_argument('--lambda-data',type=float,default=1.0);parser.add_argument('--lambda-derivative',type=float,default=0.5);parser.add_argument('--lambda-physics',type=float,default=1.0)
    parser.add_argument('--print-every',type=int,default=20);parser.add_argument('--validation-every',type=int,default=20)
    parser.add_argument('--seed',type=int,default=11);parser.add_argument('--train-limit',type=int)
    parser.add_argument('--model',choices=['daehn'],default='daehn')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:args.epochs=2;args.pretrain_epochs=1;args.batch_size=8;args.hidden_size=8;args.hidden_layers=1;args.train_limit=16;args.print_every=1;args.validation_every=1
    if not 0<=args.pretrain_epochs<args.epochs or min(args.batch_size,args.hidden_size,args.hidden_layers,args.print_every,args.validation_every)<1:parser.error('Invalid epoch, batch, architecture or logging settings')
    if min(args.learning_rate,args.projected_learning_rate)<=0 or min(args.lambda_data,args.lambda_derivative,args.lambda_physics)<0:parser.error('Learning rates must be positive and loss weights nonnegative')
    a,meta=load_dataset(args.data_dir);rng=np.random.default_rng(args.seed)
    train=np.flatnonzero(a['split']==0);val=np.flatnonzero(a['split']==1);test=np.flatnonzero(a['split']==2)
    if args.train_limit is not None:
        if args.train_limit<1:parser.error('train-limit must be positive')
        train=rng.choice(train,min(args.train_limit,len(train)),replace=False)
    if args.smoke:val=val[:16];test=test[:16]
    if min(len(train),len(val),len(test))<1:raise ValueError('All three dataset splits must be nonempty')
    dd=a['interval_derivatives'][train].reshape((-1,4))
    cfg={'task':'cstr_mpc_flow','model':'daehn','projection':'explicit_algebraic_rate','differential_dynamics':'loss_enforced','initial_condition':'hard','hidden_sizes':[args.hidden_size]*args.hidden_layers,'batch_size':args.batch_size,'sampling_interval':meta['effective_config']['sampling_interval'],'input_mean':a['inputs'][train].mean(axis=0).tolist(),'input_scale':np.maximum(a['inputs'][train].std(axis=0),1e-8).tolist(),'target_scale':np.maximum(a['targets'][train].std(axis=0),1e-4).tolist(),'velocity_mean':dd.mean(axis=0).tolist(),'velocity_scale':np.maximum(dd.std(axis=0),1e-4).tolist(),'plant_config':meta['effective_config'],'epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,'learning_rate':args.learning_rate,'projected_learning_rate':args.projected_learning_rate,'lambda_data':args.lambda_data,'lambda_derivative':args.lambda_derivative,'lambda_physics':args.lambda_physics,'seed':args.seed,'data_sha256':meta['data_sha256'],'train_samples':len(train),'feature_order':meta['input_names']+['local_tau'],'output_order':meta['output_names']}
    network,predict,derivative=make_functions(cfg,args.batch_size)
    parameters=network.init(jax.random.key(args.seed),9,4,jnp.float64)
    def loss(p,inputs,tau,targets,labels,mask,joint):
        pred=predict(p,inputs,tau);dd=derivative(p,inputs,tau)
        data=jnp.mean(((pred-targets)/jnp.asarray(cfg['target_scale']))**2,axis=1)
        dloss=jnp.mean(((dd-labels)/jnp.asarray(cfg['velocity_scale']))**2,axis=1)
        phys=jnp.mean(((dd-rhs_jax(pred[:,:4],inputs))/jnp.asarray(cfg['velocity_scale']))**2,axis=1)
        mean=lambda x:jnp.sum(x*mask)/jnp.sum(mask)
        components=(mean(data),mean(dloss),mean(phys))
        total=args.lambda_data*components[0]
        if joint:total+=args.lambda_derivative*components[1]+args.lambda_physics*components[2]
        return total,components
    def build_step(optimizer,joint):
        @jax.jit
        def step(p,state,x,t,y,d,mask):
            (total,components),gradient=jax.value_and_grad(lambda p:loss(p,x,t,y,d,mask,joint),has_aux=True)(p)
            updates,state=optimizer.update(gradient,state,p)
            return apply_updates(p,updates),state,total,components
        return step
    pre=Adam(args.learning_rate);joint=Adam(args.projected_learning_rate)
    pre_step=build_step(pre,False);joint_step=build_step(joint,True);state=pre.init(parameters)
    history=[];start=time.perf_counter();b=args.batch_size
    print(f"Train/validation/test transitions: {len(train)}/{len(val)}/{len(test)}; {args.epochs} total epochs, {args.pretrain_epochs} pretraining",flush=True)
    for epoch in range(1,args.epochs+1):
        active=epoch>args.pretrain_epochs
        if epoch==args.pretrain_epochs+1:state=joint.init(parameters)
        order=rng.permutation(train);sums=np.zeros(4);count=0
        for offset in range(0,len(order),b):
            ids=order[offset:offset+b];n=len(ids);ids=np.pad(ids,(0,b-n),mode='edge')
            node=rng.integers(1,len(a['local_time']),size=b);node[rng.random(b)<0.5]=len(a['local_time'])-1
            target=np.concatenate((a['interval_states'][ids,node],a['interval_rates'][ids,node]),axis=1)
            mask=(np.arange(b)<n).astype(float)
            parameters,state,total,components=(joint_step if active else pre_step)(parameters,state,jnp.asarray(a['inputs'][ids]),jnp.asarray(a['local_time'][node,None]),jnp.asarray(target),jnp.asarray(a['interval_derivatives'][ids,node]),jnp.asarray(mask))
            values=np.r_[float(total),np.asarray(components)]
            if not np.all(np.isfinite(values)):raise RuntimeError(f'Nonfinite optimization at epoch {epoch}')
            sums+=n*values;count+=n
        means=sums/count;row={'epoch':epoch,'phase':'physics' if active else 'pretrain','loss':float(means[0]),'train_data_loss':float(means[1]),'train_derivative_loss':float(means[2]),'train_physics_loss':float(means[3])}
        if epoch%args.validation_every==0 or epoch==args.epochs:
            vm,_=metrics(parameters,val,a,cfg);row.update({'validation_'+k:v for k,v in vm.items()})
        history.append(row)
        if epoch==1 or epoch%args.print_every==0 or epoch==args.epochs:print(json.dumps(row),flush=True)
    wall=time.perf_counter()-start
    test_metrics,pred=metrics(parameters,test,a,cfg)
    branch='smoke' if args.smoke else 'flow'
    relative=Path(branch)/'daehn'/f'seed={args.seed}'
    model_dir=RESULTS/'model'/'new_runs'/relative;output=RESULTS/'output'/'new_runs'/relative
    if (model_dir/'parameter_tree.pkl').exists():raise FileExistsError('A completed run exists; use a different seed or fresh run directory')
    model_dir.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    host=jax.tree_util.tree_map(np.asarray,parameters)
    with (model_dir/'parameter_tree.pkl').open('wb') as handle:pickle.dump(host,handle,pickle.HIGHEST_PROTOCOL)
    save_json(model_dir/'run_config.json',cfg)
    manifest={'schema_version':1,'task':cfg['task'],'architecture':'anchored_local_flow_mlp','checkpoint_sha256':hashlib.sha256((model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(),'input_size':8,'network_input_size':9,'output_size':7,'network_output_size':4,'sampling_interval':cfg['sampling_interval'],'projection':cfg['projection'],'differential_dynamics':cfg['differential_dynamics'],'initial_condition':cfg['initial_condition'],'training_wall_seconds':wall,'data_sha256':meta['data_sha256']}
    save_json(model_dir/'checkpoint_manifest.json',manifest)
    np.savez(model_dir/'evaluation_data.npz',inputs=a['inputs'][test],targets=a['targets'][test],transition_id=test)
    trajectory=np.flatnonzero((a['split']==2)&(a['sample_kind']==1))
    if args.smoke:trajectory=trajectory[:8]
    np.savez(model_dir/'trajectory_data.npz',inputs=a['inputs'][trajectory],targets=a['targets'][trajectory],trajectory_id=a['trajectory_id'][trajectory],step_index=a['step_index'][trajectory])
    np.savez(output/'predictions.npz',inputs=a['inputs'][test],exact=a['targets'][test],prediction=pred,transition_id=test)
    _,traj_pred=metrics(parameters,trajectory,a,cfg)
    np.savez(output/'trajectory_predictions.npz',inputs=a['inputs'][trajectory],exact=a['targets'][trajectory],prediction=traj_pred,trajectory_id=a['trajectory_id'][trajectory],step_index=a['step_index'][trajectory])
    test_metrics.update({'algebraic_violation_max':float(np.max(np.abs(pred[:,4:]-reaction_rates(pred[:,:4])))),'test_transitions':len(test),'predicted_unsafe_transitions':int(np.sum(~admissible_state(pred[:,:4],cfg['plant_config']))),'true_unsafe_transitions':int(np.sum(~a['next_in_domain'][test])),'trajectory_predictions':'one-step teacher-forced predictions; not recursive/closed-loop validation','closed_loop_MPC_validated':False,'training_wall_seconds':wall})
    save_json(output/'metrics.json',test_metrics);save_json(output/'training_history.json',history)
    save_json(output/'training_summary.json',{'status':'trained','epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,'training_wall_seconds':wall,'final_history':history[-1]})
    # Saved-model reload, hard-IC, and differentiable MPC interface checks.
    saved=SavedFlowModel(model_dir);probe=a['inputs'][test[0]]
    assert np.allclose(saved(probe),pred[0],rtol=1e-9,atol=1e-11)
    assert np.allclose(saved.predict_jax(probe,tau=0)[:4],probe[:4],rtol=0,atol=1e-12)
    assert np.all(np.isfinite(saved.linearize(probe)))
    print(json.dumps({'model_dir':str(model_dir),'output_dir':str(output),'metrics':test_metrics,'reload_hard_IC_and_Jacobian_checks':'passed'},indent=2))

if __name__=='__main__':main()
