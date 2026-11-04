"""Train a matched soft-constraint CSTR PINN; no control runs or projection."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from daehn.train.projected_neural import Adam
from daehn.train.steps import make_train_step
from train import load_dataset,save_json
from model import reaction_rates,admissible_state
from pinn_surrogate import make_functions,rates_jax,rhs_with_rates,SavedPINNFlowModel

CASE=_CASE_DIR
RESULTS=CASE.parent/'results'/CASE.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=CASE/'data')
    parser.add_argument('--name',default='matched_tanh_64x3')
    parser.add_argument('--epochs',type=int,default=5000,help='Total including pretraining')
    parser.add_argument('--pretrain-epochs',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--hidden-layers',type=int,default=3)
    parser.add_argument('--hidden-nodes','--hidden-size',dest='hidden_nodes',type=int,default=64)
    parser.add_argument('--activation',choices=['tanh','relu'],default='tanh')
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--physics-learning-rate',type=float,default=.00005)
    parser.add_argument('--lambda-data',type=float,default=1.)
    parser.add_argument('--lambda-derivative',type=float,default=.5)
    parser.add_argument('--lambda-physics',type=float,default=1.)
    parser.add_argument('--lambda-algebraic',type=float,default=1.)
    parser.add_argument('--print-every',type=int,default=20)
    parser.add_argument('--validate-every','--validation-every',dest='validate_every',type=int,default=20)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--train-limit',type=int)
    parser.add_argument('--smoke',action='store_true',help='Two epochs; tiny separate smoke artifacts')
    args=parser.parse_args()
    if args.smoke:
        args.epochs=2; args.pretrain_epochs=1; args.batch_size=8; args.hidden_nodes=8
        args.hidden_layers=1; args.train_limit=16; args.print_every=1; args.validate_every=1
    if not 0<=args.pretrain_epochs<args.epochs:parser.error('Require 0 <= pretrain-epochs < epochs')
    if min(args.batch_size,args.hidden_nodes,args.hidden_layers,args.print_every,args.validate_every)<1:parser.error('Batch, architecture and log intervals must be positive')
    if not np.isfinite([args.learning_rate,args.physics_learning_rate,args.lambda_data,args.lambda_derivative,args.lambda_physics,args.lambda_algebraic]).all():parser.error('Nonfinite hyperparameter')
    if min(args.learning_rate,args.physics_learning_rate)<=0 or min(args.lambda_data,args.lambda_derivative,args.lambda_physics,args.lambda_algebraic)<0:parser.error('Invalid learning rate or loss weight')
    if args.lambda_algebraic==0 or args.lambda_physics==0:parser.error('PINN requires positive physics and algebraic weights')
    if args.train_limit is not None and args.train_limit<1:parser.error('train-limit must be positive')
    if not args.name or any(c in args.name for c in '/\\:') or args.name in ('.','..'):parser.error('name must be one folder component')
    relative=Path('smoke' if args.smoke else 'flow')/'pinn'/args.name/f'seed={args.seed}'
    model_dir=RESULTS/'model'/relative; output=RESULTS/'output'/relative
    if model_dir.exists() or output.exists():parser.error(f'Run exists; choose another --name: {model_dir}')
    a,meta=load_dataset(args.data_dir); rng=np.random.default_rng(args.seed)
    train=np.flatnonzero(a['split']==0); val=np.flatnonzero(a['split']==1); test=np.flatnonzero(a['split']==2)
    if args.train_limit is not None:train=rng.choice(train,min(args.train_limit,len(train)),replace=False)
    if args.smoke:val=val[:16];test=test[:16]
    if min(len(train),len(val),len(test))<1:raise ValueError('Empty dataset split')
    dd=a['interval_derivatives'][train].reshape(-1,4)
    rate_slopes=((a['interval_rates'][train,1:]-a['interval_rates'][train,:1])/a['local_time'][None,1:,None]).reshape(-1,3)
    cfg={'task':'cstr_mpc_flow','model':'pinn','projection':'none','algebraic_constraints':'soft_loss',
         'differential_dynamics':'loss_enforced_using_predicted_rates','initial_condition':'hard',
         'hidden_sizes':[args.hidden_nodes]*args.hidden_layers,'activation':args.activation,'batch_size':args.batch_size,
         'sampling_interval':meta['effective_config']['sampling_interval'],
         'input_mean':a['inputs'][train].mean(axis=0).tolist(),'input_scale':np.maximum(a['inputs'][train].std(axis=0),1e-8).tolist(),
         'target_scale':np.maximum(a['targets'][train].std(axis=0),1e-4).tolist(),
         'velocity_mean':dd.mean(axis=0).tolist(),'velocity_scale':np.maximum(dd.std(axis=0),1e-4).tolist(),
         'rate_velocity_mean':rate_slopes.mean(axis=0).tolist(),'rate_velocity_scale':np.maximum(rate_slopes.std(axis=0),1e-4).tolist(),
         'plant_config':meta['effective_config'],'epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,
         'learning_rate':args.learning_rate,'physics_learning_rate':args.physics_learning_rate,
         'lambda_data':args.lambda_data,'lambda_derivative':args.lambda_derivative,'lambda_physics':args.lambda_physics,
         'lambda_algebraic':args.lambda_algebraic,'seed':args.seed,'data_sha256':meta['data_sha256'],
         'train_samples':len(train),'feature_order':meta['input_names']+['local_tau'],'output_order':meta['output_names'],
         'checkpoint_selection':'final_epoch_matches_existing_DAEHN','run_name':args.name}
    network,predict,derivative=make_functions(cfg)
    parameters=network.init(jax.random.key(args.seed),9,7,jnp.float64)
    cfg['parameter_count']=sum(x.size for x in jax.tree_util.tree_leaves(parameters))
    target_scale=jnp.asarray(cfg['target_scale']); velocity_scale=jnp.asarray(cfg['velocity_scale'])
    def losses(p,batch,joint):
        x,t,y,d,mask=batch; pred=predict(p,x,t); velocity=derivative(p,x,t)
        mean=lambda v:jnp.sum(v*mask)/jnp.sum(mask)
        data=mean(jnp.mean(((pred-y)/target_scale)**2,axis=1))
        deriv=mean(jnp.mean(((velocity-d)/velocity_scale)**2,axis=1))
        phys=mean(jnp.mean(((velocity-rhs_with_rates(pred[:,:4],pred[:,4:],x))/velocity_scale)**2,axis=1))
        alg=mean(jnp.mean(((pred[:,4:]-rates_jax(pred[:,:4]))/target_scale[4:])**2,axis=1))
        total=args.lambda_data*data
        if joint:total+=args.lambda_derivative*deriv+args.lambda_physics*phys+args.lambda_algebraic*alg
        return total,{'data_loss':data,'derivative_loss':deriv,'physics_loss':phys,'algebraic_loss':alg}
    pre=Adam(args.learning_rate); physics=Adam(args.physics_learning_rate)
    pre_step=jax.jit(make_train_step(lambda p,b:losses(p,b,False),pre))
    physics_step=jax.jit(make_train_step(lambda p,b:losses(p,b,True),physics))
    @jax.jit
    def evaluate(p,x,y,d):
        tau=jnp.full((len(x),1),cfg['sampling_interval']); pred=predict(p,x,tau)
        _,m=losses(p,(x,tau,y,d,jnp.ones(len(x))),True)
        # Return per-sample values for exact unpadded aggregate evaluation.
        velocity=derivative(p,x,tau)
        errors=jnp.stack((jnp.mean(((pred-y)/target_scale)**2,axis=1),jnp.mean(((velocity-d)/velocity_scale)**2,axis=1),jnp.mean(((velocity-rhs_with_rates(pred[:,:4],pred[:,4:],x))/velocity_scale)**2,axis=1),jnp.mean(((pred[:,4:]-rates_jax(pred[:,:4]))/target_scale[4:])**2,axis=1)),axis=1)
        return pred,errors
    def score(p,indices):
        predictions=[]; errors=[]; b=args.batch_size
        for start in range(0,len(indices),b):
            ids=indices[start:start+b]; n=len(ids); ids=np.pad(ids,(0,b-n),mode='edge')
            pred,e=evaluate(p,jnp.asarray(a['inputs'][ids]),jnp.asarray(a['targets'][ids]),jnp.asarray(a['interval_derivatives'][ids,-1]))
            predictions.append(np.asarray(pred)[:n]); errors.append(np.asarray(e)[:n])
        pred=np.concatenate(predictions); e=np.concatenate(errors).mean(axis=0)
        return dict(zip(['data_loss','derivative_loss','physics_loss','algebraic_loss'],map(float,e))),pred
    # Fail early if a projection or disconnected rate head accidentally enters the PINN.
    probe=jnp.asarray(a['inputs'][train[:2]]); tau=jnp.full((len(probe),1),cfg['sampling_interval'])
    init=predict(parameters,probe,tau)
    assert float(jnp.max(jnp.abs(init[:,4:]-rates_jax(init[:,:4]))))>1e-10
    rate_grad=jax.grad(lambda pp:jnp.sum(predict(pp,probe,tau)[:,4:]))(parameters)
    assert float(jnp.linalg.norm(rate_grad[-1]['b'][4:]))>0
    state=pre.init(parameters); history=[]; names=['loss','data_loss','derivative_loss','physics_loss','algebraic_loss']
    model_dir.mkdir(parents=True); output.mkdir(parents=True)
    save_json(model_dir/'run_config.json',cfg)
    np.savez(model_dir/'training_indices.npz',train=train,validation=val,test=test)
    started=time.perf_counter(); phase_started=started; phase_times={}
    print(f'Train/validation/test transitions: {len(train)}/{len(val)}/{len(test)}; PINN parameters: {cfg["parameter_count"]}',flush=True)
    for epoch in range(1,args.epochs+1):
        active=epoch>args.pretrain_epochs
        if epoch==args.pretrain_epochs+1:
            phase_times['pretrain_seconds']=time.perf_counter()-phase_started
            phase_started=time.perf_counter(); state=physics.init(parameters)
        order=rng.permutation(train); sums=np.zeros(5); count=0; epoch_started=time.perf_counter()
        for offset in range(0,len(order),args.batch_size):
            ids=order[offset:offset+args.batch_size]; n=len(ids); ids=np.pad(ids,(0,args.batch_size-n),mode='edge')
            node=rng.integers(1,len(a['local_time']),size=args.batch_size); node[rng.random(args.batch_size)<.5]=len(a['local_time'])-1
            y=np.concatenate((a['interval_states'][ids,node],a['interval_rates'][ids,node]),axis=1)
            batch=tuple(map(jnp.asarray,(a['inputs'][ids],a['local_time'][node,None],y,a['interval_derivatives'][ids,node],(np.arange(args.batch_size)<n).astype(float))))
            parameters,state,m=(physics_step if active else pre_step)(parameters,state,batch)
            vals=np.array([float(m[k]) for k in names])
            if not bool(m['update_accepted']) or not np.isfinite(vals).all():raise RuntimeError(f'Invalid update at epoch {epoch}')
            sums+=n*vals; count+=n
        row={'epoch':epoch,'phase':'physics' if active else 'pretrain','loss':float(sums[0]/count),**{'train_'+k:float(v) for k,v in zip(names[1:],sums[1:]/count)}}
        if epoch%args.validate_every==0 or epoch==args.epochs:
            vm,vpred=score(parameters,val); vm['mse_state']=float(np.mean((vpred[:,:4]-a['targets'][val,:4])**2))
            row.update({'validation_'+k:v for k,v in vm.items()})
        row['epoch_wall_seconds']=time.perf_counter()-epoch_started; history.append(row)
        if epoch==1 or epoch%args.print_every==0 or epoch==args.epochs:print(json.dumps(row),flush=True)
        if epoch%args.validate_every==0:save_json(output/'training_history.json',history)
    wall=time.perf_counter()-started; phase_times['physics_seconds']=time.perf_counter()-phase_started
    tm,pred=score(parameters,test); tm['mse_state']=float(np.mean((pred[:,:4]-a['targets'][test,:4])**2))
    with (model_dir/'parameter_tree.pkl').open('wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,parameters),f,pickle.HIGHEST_PROTOCOL)
    manifest={'schema_version':1,'task':cfg['task'],'model':'pinn','architecture':'anchored_local_flow_mlp_with_unprojected_rate_head','checkpoint_sha256':hashlib.sha256((model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(),'data_sha256':meta['data_sha256'],'input_size':8,'output_size':7,'network_input_size':9,'network_output_size':7,'parameter_count':cfg['parameter_count'],'projection':'none','sampling_interval':cfg['sampling_interval'],'training_wall_seconds':wall}
    save_json(model_dir/'checkpoint_manifest.json',manifest)
    np.savez(model_dir/'evaluation_data.npz',inputs=a['inputs'][test],targets=a['targets'][test],transition_id=test)
    np.savez(output/'predictions.npz',inputs=a['inputs'][test],exact=a['targets'][test],prediction=pred,transition_id=test)
    trajectory=np.flatnonzero((a['split']==2)&(a['sample_kind']==1))
    if args.smoke:trajectory=trajectory[:8]
    if len(trajectory):
        _,tpred=score(parameters,trajectory)
        fields={'inputs':a['inputs'][trajectory],'exact':a['targets'][trajectory],'prediction':tpred,'trajectory_id':a['trajectory_id'][trajectory],'step_index':a['step_index'][trajectory]}
        np.savez(output/'trajectory_predictions.npz',**fields)
        np.savez(model_dir/'trajectory_data.npz',inputs=fields['inputs'],targets=fields['exact'],trajectory_id=fields['trajectory_id'],step_index=fields['step_index'])
    tm.update({'algebraic_violation_max':float(np.max(abs(pred[:,4:]-reaction_rates(pred[:,:4])))),'test_transitions':len(test),'predicted_unsafe_transitions':int(np.sum(~admissible_state(pred[:,:4],cfg['plant_config']))),'true_unsafe_transitions':int(np.sum(~a['next_in_domain'][test])),'training_wall_seconds':wall,'closed_loop_MPC_validated':False,'trajectory_predictions':'one-step teacher-forced; not recursive validation'})
    saved=SavedPINNFlowModel(model_dir); x=a['inputs'][test[0]]
    np.testing.assert_allclose(saved(x),pred[0],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(saved.predict_jax(x,tau=0),np.r_[x[:4],reaction_rates(x[:4])],rtol=0,atol=1e-12)
    J=saved.linearize(x); eps=1e-5
    fd=np.column_stack([(saved(x+eps*np.eye(8)[i])-saved(x-eps*np.eye(8)[i]))/(2*eps) for i in range(8)])
    assert J.shape==(7,8) and np.isfinite(J).all()
    np.testing.assert_allclose(J,fd,rtol=2e-4,atol=2e-7)
    checks={'reload':'passed','hard_consistent_IC':'passed','raw_rate_head_connected':'passed','untrained_nonzero_algebraic_residual':'passed','finite_7_by_8_jacobian':'passed','jacobian_finite_difference_max_error':float(np.max(abs(J-fd)))}
    save_json(output/'metrics.json',tm); save_json(output/'verification.json',checks); save_json(output/'training_history.json',history)
    save_json(output/'training_summary.json',{'status':'trained','epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,'training_wall_seconds':wall,**phase_times,'final_history':history[-1]})
    print(json.dumps({'model_dir':str(model_dir),'output_dir':str(output),'metrics':tm,'checks':checks},indent=2),flush=True)


if __name__=='__main__':main()
