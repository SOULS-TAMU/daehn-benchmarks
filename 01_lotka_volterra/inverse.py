"""Blind Lotka--Volterra parameter estimation using the installed DAEHN library.

Only time and the two observed populations are read during fitting. Derivative,
invariant and parameter columns are deliberately ignored. The notebook's PINN
and projected derivative-consistency objectives are supported; an optional
integral-data term improves identification without supplying true coefficients.

Models and outputs follow the existing five-file, two-branch results policy.
See INVERSE.md for the notebook differences and commands.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

from daehn.dae import Coordinate,DifferentialState,Parameter,DerivativeSpec,EquationBlock,DAESystem,TaylorNeighborhood,ScalingConfig,compile_dae,ProjectedDAEModel
from daehn.opt import ProjectionConfig,ImplicitGradientConfig
from daehn.train.projected_neural import Adam,MLPBackbone,apply_updates

CASE=Path(__file__).resolve().parent
sys.path.insert(0,str(CASE.parent/'_shared'))
from artifacts import output_path
from project_paths import result_path
NAMES=('alpha','beta','gamma','delta')

def read_observations(path):
    table=np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8-sig')
    columns=table.dtype.names or ()
    selected=next((names for names in [('t','x','y'),('time','prey_x','predator_y')] if all(n in columns for n in names)),None)
    if selected is None:raise ValueError('CSV must contain t,x,y or time,prey_x,predator_y')
    a=np.column_stack([table[n] for n in selected]).astype(np.float64)
    a=a[np.argsort(a[:,0])]
    if len(a)<4 or not np.all(np.isfinite(a)) or np.any(np.diff(a[:,0])<=0):raise ValueError('Need finite observations at at least four distinct increasing times')
    if np.any(a[:,1:]<=0):raise ValueError('Population observations must be positive for the integral identity')
    return a[:,0:1],a[:,1:3],selected

def integral_design(t,y):
    """Identify coefficients from observations, without derivative labels."""
    dt=t[:,0]-t[0,0]
    ix=cumulative_trapezoid(y[:,0],t[:,0],initial=0)
    iy=cumulative_trapezoid(y[:,1],t[:,0],initial=0)
    ax=np.column_stack((dt,-iy));ay=np.column_stack((-dt,ix))
    bx=np.log(y[:,0]/y[0,0]);by=np.log(y[:,1]/y[0,1])
    if np.linalg.matrix_rank(ax)<2 or np.linalg.matrix_rank(ay)<2:raise ValueError('Observed trajectories do not identify all four parameters')
    estimate=np.r_[np.linalg.lstsq(ax,bx,rcond=None)[0],np.linalg.lstsq(ay,by,rcond=None)[0]]
    qx,rx=np.linalg.qr(ax,mode='reduced');qy,ry=np.linalg.qr(ay,mode='reduced')
    return estimate,tuple(jnp.asarray(a) for a in (ax,ay,bx,by,qx,rx,qy,ry))

def rhs(y,p):
    x,z=y[...,0],y[...,1]
    a,b,g,d=p
    return jnp.stack((a*x-b*x*z,-g*z+d*x*z),axis=-1)

def make_model(config,batch):
    backbone=MLPBackbone(hidden_sizes=tuple(config['hidden_sizes']),activation=jnp.tanh)
    t0=float(config['time_start']);span=float(config['time_span'])
    mean=jnp.asarray(config['state_mean']);scale=jnp.asarray(config['state_scale'])
    def raw(parameters,q):return mean+scale*backbone.apply(parameters['network'],2*(q-t0)/span-1)
    x=DifferentialState('x');y=DifferentialState('y')
    def equations(q,states,derivatives,p):return derivatives-rhs(states,p)
    system=DAESystem(coordinates=(Coordinate('t',t0,t0+span),),differential_states=(x,y),parameters=tuple(Parameter(n) for n in NAMES),derivatives=(DerivativeSpec(x,(1,)),DerivativeSpec(y,(1,))),differential_equations=EquationBlock(equations,size=2,affine_in_projection_variables=False),name='lotka_volterra_inverse')
    compiled=compile_dae(system,discretization=TaylorNeighborhood(deltas=(config['taylor_step'],),order=1),scaling=ScalingConfig(residual_mode='none'))
    prepared=compiled.prepare_projection(batch_size=batch,method='fb',config=ProjectionConfig(tolerance=config['tolerance'],max_iterations=config['projection_iterations'],backtracking=True,implicit_gradient=ImplicitGradientConfig(differentiate_problem_parameters=True,policy='residual_checked',inexact_tolerance=1e-7,adjoint_tolerance=1e-7)))
    model=ProjectedDAEModel(compiled,raw,prepared)
    def raw_derivatives(parameters,q):
        return jax.vmap(jax.jacfwd(lambda point:raw(parameters,point)))(q)[...,0]
    def forward(parameters,q):
        p=jnp.broadcast_to(parameters['coefficients'],(q.shape[0],4))
        decoded,info,*_=model.forward(parameters,q,p)
        return decoded['states'],decoded['derivatives'],info
    return backbone,raw,raw_derivatives,forward

def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=True)+'\n',encoding='utf-8')

def predict_payload(parameters,q,observed,config):
    _,raw,derivative,forward=make_model(config,len(q))
    fn=jax.jit(lambda p,t:forward(p,t)[:2]) if config['mode']=='daehn' else jax.jit(lambda p,t:(raw(p,t),derivative(p,t)))
    values,aux=fn(parameters,jnp.asarray(q))
    if config['mode']=='daehn':
        h=config['derivative_step']
        plus,_=fn(parameters,jnp.asarray(q)+h);minus,_=fn(parameters,jnp.asarray(q)-h)
        full=(plus-minus)/(2*h)
    else:full=aux
    residual=full-rhs(values,parameters['coefficients'])
    raw_d=derivative(parameters,jnp.asarray(q))
    payload={'coordinates':np.asarray(q),'exact':np.asarray(observed),'prediction':np.asarray(values),'model_derivatives':np.asarray(full),'projected_derivatives':np.asarray(aux),'differential_residual':np.asarray(residual),'raw_derivatives':np.asarray(raw_d)}
    return payload

def finish_outputs(parameters,config,model_dir,history,wall,source,columns,baseline):
    dest=output_path(model_dir);dest.mkdir(parents=True,exist_ok=True)
    t,y,_=read_observations(source)
    trajectory=np.linspace(t[0,0],t[-1,0],config['trajectory_samples'])[:,None]
    targets=np.column_stack([np.interp(trajectory[:,0],t[:,0],y[:,i]) for i in range(2)])
    np.savez(model_dir/'evaluation_data.npz',coordinates=t,targets=y)
    np.savez(model_dir/'trajectory_data.npz',coordinates=trajectory,targets=targets)
    host=jax.tree_util.tree_map(lambda a:np.asarray(a),parameters)
    with (model_dir/'parameter_tree.pkl').open('wb') as handle:pickle.dump(host,handle,pickle.HIGHEST_PROTOCOL)
    save_json(model_dir/'run_config.json',config)
    manifest={'schema_version':3,'task':'inverse','example':11,'model':config['mode'],'seed':config['seed'],'case_script':'inverse.py','parameter_names':NAMES,'source_csv':str(source),'source_csv_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'observed_columns':columns,'ignored_columns':'all other CSV columns','checkpoint_file':'parameter_tree.pkl','checkpoint_sha256':hashlib.sha256((model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(),'training_wall_seconds':wall,'derivative_evaluation':'symmetric finite difference of full projected model' if config['mode']=='daehn' else 'JAX automatic differentiation'}
    save_json(model_dir/'checkpoint_manifest.json',manifest)
    evaluation=predict_payload(parameters,t,y,config)
    trajectory_payload=predict_payload(parameters,trajectory,targets,config)
    np.savez(dest/'predictions.npz',**evaluation);np.savez(dest/'trajectory_predictions.npz',**trajectory_payload)
    estimates=np.asarray(parameters['coefficients'])
    metrics={'task':'inverse','mode':config['mode'],'estimated_parameters':dict(zip(NAMES,map(float,estimates))),'integral_data_estimate':dict(zip(NAMES,map(float,baseline))),'mse_data':float(np.mean((evaluation['prediction']-y)**2)),'continuous_physics_loss':float(np.mean(np.sum(evaluation['differential_residual']**2,axis=1))),'continuous_violation':float(np.mean(np.linalg.norm(evaluation['differential_residual'],axis=1))),'projected_auxiliary_violation':float(np.mean(np.linalg.norm(evaluation['projected_derivatives']-np.asarray(rhs(jnp.asarray(evaluation['prediction']),parameters['coefficients'])),axis=1))),'derivative_consistency_mse':float(np.mean(np.sum((evaluation['raw_derivatives']-evaluation['projected_derivatives'])**2,axis=1))),'training_wall_seconds':wall,'reference_parameters_used_for_fitting':False}
    if config['mode']=='pinn':
        metrics['projected_auxiliary_violation']=None
        metrics['derivative_consistency_mse']=None
    # Explicitly read reference metadata only AFTER optimization and prediction.
    if config['check_reference']:
        with np.load(CASE/'data/data.npz',allow_pickle=False) as a:reference=np.asarray(a['parameters'][0,:4])
        metrics['reference_comparison']={'reference':dict(zip(NAMES,map(float,reference))),'relative_errors':dict(zip(NAMES,map(float,np.abs(estimates-reference)/np.maximum(np.abs(reference),1e-12)))),'same_csv_as_forward_dataset':str(Path(source).resolve())==str((CASE/'data/trajectory.csv').resolve())}
    save_json(dest/'metrics.json',metrics);save_json(dest/'training_history.json',history)
    save_json(dest/'training_summary.json',{'status':'trained','epochs':config['epochs'],'pretrain_epochs':config['pretrain_epochs'],'training_wall_seconds':wall,'estimated_parameters':metrics['estimated_parameters']})
    print(json.dumps(metrics,indent=2))

def train(args):
    t,y,columns=read_observations(args.csv)
    baseline,design=integral_design(t,y)
    batch=len(t) if args.batch_size==0 else min(args.batch_size,len(t))
    config={'mode':args.mode,'epochs':args.epochs,'pretrain_epochs':args.pretrain_epochs,'seed':args.seed,'hidden_sizes':[args.hidden_size]*args.hidden_layers,'time_start':float(t[0,0]),'time_span':float(t[-1,0]-t[0,0]),'state_mean':y.mean(axis=0).tolist(),'state_scale':np.maximum(y.std(axis=0),1e-3).tolist(),'taylor_step':args.taylor_step,'derivative_step':args.derivative_step,'tolerance':args.tolerance,'projection_iterations':args.projection_iterations,'learning_rate':args.learning_rate,'data_weight':args.data_weight,'physics_weight':args.physics_weight,'integral_weight':args.integral_weight,'initialize':args.initialize,'batch_size':batch,'trajectory_samples':args.trajectory_samples,'check_reference':args.check_reference,'source_csv':str(args.csv.resolve())}
    backbone,raw,derivative,forward=make_model(config,batch)
    initial=baseline if args.initialize=='integral' else np.ones(4)
    parameters={'network':backbone.init(jax.random.key(args.seed),1,2,jnp.float64),'coefficients':jnp.asarray(initial)}
    optimizer=Adam(learning_rate=args.learning_rate);state=optimizer.init(parameters)
    ax,ay,bx,by,qx,rx,qy,ry=design
    def loss(p,q,target,projected):
        if projected and args.mode=='daehn':pred,aux,_=forward(p,q)
        else:pred=raw(p,q);aux=rhs(pred,p['coefficients'])
        dd=derivative(p,q)
        data=jnp.mean(jnp.sum((pred-target)**2,axis=1))
        physics=jnp.mean(jnp.sum((dd-aux)**2,axis=1))
        c=p['coefficients']
        # Whiten the identifiable moment directions using only the observation
        # matrix. This prevents correlated integral columns from slowing the
        # coefficient updates; no reference parameter values enter the loss.
        residual_x=jnp.linalg.solve(rx,qx.T@(ax@c[:2]-bx))
        residual_y=jnp.linalg.solve(ry,qy.T@(ay@c[2:]-by))
        integral=jnp.sum(residual_x**2)+jnp.sum(residual_y**2)
        total=args.data_weight*data+(args.physics_weight*physics+args.integral_weight*integral if projected else 0)
        return total,(data,physics,integral)
    def step(p,s,q,target,projected):
        (value,aux),grad=jax.value_and_grad(loss,has_aux=True)(p,q,target,projected)
        updates,s=optimizer.update(grad,s,p)
        return apply_updates(p,updates),s,value,aux
    # Separate compiled graphs: projected is a static branch, not a tracer.
    step=jax.jit(step,static_argnums=(4,))
    rng=np.random.default_rng(args.seed);history=[];start=time.perf_counter()
    for epoch in range(1,args.pretrain_epochs+args.epochs+1):
        active=epoch>args.pretrain_epochs
        indices=np.arange(len(t)) if batch==len(t) else rng.choice(len(t),batch,replace=False)
        parameters,state,total,parts=step(parameters,state,jnp.asarray(t[indices]),jnp.asarray(y[indices]),active)
        total=float(total);data,physics,integral=map(float,parts)
        coefficients=np.asarray(parameters['coefficients'])
        if not np.isfinite(total) or not np.all(np.isfinite(coefficients)):raise RuntimeError(f'Nonfinite training state at epoch {epoch}')
        record={'epoch':epoch,'phase':'inverse' if active else 'pretrain','loss_total':total,'loss_data':data,'loss_grad' if args.mode=='daehn' else 'loss_phys':physics,'loss_integral':integral,**dict(zip(NAMES,map(float,coefficients)))}
        history.append(record)
        if epoch==1 or epoch%args.print_every==0 or epoch==args.pretrain_epochs+args.epochs:print(f"{epoch:6d} data={data:.3e} derivative/physics={physics:.3e} params={coefficients}",flush=True)
    wall=time.perf_counter()-start
    study='inverse_smoke' if args.smoke else 'inverse'
    destination=result_path(Path('model')/study/args.mode/f'seed={args.seed}',__file__)
    destination.mkdir(parents=True,exist_ok=True)
    finish_outputs(parameters,config,destination,history,wall,args.csv,columns,baseline)

def load_saved(mode,seed,smoke=False):
    study='inverse_smoke' if smoke else 'inverse'
    path=result_path(Path('model')/study/mode/f'seed={seed}',__file__)
    config=json.loads((path/'run_config.json').read_text())
    manifest=json.loads((path/'checkpoint_manifest.json').read_text())
    if hashlib.sha256((path/'parameter_tree.pkl').read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
    with (path/'parameter_tree.pkl').open('rb') as handle:parameters=jax.tree_util.tree_map(jnp.asarray,pickle.load(handle))
    return path,config,parameters

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['train','predict','plot','report'],default='train')
    parser.add_argument('--csv',type=Path,default=CASE/'data/trajectory.csv')
    parser.add_argument('--mode',choices=['daehn','pinn'],default='daehn')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--pretrain-epochs',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=0,help='0: full observations, as in notebook; positive: sampled minibatch steps')
    parser.add_argument('--hidden-size',type=int,default=64);parser.add_argument('--hidden-layers',type=int,default=3)
    parser.add_argument('--learning-rate',type=float,default=1e-3)
    parser.add_argument('--data-weight',type=float,default=1);parser.add_argument('--physics-weight',type=float,default=1)
    parser.add_argument('--integral-weight',type=float,default=0,help='Optional observed-data integral identity term; no true parameters required')
    parser.add_argument('--initialize',choices=['ones','integral'],default='ones')
    parser.add_argument('--taylor-step',type=float,default=1e-3);parser.add_argument('--derivative-step',type=float,default=1e-3)
    parser.add_argument('--tolerance',type=float,default=1e-10);parser.add_argument('--projection-iterations',type=int,default=20)
    parser.add_argument('--trajectory-samples',type=int,default=1000);parser.add_argument('--print-every',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--check-reference',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Three updates with a tiny network; validates execution, not convergence')
    args=parser.parse_args()
    if args.epochs is None:args.epochs=50000 if args.mode=='daehn' else 10000
    if args.smoke:args.epochs=3;args.hidden_size=8;args.hidden_layers=1;args.batch_size=8;args.trajectory_samples=32;args.print_every=1
    if args.epochs<1 or args.pretrain_epochs<0 or args.batch_size<0 or args.print_every<1:parser.error('Invalid epoch, batch or logging settings')
    if args.action=='train':train(args)
    elif args.action=='report':
        study='inverse_smoke' if args.smoke else 'inverse'
        path=result_path(Path('model')/study/args.mode/f'seed={args.seed}',__file__)
        print((output_path(path)/'metrics.json').read_text(encoding='utf-8'))
    elif args.action=='predict':
        path,config,parameters=load_saved(args.mode,args.seed,args.smoke)
        for input_name,output_name in [('evaluation_data.npz','predictions.npz'),('trajectory_data.npz','trajectory_predictions.npz')]:
            with np.load(path/input_name) as a:payload=predict_payload(parameters,a['coordinates'],a['targets'],config)
            np.savez(output_path(path)/output_name,**payload)
        print('Predictions refreshed using saved parameters and input arrays.')
    else:
        import matplotlib.pyplot as plt
        path,_,_=load_saved(args.mode,args.seed,args.smoke);history=json.loads((output_path(path)/'training_history.json').read_text())
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for name in NAMES:axes[0].plot([r['epoch'] for r in history],[r[name] for r in history],label=name)
        axes[0].set_title('Blind parameter estimates');axes[0].legend()
        for key in ['loss_data','loss_grad' if args.mode=='daehn' else 'loss_phys']:axes[1].semilogy([r['epoch'] for r in history],[r[key] for r in history],label=key)
        axes[1].legend();axes[1].set_title('Training losses')
        with np.load(output_path(path)/'trajectory_predictions.npz') as a:
            for i,name in enumerate(['prey','predator']):axes[2].plot(a['coordinates'][:,0],a['exact'][:,i],label=name+' observed');axes[2].plot(a['coordinates'][:,0],a['prediction'][:,i],'--',label=name+' estimated')
        axes[2].legend();fig.tight_layout();plt.show()

if __name__=='__main__':main()
