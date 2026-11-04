"""Train a native DAEHN battery DAE on public voltage/current measurements."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,pickle,hashlib,time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from daehn.train.projected_neural import Adam,apply_updates
from model import make_model,scalar_parameters,SavedBatteryModel,VARIANTS,state_names,pack_state
CASE=_CASE_DIR
RESULTS=CASE.parent/'results'/CASE.name
def save_json(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def windows(arrays,metadata,split,length):
    rows=[]
    for protocol in metadata['protocols']:
        if protocol['split']!=split: continue
        k=protocol['id']; n=len(arrays[f't_{k}'])
        for start in range(0,n-1,length-1): rows.append((k,start,min(n,start+length)))
    return rows

def batch_data(rows,a,length):
    values={key:[] for key in ['current','voltage','dt','charge','soc0','mask']}
    for k,start,stop in rows:
        count=stop-start; pad=length-count; ids=np.arange(start,stop)
        current=a[f'i_{k}'][ids]; voltage=a[f'v_{k}'][ids]; t=a[f't_{k}'][ids]
        values['current'].append(np.pad(current,(0,pad),mode='edge')); values['voltage'].append(np.pad(voltage,(0,pad),mode='edge'))
        values['dt'].append(np.pad(np.r_[0,np.diff(t)],(0,pad))); values['charge'].append(a[f'charge_{k}'][start]); values['soc0'].append(a[f'soc0_{k}'])
        mask=np.arange(length)<count; mask[0]=False; values['mask'].append(mask)
    return {k:jnp.asarray(v) for k,v in values.items()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=3000); parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--window-length',type=int,default=128); parser.add_argument('--hidden-size','--hidden-nodes',type=int,default=32); parser.add_argument('--hidden-layers',type=int,default=2)
    parser.add_argument('--activation',choices=['tanh','relu'],default='tanh'); parser.add_argument('--model',choices=VARIANTS,default='daehn')
    parser.add_argument('--fixed-capacity-ah',type=float,default=191.5)
    parser.add_argument('--hysteresis-rate',type=float,default=20.); parser.add_argument('--hysteresis-relaxation-seconds',type=float,default=10000.)
    parser.add_argument('--learning-rate',type=float,default=1e-3); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--study',default='new_runs'); parser.add_argument('--print-every',type=int,default=20)
    parser.add_argument('--validate-every',type=int,default=1,help='Validation/checkpoint interval, independent of printing')
    parser.add_argument('--steps-per-epoch',type=int,help='Optional smoke-test limit; default traverses every training window')
    parser.add_argument('--capacity-prior-weight',type=float,default=.001); parser.add_argument('--soc-bound-weight',type=float,default=10)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.hidden_size,args.hidden_layers,args.print_every,args.validate_every)<1 or args.window_length<3 or args.learning_rate<=0: parser.error('Invalid settings')
    if args.steps_per_epoch is not None and args.steps_per_epoch<1: parser.error('steps-per-epoch must be positive')
    if min(args.capacity_prior_weight,args.soc_bound_weight)<0: parser.error('Loss weights must be nonnegative')
    if not 160<args.fixed_capacity_ah<220 or min(args.hysteresis_rate,args.hysteresis_relaxation_seconds)<=0: parser.error('Invalid capacity/hysteresis settings')
    if Path(args.study).name!=args.study: parser.error('study must be a single folder name')
    with np.load(CASE/'data'/'data.npz') as archive: a={k:archive[k] for k in archive.files}
    meta=json.loads((CASE/'data'/'metadata.json').read_text())
    if hashlib.sha256((CASE/'data'/'data.npz').read_bytes()).hexdigest()!=meta['data_sha256']: raise ValueError('Data hash mismatch')
    config={'task':'battery_voltage_forecasting','model':'daehn','projection':'native_linear_voltage_equality','hidden_sizes':[args.hidden_size]*args.hidden_layers,'hysteresis_current_scale_A':.25,'batch_size':args.batch_size,'window_length':args.window_length,'epochs':args.epochs,'learning_rate':args.learning_rate,'seed':args.seed,'steps_per_epoch':args.steps_per_epoch,'capacity_prior_weight':args.capacity_prior_weight,'soc_bound_weight':args.soc_bound_weight,'time_discretization':'exponential RC step with frozen midpoint constitutive coefficient and trapezoidal current','state_order':['soc','polarization_V'],'algebraic_order':['terminal_voltage_V'],'input_contract':'initial SOC/polarization, current profile, times; no future voltage input','data_sha256':meta['data_sha256'],'training_initialization':'Each training window conditions on first observed voltage and SOC derived from accumulated charge. Loss excludes that anchor. Validation selection uses same conditional windows. Test evaluation is a full open-loop profile from its first observation only.'}
    config.update(variant=args.model,model=args.model,activation=args.activation,fixed_capacity_Ah=args.fixed_capacity_ah,hysteresis_rate=args.hysteresis_rate,hysteresis_relaxation_seconds=args.hysteresis_relaxation_seconds,state_order=state_names({'variant':args.model}),projection='none' if args.model=='voltage_mlp' else 'native_linear_voltage_equality',inequality_projection_added=False,validate_every=args.validate_every,print_every=args.print_every)
    init,rollout,_,_,_=make_model({**config,'training_gradients':True},a['ocv_soc'],a['ocv_voltage'],args.batch_size); p=init(args.seed)
    train=windows(a,meta,'train',args.window_length); val=windows(a,meta,'validation',args.window_length); rng=np.random.default_rng(args.seed)
    def objective(parameters,b):
        capacity,_,r0,h=scalar_parameters(parameters); soc=b['soc0']-b['charge']/(capacity*3600)
        initial_ocv=jnp.interp(soc,jnp.asarray(a['ocv_soc']),jnp.asarray(a['ocv_voltage']))
        offset=0. if 'hysteresis_state' in config['state_order'] else h*jnp.tanh(b['current'][:,0]/config['hysteresis_current_scale_A'])
        polarization=initial_ocv-r0*b['current'][:,0]-offset-b['voltage'][:,0]
        if len(config['state_order'])>2: polarization=jnp.concatenate((polarization[:,None],jnp.zeros((args.batch_size,len(config['state_order'])-2))),axis=-1)
        prediction=rollout(parameters,soc,polarization,b['current'],b['dt'])
        mask=b['mask']; denominator=jnp.maximum(jnp.sum(mask),1)
        mse=jnp.sum(((prediction[:,:,-1]-b['voltage'])/.05)**2*mask)/denominator
        bounds=jnp.maximum(-prediction[:,:,0],0)**2+jnp.maximum(prediction[:,:,0]-1,0)**2
        bound_loss=jnp.sum(bounds*mask)/denominator
        prior=((capacity-191.5)/20)**2
        return mse+args.soc_bound_weight*bound_loss+args.capacity_prior_weight*prior,(mse,bound_loss)
    optimizer=Adam(learning_rate=args.learning_rate); optstate=optimizer.init(p)
    @jax.jit
    def step(parameters,state,b):
        (loss,parts),grad=jax.value_and_grad(objective,has_aux=True)(parameters,b)
        updates,state=optimizer.update(grad,state,parameters)
        return apply_updates(parameters,updates),state,loss,parts
    evaluate=jax.jit(objective)
    def validation(parameters):
        losses=[]
        for start in range(0,len(val),args.batch_size):
            rows=val[start:start+args.batch_size]; n=len(rows); padded=rows+[rows[-1]]*(args.batch_size-n); b=batch_data(padded,a,args.window_length)
            b['mask']=b['mask'].at[n:].set(False); losses.append((float(evaluate(parameters,b)[1][0]),float(jnp.sum(b['mask']))))
        return float(sum(value*count for value,count in losses)/sum(count for _,count in losses))
    print('Measurements:',meta['measurement_counts'],'training windows:',len(train),'validation windows:',len(val),flush=True)
    history=[]; best=float('inf'); best_p=p; best_epoch=0; start_time=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        order=rng.permutation(len(train)); losses=[]
        for step_index,start in enumerate(range(0,len(order),args.batch_size)):
            if args.steps_per_epoch is not None and step_index>=args.steps_per_epoch: break
            ids=order[start:start+args.batch_size]; rows=[train[i] for i in ids]; n=len(rows); rows+= [rows[-1]]*(args.batch_size-n); b=batch_data(rows,a,args.window_length); b['mask']=b['mask'].at[n:].set(False)
            p,optstate,loss,parts=step(p,optstate,b)
            if not np.isfinite(float(loss)): raise RuntimeError('Nonfinite training loss')
            losses.append(float(loss))
        if epoch==1 or epoch%args.validate_every==0 or epoch==args.epochs:
            vl=validation(p); cap,c1,r0,h=map(float,scalar_parameters(p)); row={'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_scaled_voltage_mse':vl,'capacity_Ah':cap,'capacitance_F':c1,'series_resistance_ohm':r0,'hysteresis_V':h}; history.append(row)
            if vl<best: best=vl; best_p=jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),p); best_epoch=epoch
        if epoch==1 or epoch%args.print_every==0 or epoch==args.epochs:
            cap,c1,r0,h=map(float,scalar_parameters(p))
            print(json.dumps({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_scaled_voltage_mse':history[-1]['validation_scaled_voltage_mse'],'validation_epoch':history[-1]['epoch'],'capacity_Ah':cap,'capacitance_F':c1,'series_resistance_ohm':r0,'hysteresis_V':h}),flush=True)
    p=jax.tree_util.tree_map(jnp.asarray,best_p); wall=time.perf_counter()-start_time; config.update(selected_epoch=best_epoch,train_windows=len(train),validation_windows=len(val))
    config['parameter_count']=sum(x.size for x in jax.tree_util.tree_leaves(p)); config['capacity_trainable']=args.model!='fixed_capacity'
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    model_dir=RESULTS/'model'/args.study/args.model/f'seed={args.seed}__{signature}'; out=RESULTS/'output'/args.study/args.model/model_dir.name
    if (model_dir/'parameter_tree.pkl').exists(): raise FileExistsError('Completed model already exists; choose another study or settings')
    model_dir.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    with (model_dir/'parameter_tree.pkl').open('wb') as f: pickle.dump(jax.tree_util.tree_map(np.asarray,p),f,pickle.HIGHEST_PROTOCOL)
    save_json(model_dir/'run_config.json',config); np.savez(model_dir/'ocv_data.npz',soc=a['ocv_soc'],voltage=a['ocv_voltage'])
    save_json(model_dir/'checkpoint_manifest.json',{'schema_version':2,'task':config['task'],'model':args.model,'checkpoint_file':'parameter_tree.pkl','checkpoint_sha256':hashlib.sha256((model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(),'data_sha256':meta['data_sha256'],'training_wall_seconds':wall,'epochs_executed':args.epochs,'selected_epoch':best_epoch,'input_contract':config['input_contract'],'dae_index':None if args.model=='voltage_mlp' else 1,'projection':'none' if args.model=='voltage_mlp' else 'LinearEqualityProjection','differential_states':config['state_order'],'algebraic_states':[] if args.model=='voltage_mlp' else config['algebraic_order'],'parameter_count':config['parameter_count'],'activation':args.activation})
    save_json(out/'training_history.json',history); save_json(out/'training_summary.json',{'epochs_executed':args.epochs,'selected_epoch':best_epoch,'training_wall_seconds':wall,'protocol_split':meta['measurement_counts'],'smoke_step_limit':args.steps_per_epoch})
    from analysis import analyze
    metrics=analyze(model_dir,out)
    # Explicit reload and differentiable downstream state/current interface checks.
    saved=SavedBatteryModel(model_dir); probe_current=np.array([0.,50.,50.]); probe_time=np.array([0.,1.,2.]); probe=saved.predict(probe_current,probe_time,.5)
    def probe_step(state):
        aux=state[1:] if saved.state_dim==2 else state[None,1:]
        z,u=saved.transition(saved.parameters,state[0:1],aux,jnp.array([1.]),jnp.array([50.]),jnp.array([50.])); return pack_state(z,u)[0]
    jac=jax.jacfwd(probe_step)(jnp.r_[.5,jnp.zeros(saved.state_dim-1)])
    if not np.isfinite(probe).all() or not np.isfinite(np.asarray(jac)).all(): raise RuntimeError('Reload/Jacobian check failed')
    metrics['reload_and_state_jacobian_check']='passed'; save_json(out/'metrics.json',metrics)
    print(json.dumps({'model_dir':str(model_dir),'output_dir':str(out),'variant':args.model,'activation':args.activation,'selected_epoch':best_epoch,'test_voltage_rmse_V':metrics['test_voltage_rmse_V'],'reload_and_state_jacobian_check':'passed'},indent=2),flush=True)
if __name__=='__main__': main()
