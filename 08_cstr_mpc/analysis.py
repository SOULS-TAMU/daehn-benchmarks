"""Verify MPC-oriented CSTR input data or view its coverage; saves no extra files."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,hashlib,importlib.util,sys
import numpy as np
from model import load_config,reaction_rates,rhs,integrate_interval,state_constraint_values,admissible_state,make_algebraic_decoder,steady_state

CASE=_CASE_DIR

def verify(directory,reference_checks=64):
    metadata=json.loads((directory/'metadata.json').read_text())
    config=metadata['effective_config']
    if hashlib.sha256((directory/'data.npz').read_bytes()).hexdigest()!=metadata['data_sha256']:raise ValueError('Dataset hash mismatch')
    with np.load(directory/'data.npz',allow_pickle=False) as archive:a={name:archive[name] for name in archive.files}
    n=len(a['inputs']);nodes=config['interval_nodes']
    assert a['inputs'].shape==(n,8) and a['targets'].shape==(n,7)
    assert a['interval_states'].shape==(n,nodes,4)
    assert np.all(np.isfinite(a['inputs'])) and np.all(np.isfinite(a['targets']))
    assert np.all(admissible_state(a['inputs'][:,:4],config))
    assert np.array_equal(a['interval_states'][:,0,:],a['inputs'][:,:4])
    assert np.array_equal(a['interval_states'][:,-1,:],a['targets'][:,:4])
    assert np.max(np.abs(a['interval_rates']-reaction_rates(a['interval_states'])))<1e-12
    assert np.max(np.abs(a['targets'][:,4:]-reaction_rates(a['targets'][:,:4])))<1e-12
    assert np.max(np.abs(a['interval_derivatives']-rhs(a['interval_states'],a['inputs'][:,None,4:6],a['inputs'][:,None,6:8])))<1e-12
    violation=np.maximum(state_constraint_values(a['interval_states'],config),0)
    assert np.allclose(violation,a['interval_violations'],rtol=0,atol=1e-12)
    assert np.array_equal(violation.max(axis=(1,2))<=1e-9,a['interval_feasible_sampled'])
    assert np.array_equal(admissible_state(a['targets'][:,:4],config),a['next_in_domain'])
    for offset,key in [(4,'control_bounds'),(6,'disturbance_bounds')]:
        bounds=np.asarray(config[key]);values=a['inputs'][:,offset:offset+2]
        assert np.all(values>=bounds[:,0]-1e-12) and np.all(values<=bounds[:,1]+1e-12)
    assert np.all(np.abs(a['inputs'][:,4:6]-a['previous_control'])<=np.asarray(config['control_move_limits'])+1e-12)
    for group in np.unique(a['trajectory_id']):
        ids=np.flatnonzero(a['trajectory_id']==group)
        assert len(np.unique(a['split'][ids]))==1
        if a['sample_kind'][ids[0]]==1 and len(ids)>1:
            assert np.array_equal(a['targets'][ids[:-1],:4],a['inputs'][ids[1:],:4])
            assert np.array_equal(a['inputs'][ids[:-1],4:6],a['previous_control'][ids[1:]])
    train=a['split']==0
    normalization=metadata['normalization_training_only']
    assert np.allclose(normalization['input_mean'],a['inputs'][train].mean(axis=0))
    assert np.allclose(normalization['target_mean'],a['targets'][train].mean(axis=0))
    assert set(a['split'].tolist())=={0,1,2}
    sample=np.random.default_rng(901).choice(n,min(reference_checks,n),replace=False)
    max_reference_error=0.0
    for i in sample:
        ref=integrate_interval(a['inputs'][i,:4],a['inputs'][i,4:6],a['inputs'][i,6:8],config,method='Radau',rtol=1e-11,atol=1e-13)
        max_reference_error=max(max_reference_error,float(np.max(np.abs(ref['states']-a['interval_states'][i]))))
    assert max_reference_error<1e-7,max_reference_error
    # Confirm the copied coefficients match the original source, independently
    # of whether stored derivative labels match the new implementation.
    original=CASE.parent/'06_cstr/train.py'
    sys.path.insert(0,str(original.parent))
    spec=importlib.util.spec_from_file_location('_original_cstr_for_parity',original)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for i in sample[:16]:
        x=a['inputs'][i,:4];u=a['inputs'][i,4:6];d=a['inputs'][i,6:8]
        assert np.allclose(rhs(x,u,d),module.cstr_rhs_from_values(x,*u,*d),rtol=0,atol=1e-14)
    # Native DAEHN algebraic decoder must preserve states and differentiability.
    import jax
    import jax.numpy as jnp
    decoder=make_algebraic_decoder(1)
    outside=jnp.asarray([[0.2,0.3,0.1,2.3]])
    decoded=jax.jit(decoder)(outside)
    assert np.allclose(decoded[:,:4],outside)
    assert np.allclose(decoded[:,4:],reaction_rates(np.asarray(outside)),atol=1e-12)
    jac=np.asarray(jax.jacrev(lambda x:decoder(x[None,:])[0])(outside[0]))
    expected=np.zeros((7,4));expected[:4]=np.eye(4)
    r=reaction_rates(np.asarray(outside))[0];x=np.asarray(outside)[0]
    expected[4,0]=0.55*np.exp(0.25*x[3]);expected[5,1]=0.08*np.exp(0.20*x[3]);expected[6,2]=0.05*np.exp(0.15*x[3])
    expected[4:,3]=np.array([0.25,0.20,0.15])*r
    assert np.allclose(jac,expected,rtol=1e-10,atol=1e-12)
    # Check teacher control sensitivity: both manipulated inputs must influence flow.
    base=np.asarray(config['nominal_state']);u=np.asarray(config['nominal_control']);d=np.asarray(config['nominal_disturbance'])
    sensitivities=[]
    for j in range(2):
        h=1e-5;du=np.eye(2)[j]*h
        sensitivities.append((integrate_interval(base,u+du,d,config)['next_state']-integrate_interval(base,u-du,d,config)['next_state'])/(2*h))
    sensitivity=np.column_stack(sensitivities)
    assert np.all(np.linalg.norm(sensitivity,axis=0)>1e-6)
    report={'samples':n,'split_counts':metadata['split_counts'],'unsafe_intervals_retained':int(np.sum(~a['interval_feasible_sampled'])),'maximum_DOP853_Radau_difference':max_reference_error,'reference_checks':len(sample),'original_CSTR_equations_match':True,'group_split_leakage':False,'training_only_normalization_verified':True,'native_daehn_algebraic_projection_verified':True,'native_projection_preserves_unsafe_temperature':True,'native_projection_gradient_verified':True,'teacher_control_sensitivity':sensitivity.tolist(),'trained_surrogate_available':False,'closed_loop_MPC_validated':False}
    print(json.dumps(report,indent=2))
    return a,config

def reachability(config):
    cb=np.asarray(config['control_bounds']);dist=config['nominal_disturbance'];rows=[]
    for flow in np.linspace(*cb[0],31):
        for heat in np.linspace(*cb[1],31):
            x=steady_state([flow,heat],dist,config)
            if x is not None:rows.append((abs(x[1]-config['mpc']['product_B_reference']),flow,heat,x))
    rows.sort(key=lambda row:row[0])
    if not rows:raise RuntimeError('No admissible equilibrium found in control grid')
    best=rows[0]
    print(json.dumps({'nominal_disturbance':dist,'target':config['mpc']['product_B_reference'],'sampled_equilibrium_B_range':[min(r[3][1] for r in rows),max(r[3][1] for r in rows)],'closest_sampled_equilibrium':{'phi':best[1],'q':best[2],'state':best[3].tolist(),'concentration_error':best[0]},'note':'Grid reachability screen only; not an MPC or stability validation.'},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','plot','reachability'],nargs='?',default='verify')
    parser.add_argument('--data-dir',type=Path,default=CASE/'data');parser.add_argument('--reference-checks',type=int,default=64)
    args=parser.parse_args()
    if args.action=='reachability':reachability(load_config());return
    a,config=verify(args.data_dir,args.reference_checks)
    if args.action=='plot':
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for i,name in enumerate(['train','validation','test']):
            mask=a['split']==i;axes[0].scatter(a['inputs'][mask,4],a['inputs'][mask,5],s=3,label=name)
        axes[0].set(xlabel='phi',ylabel='q',title='Control coverage');axes[0].legend()
        axes[1].scatter(a['inputs'][:,3],a['targets'][:,3],s=3,c=~a['interval_feasible_sampled']);axes[1].set(xlabel='Current theta',ylabel='Next theta',title='Thermal transitions (coloured by violation)')
        axes[2].hist(a['targets'][:,1],bins=50);axes[2].axvline(config['mpc']['product_B_reference'],color='r');axes[2].set(xlabel='Next product-B concentration',title='Outlet target coverage')
        fig.tight_layout();plt.show()

if __name__=='__main__':main()
