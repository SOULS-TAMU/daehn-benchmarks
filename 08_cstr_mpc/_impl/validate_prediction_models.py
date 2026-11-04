"""Independent flow accuracy, RK convergence and warmed single-prediction timing."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import json,time
from scipy.optimize import linprog
import numpy as np
import jax,jax.numpy as jnp
from prediction_models import make_predictor
from control import DEFAULT_MODEL
from model import integrate_interval,admissible_state
from compare_prediction_models import ROOT

def main():
    models={k:make_predictor(DEFAULT_MODEL,k) for k in ['daehn','linear','nonlinear']}
    rng=np.random.default_rng(812);c=models['daehn'].config['plant_config']
    inputs=[];truth=[]
    for i in range(128):
        x=np.r_[rng.dirichlet([2,2,2])*rng.uniform(.5,1.1),rng.uniform(-1,1)]
        u=rng.uniform([.4,-.4],[2,.4]);d=rng.uniform([.8,-.2],[1.2,.2]);inputs.append(np.r_[x,u,d])
        truth.append(integrate_interval(x,u,d,c)['next_state'])
    inputs=np.array(inputs);truth=np.array(truth);scores={};predictions={}
    for name,model in models.items():
        z=jnp.asarray(inputs[0]);model(z);fn=jax.jit(model.predict_jax);fn(z).block_until_ready()
        times=[]
        for i in range(100):
            start=time.perf_counter();fn(z).block_until_ready();times.append(time.perf_counter()-start)
        values=np.array([np.asarray(fn(jnp.asarray(v)))[:4] for v in inputs]);predictions[name]=values
        J=np.asarray(jax.jacfwd(model.predict_jax)(z))[:4];eps=1e-5
        fd=np.column_stack([(np.asarray(fn(z+jnp.eye(8)[i]*eps))[:4]-np.asarray(fn(z-jnp.eye(8)[i]*eps))[:4])/(2*eps) for i in range(8)])
        scores[name]=dict(state_rmse=float(np.sqrt(np.mean((values-truth)**2))),max_absolute_error=float(np.max(np.abs(values-truth))),warm_single_inference_median_seconds=float(np.median(times)),jacobian_fd_max_error=float(np.max(np.abs(J-fd))))
        assert scores[name]['jacobian_fd_max_error']<1e-6
    # Independent raw recursive prediction check: eight intervals, fixed inputs.
    recursive_truth=[]
    for z in inputs[:32]:
        state=z[:4].copy();trajectory=[]
        for _ in range(8):
            state=integrate_interval(state,z[4:6],z[6:8],c)['next_state'];trajectory.append(state)
        recursive_truth.append(trajectory)
    recursive_truth=np.asarray(recursive_truth)
    for name,model in models.items():
        fn=jax.jit(model.predict_jax);rollouts=[]
        for z in inputs[:32]:
            state=z[:4].copy();trajectory=[]
            for _ in range(8):
                state=np.asarray(fn(jnp.asarray(np.r_[state,z[4:]])))[:4];trajectory.append(state)
            rollouts.append(trajectory)
        predicted=np.asarray(rollouts);predictions[name+'_recursive']=predicted
        scores[name]['raw_recursive_8_step_rmse']=float(np.sqrt(np.mean((predicted-recursive_truth)**2)))
        scores[name]['raw_recursive_rmse_by_step']=np.sqrt(np.mean((predicted-recursive_truth)**2,axis=(0,2))).tolist()
        scores[name]['recursive_predicted_outside_physical_bounds']=int(np.sum(~admissible_state(predicted,c)))
    predictions['recursive_reference_states']=recursive_truth
    predictions['recursive_inputs']=inputs[:32]
    convergence={}
    for n in [4,8,16]:
        model=make_predictor(DEFAULT_MODEL,'nonlinear',n);fn=jax.jit(model.predict_jax)
        v=np.array([np.asarray(fn(jnp.asarray(z)))[:4] for z in inputs]);convergence[n]=float(np.max(np.abs(v-truth)))
    assert convergence[8]<1e-7 and convergence[16]<convergence[8]
    linear=models['linear'];z=linear.origin
    M=np.asarray(jax.jacfwd(linear.predict_jax)(jnp.asarray(z)))[:4];f=np.asarray(linear(z))[:4]
    eq=np.c_[np.eye(4)-M[:,:4],-M[:,4:6]];b=f-M[:,:6]@z[:6];reachable=[]
    for sign in [1,-1]:
        sol=linprog([0,sign,0,0,0,0],A_eq=eq,b_eq=b,A_ub=[[1,1,1,0,0,0]],b_ub=[1.2],bounds=c['state_bounds']+c['control_bounds'],method='highs')
        assert sol.success;reachable.append(float(sol.x[1]))
    result=dict(recursive_validation=dict(samples=32,steps=8,constant_inputs=True,reference_outside_physical_bounds=int(np.sum(~admissible_state(recursive_truth,c)))),fixed_linear_nominal_feed_steady_B_range=reachable,samples=128,seed=812,sampling_interval_minutes=.25,predictors=scores,nonlinear_RK4_substeps_max_error=convergence,notes='Random physical-domain one-step tests. Timings include synchronized JAX dispatch, exclude compilation, and do not imply MPC speed. Nominal equation model is an oracle baseline for the nominal simulation plant.')
    output=ROOT/'prediction_validation';output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez(output/'trajectory_predictions.npz',inputs=inputs,reference_next_states=truth,**predictions)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
