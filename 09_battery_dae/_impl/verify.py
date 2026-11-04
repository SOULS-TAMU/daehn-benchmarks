"""Numerical DAE consistency and differentiable-interface checks for a saved model."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from model import SavedBatteryModel,scalar_parameters,initial_state,continuous_rhs,pack_state

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model-dir',type=Path,required=True); args=parser.parse_args()
    saved=SavedBatteryModel(args.model_dir); p=saved.parameters; capacity,c1,r0,h=map(float,scalar_parameters(p)); rr=jax.jit(saved.r1)
    soc,current=np.meshgrid(np.linspace(0,1,21),np.linspace(-180,180,21)); resistance=np.asarray(rr(p,jnp.asarray(soc.ravel()),jnp.asarray(current.ravel())))
    assert resistance.min()>0 and min(capacity,c1,r0)>0
    zero=saved.predict(np.zeros(11),np.linspace(0,100,11),.6,.05); expected=.05*np.exp(-np.linspace(0,100,11)/(float(rr(p,jnp.array(.6),jnp.array(0.)))*c1))
    relaxation_error=float(np.max(abs(zero[:,1]-expected))); assert relaxation_error<1e-10
    def rhs(t,x): return np.asarray(continuous_rhs(saved.config,p,jnp.asarray(x),jnp.array(50.),rr(p,jnp.array(x[0]),jnp.array(50.))))
    initial=initial_state(saved.config,.8,.01); reference=solve_ivp(rhs,(0,300),initial,method='DOP853',rtol=1e-11,atol=1e-13)
    assert reference.success
    errors=[]
    for points in [31,61]:
        time=np.linspace(0,300,points); pred=saved.predict(np.full(points,50.),time,.8,.01); errors.append(float(np.linalg.norm(pred[-1,:-1]-reference.y[:,-1])))
    assert errors[1]<=errors[0]+1e-10
    def transition(x):
        aux=x[1:] if saved.state_dim==2 else x[None,1:]; z,u=saved.transition(p,x[:1],aux,jnp.array([2.]),jnp.array([50.]),jnp.array([50.])); return pack_state(z,u)[0]
    x=initial_state(saved.config,.6,.01); jac=np.asarray(jax.jacfwd(transition)(jnp.asarray(x))); eps=1e-6
    fd=np.column_stack([(np.asarray(transition(x+eps*np.eye(saved.state_dim)[k]))-np.asarray(transition(x-eps*np.eye(saved.state_dim)[k])))/(2*eps) for k in range(saved.state_dim)])
    jac_error=float(np.max(abs(jac-fd))); assert jac_error<1e-6
    # The public API accepts integer zero for initial polarization.
    assert np.isfinite(saved.predict([0,50,50],[0,1,2],.6,0)).all()
    result={'status':'passed','R1_min_ohm_on_SOC_current_grid':float(resistance.min()),'zero_current_exact_relaxation_max_error_V':relaxation_error,'independent_continuous_DAE_final_state_error_10s_and_5s_steps':errors,'state_jacobian_vs_finite_difference_max_error':jac_error,'integer_initial_polarization_interface':'passed','note':'Checks validate implementation and local temporal refinement, not general predictive accuracy.'}
    base=_CASE_DIR.parent/'results'/'09_battery_dae'; directory=args.model_dir.resolve()
    if not directory.is_relative_to(base/'model'): raise ValueError('Model outside this case')
    out=base/'output'/directory.relative_to(base/'model'); out.mkdir(parents=True,exist_ok=True); (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
