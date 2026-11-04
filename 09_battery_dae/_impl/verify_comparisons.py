"""Verify saved comparison models, circuit contracts, Jacobians and extension limits."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *
from model import VARIANTS,state_names,continuous_rhs

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--suite-manifest',type=Path,required=True); args=p.parse_args()
    suite=json.loads(args.suite_manifest.read_text()); results=[]
    for run in suite['runs']:
        if 'model_dir' not in run: raise ValueError('Incomplete training suite')
        ctx=Context(run['model_dir'],'comparison_contract_checks'); saved=ctx.saved; x=initial_state(saved.config,.43,.01)
        if 'slow_polarization_V' in saved.state_names: x[2]=.005
        if 'hysteresis_state' in saved.state_names: x[-1]=.2
        current=np.array([0.,50.,-20.,0.]); t=np.array([0.,5.,15.,100.]); prediction=saved.predict(current,t,.43,.01)
        assert prediction.shape==(4,saved.state_dim+1) and np.isfinite(prediction).all()
        charge=np.r_[0,np.cumsum(.5*(current[:-1]+current[1:])*np.diff(t))]
        charge_error=float(np.max(abs(prediction[:,0]-(.43-charge/(ctx.capacity*3600)))))
        assert charge_error<1e-12
        jac=np.asarray(ctx.F(jnp.asarray(x),50.,5.)); eps=1e-6; fd=np.column_stack([(np.asarray(ctx.step(jnp.asarray(x+eps*np.eye(saved.state_dim)[k]),50.,5.))-np.asarray(ctx.step(jnp.asarray(x-eps*np.eye(saved.state_dim)[k]),50.,5.)))/(2*eps) for k in range(saved.state_dim)])
        jac_error=float(np.max(abs(jac-fd))); assert jac_error<1e-6
        H=np.asarray(ctx.H(jnp.asarray(x),50.)); Hfd=np.array([(float(ctx.measure(jnp.asarray(x+eps*np.eye(saved.state_dim)[k]),50.))-float(ctx.measure(jnp.asarray(x-eps*np.eye(saved.state_dim)[k]),50.)))/(2*eps) for k in range(saved.state_dim)])
        measure_error=float(np.max(abs(H-Hfd))); assert measure_error<1e-6
        reference=np.asarray(circuit_voltage(saved.config,saved.ox,saved.oy,ctx.p,jnp.asarray(prediction[:,:-1]),jnp.asarray(current))); residual=float(np.max(abs(prediction[:,-1]-reference)))
        if run['model']!='voltage_mlp': assert residual<1e-12
        else: assert residual>1e-7 # Readout genuinely bypasses the circuit equality.
        if run['model']=='ecm':
            resistance=np.asarray(saved.r1(ctx.p,jnp.array([.2,.8]),jnp.array([-50.,180.]))); assert resistance[0]==resistance[1] and 'network' not in ctx.p
        if run['model']=='fixed_capacity': assert abs(ctx.capacity-saved.config['fixed_capacity_Ah'])<1e-12
        if run['model']=='detached_ocv':
            _,_,v,_,_=make_model({**saved.config,'training_gradients':True},saved.ox,saved.oy,1)
            training_H=float(jax.grad(lambda z:v(ctx.p,jnp.array([z]),jnp.array([.01]),jnp.array([50.]))[0])(.43))
            assert abs(training_H)<1e-12 and abs(H[0])>1e-5
        if 'hysteresis_state' in saved.state_names: assert np.max(abs(prediction[:,saved.state_names.index('hysteresis_state')]))<=1+1e-12
        if 'slow_polarization_V' in saved.state_names:
            rest=saved.predict(np.zeros(3),np.array([0.,20.,40.]),.43,.01)
            assert np.max(abs(rest[:,2]))<1e-12 and rest[-1,1]<rest[0,1]
        checksum=hashlib.sha256((ctx.model_dir/'parameter_tree.pkl').read_bytes()).hexdigest()
        results.append({'variant':run['model'],'activation':saved.config.get('activation','tanh'),'state_dimension':saved.state_dim,'charge_error':charge_error,'transition_jacobian_fd_error':jac_error,'measurement_jacobian_fd_error':measure_error,'circuit_reference_residual_V':residual,'checkpoint_sha256':checksum,'status':'passed'})
    destination=args.suite_manifest.parent/'verification.json'; save_json(destination,{'status':'passed','checks':results,'inequality_projection_added':False,'note':'Short-run implementation checks; not convergence or application validation.'}); print(destination)

if __name__=='__main__': main()
