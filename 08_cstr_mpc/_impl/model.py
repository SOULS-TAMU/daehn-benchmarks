"""State/control-conditioned CSTR physics and the MPC prediction contract."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp

STATE_NAMES=('C_A','C_B','C_C','theta')
CONTROL_NAMES=('phi','q')
DISTURBANCE_NAMES=('C_Ai','theta_i')
RATE_NAMES=('r1','r2','r3')
INPUT_NAMES=STATE_NAMES+CONTROL_NAMES+DISTURBANCE_NAMES
OUTPUT_NAMES=tuple(n+'_next' for n in STATE_NAMES+RATE_NAMES)
CONSTRAINT_NAMES=tuple(n+'_lower' for n in STATE_NAMES)+tuple(n+'_upper' for n in STATE_NAMES)+('inventory_upper',)

def load_config(path=None):
    config=json.loads(Path(path or (_CASE_DIR / 'config.json')).read_text(encoding='utf-8'))
    validate_config(config)
    return config

def validate_config(c):
    for name,size in [('state_bounds',4),('control_bounds',2),('disturbance_bounds',2)]:
        b=np.asarray(c[name],dtype=float)
        if b.shape!=(size,2) or not np.all(np.isfinite(b)) or np.any(b[:,0]>=b[:,1]):raise ValueError(f'Invalid {name}')
    if np.any(np.asarray(c['state_bounds'])[:3,0]!=0):raise ValueError('Concentration lower limits must be zero for this benchmark')
    if c['control_bounds'][0][0]<=0 or c['disturbance_bounds'][0][0]<0:raise ValueError('Flow must be positive and feed concentration nonnegative')
    if c['inventory_upper']<c['disturbance_bounds'][0][1]:raise ValueError('Inventory invariance requires the feed upper bound not exceed inventory_upper')
    if c['sampling_interval']<=0 or c['interval_nodes']<3 or c['steps_per_trajectory']<1:raise ValueError('Invalid time discretization')
    if c['independent_samples']<10 or c['linked_trajectories']<10:raise ValueError('Need at least ten groups of each type for separate train/validation/test splits')
    if len(c['control_move_limits'])!=2 or np.any(np.asarray(c['control_move_limits'])<=0):raise ValueError('Invalid control move limits')
    fractions=np.asarray(c['split_fractions'])
    if fractions.shape!=(3,) or np.any(fractions<=0) or not np.isclose(fractions.sum(),1):raise ValueError('Split fractions must be positive and sum to one')

def reaction_rates(x):
    x=np.asarray(x)
    return np.stack((0.55*np.exp(0.25*x[...,3])*x[...,0],0.08*np.exp(0.20*x[...,3])*x[...,1],0.05*np.exp(0.15*x[...,3])*x[...,2]),axis=-1)

def rhs(x,u,d):
    x,u,d=map(np.asarray,(x,u,d));r=reaction_rates(x)
    a=0.2*u[...,0]
    return np.stack((a*(d[...,0]-x[...,0])-r[...,0],-a*x[...,1]+r[...,0]-r[...,1],-a*x[...,2]+r[...,1]-r[...,2],a*(d[...,1]-x[...,3])+0.06*r[...,0]+0.04*r[...,1]+0.02*r[...,2]+u[...,1]),axis=-1)

def algebraic_residual(x,z):return np.asarray(z)-reaction_rates(x)

def state_constraint_values(x,c):
    x=np.asarray(x);bounds=np.asarray(c['state_bounds'])
    return np.concatenate((bounds[:,0]-x,x-bounds[:,1],(np.sum(x[...,:3],axis=-1)-c['inventory_upper'])[...,None]),axis=-1)

def admissible_state(x,c,tolerance=1e-9):return np.max(state_constraint_values(x,c),axis=-1)<=tolerance

def integrate_interval(x,u,d,c,*,method=None,rtol=None,atol=None):
    """Continuous-flow teacher: algebraic variables eliminated exactly, no clipping."""
    x,u,d=map(lambda a:np.asarray(a,dtype=np.float64),(x,u,d))
    if x.shape!=(4,) or u.shape!=(2,) or d.shape!=(2,):raise ValueError('Expected x(4), u(2), d(2)')
    if not np.all(np.isfinite(np.r_[x,u,d])):raise ValueError('Nonfinite plant input')
    tau=np.linspace(0,c['sampling_interval'],c['interval_nodes'])
    solver=c['solver']
    result=solve_ivp(lambda t,value:rhs(value,u,d),(0,float(tau[-1])),x,t_eval=tau,method=method or solver['method'],rtol=rtol or solver['rtol'],atol=atol or solver['atol'])
    if not result.success or result.y.shape!=(4,len(tau)) or not np.all(np.isfinite(result.y)):raise RuntimeError('Reference integration failed: '+result.message)
    values=result.y.T
    if values[:,:3].min() < -1e-9:raise RuntimeError('Numerically negative concentration; tighten/check reference integration')
    if np.sum(values[:,:3],axis=1).max()>c['inventory_upper']+1e-8:raise RuntimeError('Inventory invariant violated by reference trajectory')
    rates=reaction_rates(values)
    return {'local_time':tau,'states':values,'rates':rates,'derivatives':rhs(values,u,d),'violations':np.maximum(state_constraint_values(values,c),0),'next_state':values[-1],'next_rates':rates[-1],'solver_nfev':result.nfev}

def steady_state(u,d,c):
    """Solve a bounded thermal balance, analytically eliminating concentrations."""
    from scipy.optimize import brentq
    u,d=map(np.asarray,(u,d));a=0.2*u[0]
    def states(theta):
        k= np.array([0.55,0.08,0.05])*np.exp(np.array([0.25,0.20,0.15])*theta)
        ca=a*d[0]/(a+k[0]);cb=k[0]*ca/(a+k[1]);cc=k[1]*cb/(a+k[2])
        return np.array([ca,cb,cc,theta])
    lower,upper=c['state_bounds'][3]
    def heat(theta):return rhs(states(theta),u,d)[3]
    if heat(lower)*heat(upper)>0:return None
    value=states(brentq(heat,lower,upper,xtol=1e-12))
    return value if admissible_state(value,c) else None

def stage_cost(x_next,u,u_previous,c,reference=None):
    cfg=c['mpc'];x_next,u,u_previous=map(np.asarray,(x_next,u,u_previous))
    ref=cfg['product_B_reference'] if reference is None else reference
    tracking=((x_next[...,1]-ref)/cfg['concentration_tracking_scale'])**2+cfg['temperature_tracking_weight']*(x_next[...,3]/cfg['temperature_tracking_scale'])**2
    deviation=np.sum(((u-np.asarray(c['nominal_control']))/np.asarray(cfg['input_deviation_scales']))**2,axis=-1)
    movement=np.sum(((u-u_previous)/np.asarray(c['control_move_limits']))**2,axis=-1)
    return tracking+cfg['input_deviation_weight']*deviation+cfg['input_move_weight']*movement

def terminal_cost(x,c,reference=None):
    cfg=c['mpc'];ref=cfg['product_B_reference'] if reference is None else reference
    return cfg['terminal_tracking_weight']*(((np.asarray(x)[...,1]-ref)/cfg['concentration_tracking_scale'])**2+cfg['temperature_tracking_weight']*(np.asarray(x)[...,3]/cfg['temperature_tracking_scale'])**2)

def mpc_constraint_values(x_next,u,u_previous,c):
    u=np.asarray(u);b=np.asarray(c['control_bounds'])
    return np.concatenate((state_constraint_values(x_next,c),b[:,0]-u,u-b[:,1],np.abs(u-np.asarray(u_previous))-np.asarray(c['control_move_limits'])),axis=-1)

def make_algebraic_decoder(batch_size=1):
    """Native DAEHN rate projection: exact algebraic consistency, states unchanged.

    Differential flow-map learning/control constraints are separate obligations.
    This is an output reconstruction helper, not a trained prediction model.
    """
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from daehn.opt.layers.projection import LinearEqualityProjection
    A=np.column_stack((np.zeros((3,4)),np.eye(3)))
    projection=LinearEqualityProjection(A=A,b0=np.zeros(3),b_parameter_map=np.eye(3)).prepare(batch_size=batch_size,dtype=jnp.float64)
    def decode(states):
        states=jnp.asarray(states)
        r=jnp.stack((0.55*jnp.exp(0.25*states[...,3])*states[...,0],0.08*jnp.exp(0.20*states[...,3])*states[...,1],0.05*jnp.exp(0.15*states[...,3])*states[...,2]),axis=-1)
        target=jnp.concatenate((states,jnp.zeros(states.shape[:-1]+(3,),dtype=states.dtype)),axis=-1)
        return projection(target,r)
    return decode
