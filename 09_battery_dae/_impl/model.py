"""Passive grey-box index-1 battery DAE with native DAEHN voltage projection."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import json,pickle,hashlib
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from daehn.train.projected_neural import MLPBackbone
from daehn.opt.layers.projection import LinearEqualityProjection
from daehn.dae import Coordinate,DifferentialState,AlgebraicState,Parameter,DerivativeSpec,EquationBlock,DAESystem

def make_battery_system(ocv_soc,ocv_voltage):
    """DAE declaration; learning supplies positive constitutive parameters."""
    t=Coordinate('time_s',0.,1e9); soc=DifferentialState('soc'); pol=DifferentialState('polarization_V'); voltage=AlgebraicState('terminal_voltage_V')
    params=tuple(Parameter(n) for n in ['current_A','capacity_C','capacitance_F','R1_ohm','R0_ohm','hysteresis_V'])
    ox=jnp.asarray(ocv_soc); oy=jnp.asarray(ocv_voltage)
    def differential(q,x,d,p):
        i,capacity,c1,r1,r0,h=p
        return jnp.array([d[0]+i/capacity,d[1]-(i-x[1]/r1)/c1])
    def algebraic(q,x,d,p):
        i,capacity,c1,r1,r0,h=p
        return jnp.array([x[2]-jnp.interp(x[0],ox,oy)+x[1]+r0*i+h*jnp.tanh(i/.25)])
    return DAESystem(name='passive_battery_equivalent_circuit',coordinates=(t,),differential_states=(soc,pol),algebraic_states=(voltage,),parameters=params,derivatives=(DerivativeSpec(soc,(1,)),DerivativeSpec(pol,(1,))),differential_equations=EquationBlock(differential,size=2,affine_in_projection_variables=True),algebraic_equalities=EquationBlock(algebraic,size=1,affine_in_projection_variables=False))

def scalar_parameters(p):
    # Bounded capacity prior, strictly positive passive circuit coefficients.
    capacity=jax.lax.stop_gradient(p['capacity_fixed']) if 'capacity_fixed' in p else 160+60*jax.nn.sigmoid(p['capacity_logit'])
    capacitance=1e5*jax.nn.softplus(p['capacitance_raw'])+1
    resistance=.001*jax.nn.softplus(p['resistance_raw'])+1e-7
    hysteresis=.05*jnp.tanh(p['hysteresis_raw'])
    return capacity,capacitance,resistance,hysteresis

VARIANTS=('daehn','ecm','voltage_mlp','fixed_capacity','detached_ocv','two_rc','dynamic_hysteresis','two_rc_hysteresis')

def state_names(config):
    variant=config.get('variant','daehn')
    names=['soc','polarization_V']
    if variant in ('two_rc','two_rc_hysteresis'): names.append('slow_polarization_V')
    if variant in ('dynamic_hysteresis','two_rc_hysteresis'): names.append('hysteresis_state')
    return names

def pack_state(soc,aux):
    return jnp.concatenate((soc[...,None],aux[...,None] if aux.ndim==soc.ndim else aux),axis=-1)

def initial_state(config,soc,polarization=0.):
    state=np.zeros(len(state_names(config))); state[0]=soc; state[1]=polarization; return state

def circuit_voltage(config,ox,oy,p,state,current):
    """Circuit equality reference, including additional RC/hysteresis states."""
    _,_,r0,h=scalar_parameters(p); names=state_names(config)
    pol=state[...,1]
    if 'slow_polarization_V' in names: pol=pol+state[...,names.index('slow_polarization_V')]
    hyst=state[...,names.index('hysteresis_state')] if 'hysteresis_state' in names else jnp.tanh(current/config['hysteresis_current_scale_A'])
    ocv=jnp.interp(state[...,0],jnp.asarray(ox),jnp.asarray(oy))
    if config.get('variant')=='detached_ocv' and config.get('training_gradients',False): ocv=jax.lax.stop_gradient(ocv)
    return ocv-pol-r0*current-h*hyst

def continuous_rhs(config,p,state,current,r1):
    capacity,c1,_,_=scalar_parameters(p); names=state_names(config)
    parts=[-current/(capacity*3600),(current-state[...,1]/r1)/c1]
    if 'slow_polarization_V' in names:
        r2=.001*jax.nn.softplus(p['r2_raw'])+1e-7; c2=1e6*jax.nn.softplus(p['c2_raw'])+1
        parts.append((current-state[...,2]/r2)/c2)
    if 'hysteresis_state' in names:
        rate=config.get('hysteresis_rate',20.)*jnp.abs(current)/(capacity*3600)+1/config.get('hysteresis_relaxation_seconds',10000.)
        parts.append(rate*(jnp.tanh(current/config['hysteresis_current_scale_A'])-state[...,-1]))
    return jnp.stack(parts,axis=-1)

def make_model(config,ocv_soc,ocv_voltage,batch_size):
    activation={'tanh':jax.nn.tanh,'relu':jax.nn.relu}[config.get('activation','tanh')]
    network=MLPBackbone(hidden_sizes=tuple(config['hidden_sizes']),activation=activation)
    variant=config.get('variant','daehn'); names=state_names(config); extra_rc='slow_polarization_V' in names; dynamic_h='hysteresis_state' in names
    projection=LinearEqualityProjection(A=np.ones((1,1)),b0=np.zeros(1),b_parameter_map=np.ones((1,1))).prepare(batch_size=batch_size,dtype=jnp.float64)
    ox=jnp.asarray(ocv_soc); oy=jnp.asarray(ocv_voltage)
    def ocv(soc): return jnp.interp(soc,ox,oy)
    def r1(p,soc,current):
        if variant=='ecm': return jnp.full_like(soc,.001*jax.nn.softplus(p['r1_raw'])+2e-5)
        features=jnp.stack((2*soc-1,current/180),axis=-1)
        value=network.apply(p['network'],features)[...,0]
        return .001*(jax.nn.softplus(value)+.02)
    def voltage(p,soc,polarization,current):
        state=pack_state(soc,polarization)
        if variant=='voltage_mlp':
            features=jnp.stack((2*soc-1,state[...,1]/.1,current/180),axis=-1)
            return 3.3+.3*network.apply(p['readout_network'],features)[...,0]
        rhs=circuit_voltage(config,ox,oy,p,state,current)
        return projection(jnp.zeros((batch_size,1)),rhs[:,None])[:,0]
    def transition(p,soc,polarization,dt,previous_current,current):
        capacity,c1,_,_=scalar_parameters(p); average=.5*(previous_current+current)
        next_soc=soc-average*dt/(capacity*3600)
        midpoint=.5*(soc+next_soc); resistance=r1(p,midpoint,average)
        exponent=-dt/(resistance*c1); decay=jnp.exp(exponent)
        auxiliary=polarization[:,None] if polarization.ndim==1 else polarization
        first=decay*auxiliary[:,0]-resistance*average*jnp.expm1(exponent)
        parts=[first]
        if extra_rc:
            r2=.001*jax.nn.softplus(p['r2_raw'])+1e-7; c2=1e6*jax.nn.softplus(p['c2_raw'])+1
            exp2=-dt/(r2*c2); parts.append(jnp.exp(exp2)*auxiliary[:,1]-r2*average*jnp.expm1(exp2))
        if dynamic_h:
            rate=config.get('hysteresis_rate',20.)*jnp.abs(average)/(capacity*3600)+1/config.get('hysteresis_relaxation_seconds',10000.)
            exp_h=-dt*rate; target=jnp.tanh(average/config['hysteresis_current_scale_A'])
            parts.append(jnp.exp(exp_h)*auxiliary[:,-1]-target*jnp.expm1(exp_h))
        next_polarization=parts[0] if len(parts)==1 else jnp.stack(parts,axis=-1)
        return next_soc,next_polarization
    def rollout(p,initial_soc,initial_polarization,current,dt):
        first_voltage=voltage(p,initial_soc,initial_polarization,current[:,0])
        def step(carry,values):
            soc,pol=carry; duration,previous_i,next_i=values
            ns,np_=transition(p,soc,pol,duration,previous_i,next_i)
            output=jnp.concatenate((pack_state(ns,np_),voltage(p,ns,np_,next_i)[...,None]),axis=-1)
            return (ns,np_),output
        _,tail=jax.lax.scan(step,(initial_soc,initial_polarization),(dt[:,1:].T,current[:,:-1].T,current[:,1:].T))
        first=jnp.concatenate((pack_state(initial_soc,initial_polarization),first_voltage[...,None]),axis=-1)
        return jnp.concatenate((first[:,None,:],jnp.swapaxes(tail,0,1)),axis=1)
    def init(seed):
        inverse_softplus=lambda x:np.log(np.expm1(x))
        p={'capacity_logit':jnp.array(np.log(31.5/28.5)),'capacitance_raw':jnp.array(inverse_softplus(1.0)),'resistance_raw':jnp.array(inverse_softplus(.28)),'hysteresis_raw':jnp.array(np.arctanh(.015/.05))}
        if variant=='ecm': p['r1_raw']=jnp.array(inverse_softplus(.5))
        else: p['network']=network.init(jax.random.key(seed),2,1,jnp.float64)
        if variant=='voltage_mlp': p['readout_network']=network.init(jax.random.fold_in(jax.random.key(seed),1),3,1,jnp.float64)
        if variant=='fixed_capacity': p['capacity_fixed']=jnp.array(config.get('fixed_capacity_Ah',191.5)); p.pop('capacity_logit')
        if extra_rc: p.update(r2_raw=jnp.array(inverse_softplus(.2)),c2_raw=jnp.array(inverse_softplus(2.)))
        return p
    return init,rollout,voltage,transition,r1

class SavedBatteryModel:
    """State/current-conditioned flow, requiring explicit initial latent state."""
    def __init__(self,directory):
        path=Path(directory); self.config=json.loads((path/'run_config.json').read_text())
        manifest=json.loads((path/'checkpoint_manifest.json').read_text())
        if hashlib.sha256((path/'parameter_tree.pkl').read_bytes()).hexdigest()!=manifest['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
        with (path/'parameter_tree.pkl').open('rb') as f: self.parameters=jax.tree_util.tree_map(jnp.asarray,pickle.load(f))
        with np.load(path/'ocv_data.npz') as a: self.ox=a['soc']; self.oy=a['voltage']
        _,self.rollout,self.voltage,self.transition,self.r1=make_model(self.config,self.ox,self.oy,1)
        self.state_names=state_names(self.config); self.state_dim=len(self.state_names)
    def predict(self,current,time,initial_soc,initial_polarization=0):
        current=np.asarray(current,float); time=np.asarray(time,float)
        if current.ndim!=1 or time.shape!=current.shape or len(time)<2 or np.any(np.diff(time)<=0): raise ValueError('Need equal 1D arrays and strictly increasing times')
        if not np.isfinite(current).all() or not np.isfinite(time).all() or not np.isfinite([initial_soc,initial_polarization]).all(): raise ValueError('Inputs must be finite')
        aux=initial_state(self.config,initial_soc,initial_polarization)[1:]
        aux=jnp.asarray(aux[None,:] if self.state_dim>2 else aux,dtype=jnp.float64)
        return np.asarray(jax.jit(self.rollout)(self.parameters,jnp.array([initial_soc],dtype=jnp.float64),aux,jnp.asarray(current[None,:]),jnp.asarray(np.r_[0,np.diff(time)][None,:])))[0]
