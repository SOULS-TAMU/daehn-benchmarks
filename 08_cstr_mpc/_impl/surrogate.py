"""Differentiable learned CSTR flow with hard IC and native DAEHN rate decoding.

The continuous ODE is learned by supervised/RHS/physics losses, not imposed by
an implicit differential projection. Algebraic rates are exact. States are never
clipped to conceal an MPC operating-constraint violation.
"""

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
from model import make_algebraic_decoder,admissible_state

def rhs_jax(x,inputs):
    ca,cb,cc,th=(x[...,i] for i in range(4))
    phi,q,cai,thi=(inputs[...,i] for i in range(4,8))
    r1=0.55*jnp.exp(0.25*th)*ca;r2=0.08*jnp.exp(0.20*th)*cb;r3=0.05*jnp.exp(0.15*th)*cc
    a=0.2*phi
    return jnp.stack((a*(cai-ca)-r1,-a*cb+r1-r2,-a*cc+r2-r3,a*(thi-th)+0.06*r1+0.04*r2+0.02*r3+q),axis=-1)

def make_functions(config,batch_size):
    network=MLPBackbone(hidden_sizes=tuple(config['hidden_sizes']),activation=jnp.tanh)
    decode=make_algebraic_decoder(batch_size)
    mean=jnp.asarray(config['input_mean']);scale=jnp.asarray(config['input_scale'])
    vmean=jnp.asarray(config['velocity_mean']);vscale=jnp.asarray(config['velocity_scale'])
    ts=config['sampling_interval']
    def state(parameters,inputs,tau):
        features=jnp.concatenate(((inputs-mean)/scale,2*tau/ts-1),axis=-1)
        velocity=vmean+vscale*network.apply(parameters,features)
        return inputs[:,:4]+tau*velocity
    def prediction(parameters,inputs,tau):return decode(state(parameters,inputs,tau))
    def derivative(parameters,inputs,tau):
        return jax.jvp(lambda q:state(parameters,inputs,q),(tau,),(jnp.ones_like(tau),))[1]
    return network,prediction,derivative

class SavedFlowModel:
    """Reusable MPC interface: eight current-condition inputs -> seven next outputs."""
    def __init__(self,directory):
        directory=Path(directory)
        self.config=json.loads((directory/'run_config.json').read_text())
        manifest=json.loads((directory/'checkpoint_manifest.json').read_text())
        raw=(directory/'parameter_tree.pkl').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        self.parameters=jax.tree_util.tree_map(jnp.asarray,pickle.loads(raw))
        self._functions={}
    def _function(self,batch):
        if batch not in self._functions:
            _,fn,_=make_functions(self.config,batch)
            self._functions[batch]=jax.jit(fn)
        return self._functions[batch]
    def predict_jax(self,inputs,tau=None):
        inputs=jnp.asarray(inputs,dtype=jnp.float64)
        single=inputs.ndim==1
        if single:inputs=inputs[None,:]
        if inputs.ndim!=2 or inputs.shape[1]!=8:raise ValueError('Expected inputs of shape (8,) or (batch,8)')
        t=jnp.full((len(inputs),1),self.config['sampling_interval']) if tau is None else jnp.broadcast_to(jnp.asarray(tau), (len(inputs),1))
        result=self._function(len(inputs))(self.parameters,inputs,t)
        return result[0] if single else result
    def __call__(self,inputs):return np.asarray(self.predict_jax(inputs))
    def linearize(self,inputs):
        """Return 7-by-8 Jacobian in physical benchmark input coordinates."""
        value=jnp.asarray(inputs,dtype=jnp.float64)
        if value.shape!=(8,):raise ValueError('linearize expects one eight-input vector')
        return np.asarray(jax.jacrev(self.predict_jax)(value))
    def in_domain(self,inputs):
        inputs=np.asarray(inputs);c=self.config['plant_config']
        valid=admissible_state(inputs[...,:4],c)
        for offset,key in [(4,'control_bounds'),(6,'disturbance_bounds')]:
            b=np.asarray(c[key]);valid=valid&np.all((inputs[...,offset:offset+2]>=b[:,0])&(inputs[...,offset:offset+2]<=b[:,1]),axis=-1)
        return valid
