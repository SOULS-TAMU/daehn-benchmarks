"""Unprojected conditional CSTR PINN using DAEHN's native MLP backbone."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import hashlib
import json
import pickle
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from daehn.train.projected_neural import MLPBackbone
from model import admissible_state


def rates_jax(x):
    return jnp.stack((.55*jnp.exp(.25*x[...,3])*x[...,0],
                      .08*jnp.exp(.20*x[...,3])*x[...,1],
                      .05*jnp.exp(.15*x[...,3])*x[...,2]),axis=-1)


def rhs_with_rates(states,rates,inputs):
    """DAE differential balance uses predicted rates, not eliminated rates."""
    ca,cb,cc,th=(states[...,i] for i in range(4))
    r1,r2,r3=(rates[...,i] for i in range(3))
    phi,q,cai,thi=(inputs[...,i] for i in range(4,8)); flow=.2*phi
    return jnp.stack((flow*(cai-ca)-r1,-flow*cb+r1-r2,-flow*cc+r2-r3,
                      flow*(thi-th)+.06*r1+.04*r2+.02*r3+q),axis=-1)


def make_functions(config,batch_size=None):
    del batch_size
    activation={'tanh':jnp.tanh,'relu':jax.nn.relu}[config.get('activation','tanh')]
    network=MLPBackbone(hidden_sizes=tuple(config['hidden_sizes']),activation=activation)
    mean=jnp.asarray(config['input_mean']); scale=jnp.asarray(config['input_scale'])
    velocity_mean=jnp.asarray(config['velocity_mean']); velocity_scale=jnp.asarray(config['velocity_scale'])
    rate_mean=jnp.asarray(config['rate_velocity_mean']); rate_scale=jnp.asarray(config['rate_velocity_scale'])
    ts=config['sampling_interval']
    def prediction(parameters,inputs,tau):
        features=jnp.concatenate(((inputs-mean)/scale,2*tau/ts-1),axis=-1)
        raw=network.apply(parameters,features)
        states=inputs[:,:4]+tau*(velocity_mean+velocity_scale*raw[:,:4])
        # Consistent IC only. For tau>0 rates are unconstrained network predictions.
        rates=rates_jax(inputs[:,:4])+tau*(rate_mean+rate_scale*raw[:,4:])
        return jnp.concatenate((states,rates),axis=-1)
    def derivative(parameters,inputs,tau):
        return jax.jvp(lambda t:prediction(parameters,inputs,t)[:,:4],
                       (tau,),(jnp.ones_like(tau),))[1]
    return network,prediction,derivative


class SavedPINNFlowModel:
    """Future comparison interface; raw rates are never projected or clipped."""
    def __init__(self,directory):
        directory=Path(directory)
        self.config=json.loads((directory/'run_config.json').read_text())
        manifest=json.loads((directory/'checkpoint_manifest.json').read_text())
        raw=(directory/'parameter_tree.pkl').read_bytes()
        if self.config['model']!='pinn':raise ValueError('Not a PINN checkpoint')
        if hashlib.sha256(raw).hexdigest()!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        self.parameters=jax.tree_util.tree_map(jnp.asarray,pickle.loads(raw))
        _,predict,_=make_functions(self.config)
        self._predict=jax.jit(predict)
    def predict_jax(self,inputs,tau=None):
        x=jnp.asarray(inputs,dtype=jnp.float64); single=x.ndim==1
        if single:x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=8:raise ValueError('Expected (8,) or (batch,8) inputs')
        t=jnp.broadcast_to(jnp.asarray(self.config['sampling_interval'] if tau is None else tau),(len(x),1))
        y=self._predict(self.parameters,x,t)
        return y[0] if single else y
    def __call__(self,inputs):return np.asarray(self.predict_jax(inputs))
    def linearize(self,inputs):
        x=jnp.asarray(inputs,dtype=jnp.float64)
        if x.shape!=(8,):raise ValueError('Expected one eight-input vector')
        return np.asarray(jax.jacrev(self.predict_jax)(x))
    def in_domain(self,inputs):
        x=np.asarray(inputs); c=self.config['plant_config']; valid=admissible_state(x[...,:4],c)
        for offset,key in [(4,'control_bounds'),(6,'disturbance_bounds')]:
            b=np.asarray(c[key]); valid=valid & np.all((x[...,offset:offset+2]>=b[:,0])&(x[...,offset:offset+2]<=b[:,1]),axis=-1)
        return valid
