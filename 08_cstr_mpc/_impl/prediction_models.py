"""Matched predictor contracts; baseline equations use nominal kinetics only."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from surrogate import SavedFlowModel, rhs_jax

class EquationFlowModel(SavedFlowModel):
    def __init__(self,directory,kind='nonlinear',substeps=8):
        super().__init__(directory)
        if kind not in ('linear','nonlinear') or substeps<1:raise ValueError('Invalid predictor settings')
        self.kind=kind;self.substeps=int(substeps)
        c=self.config['plant_config'];self.origin=np.r_[c['nominal_state'],c['nominal_control'],c['nominal_disturbance']]
        if kind=='linear':
            z=jnp.asarray(self.origin);f=np.asarray(rhs_jax(z[:4],z))
            J=np.asarray(jax.jacfwd(lambda v:rhs_jax(v[:4],v))(z))
            generator=np.zeros((9,9));generator[:4,:8]=J;generator[:4,8]=f
            self.maps={tau:jnp.asarray(expm(generator*tau)[:4]) for tau in (self.config['sampling_interval'],self.config['sampling_interval']/2)}
    def predict_jax(self,inputs,tau=None):
        z=jnp.asarray(inputs,dtype=jnp.float64)
        t=self.config['sampling_interval'] if tau is None else float(tau)
        if self.kind=='linear':
            delta=z-jnp.asarray(self.origin)
            augmented=jnp.concatenate((delta,jnp.ones(delta.shape[:-1]+(1,))),axis=-1)
            state=jnp.asarray(self.origin[:4])+augmented@self.maps[t].T
        else:
            h=t/self.substeps
            def body(_,x):
                k1=rhs_jax(x,z);k2=rhs_jax(x+h*k1/2,z);k3=rhs_jax(x+h*k2/2,z);k4=rhs_jax(x+h*k3,z)
                return x+h*(k1+2*k2+2*k3+k4)/6
            state=jax.lax.fori_loop(0,self.substeps,body,z[...,:4])
        ca,cb,cc,th=(state[...,i] for i in range(4))
        rates=jnp.stack((.55*jnp.exp(.25*th)*ca,.08*jnp.exp(.20*th)*cb,.05*jnp.exp(.15*th)*cc),axis=-1)
        return jnp.concatenate((state,rates),axis=-1)

def make_predictor(directory,kind,substeps=8):
    return SavedFlowModel(directory) if kind=='daehn' else EquationFlowModel(directory,kind,substeps)
