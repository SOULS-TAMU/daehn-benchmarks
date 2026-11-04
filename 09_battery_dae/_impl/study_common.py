"""Shared saved-model/data contracts for the battery case-study campaign."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import SavedBatteryModel,scalar_parameters,make_model,initial_state,pack_state,circuit_voltage
CASE=_CASE_DIR
BASE=CASE.parent/'results'/CASE.name
DEFAULT_MODEL=BASE/'model'/'smoke_100_full'/'daehn'/'seed=42__59620738988d'

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(np.integer,np.floating)): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value
def save_json(path,value): Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf-8')
def rmse(value): return float(np.sqrt(np.mean(np.asarray(value)**2)))
def error_metrics(error):
    e=np.asarray(error)
    if not len(e): return {'samples':0}
    return {'samples':len(e),'rmse_V':rmse(e),'mae_V':float(np.mean(abs(e))),'bias_V':float(np.mean(e)),'p95_abs_error_V':float(np.quantile(abs(e),.95)),'max_abs_error_V':float(np.max(abs(e))),'fraction_above_50mV':float(np.mean(abs(e)>.05)),'fraction_above_100mV':float(np.mean(abs(e)>.1))}
def parser(description):
    p=argparse.ArgumentParser(description=description); p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL); p.add_argument('--study',default='case_study_v1'); return p

class Context:
    def __init__(self,model_dir=DEFAULT_MODEL,study='case_study_v1'):
        self.model_dir=Path(model_dir).resolve()
        if not self.model_dir.is_relative_to((BASE/'model').resolve()): raise ValueError('Model must be in this case model branch')
        if Path(study).name!=study: raise ValueError('study must be one folder name')
        self.saved=SavedBatteryModel(self.model_dir); self.p=self.saved.parameters
        self.meta=json.loads((CASE/'data'/'metadata.json').read_text())
        if self.saved.config['data_sha256']!=self.meta['data_sha256']: raise ValueError('Checkpoint/data mismatch')
        if hashlib.sha256((CASE/'data'/'data.npz').read_bytes()).hexdigest()!=self.meta['data_sha256']: raise ValueError('Dataset hash mismatch')
        with np.load(CASE/'data'/'data.npz') as a: self.a={k:a[k] for k in a.files}
        self.root=BASE/'output'/study/self.model_dir.relative_to(BASE/'model'); self.root.mkdir(parents=True,exist_ok=True)
        self.capacity,self.c1,self.r0,self.h=map(float,scalar_parameters(self.p))
        self.step=jax.jit(self._step); self.measure=jax.jit(self._measure)
        self.F=jax.jit(jax.jacfwd(self._step,argnums=0)); self.H=jax.jit(jax.jacfwd(self._measure,argnums=0))
    def output(self,task):
        path=self.root/task; path.mkdir(parents=True,exist_ok=True); return path
    def protocols(self,split=None):
        for m in self.meta['protocols']:
            if split and m['split']!=split: continue
            k=m['id']; yield {**m,'t':self.a[f't_{k}'],'i':self.a[f'i_{k}'],'v':self.a[f'v_{k}'],'charge':self.a[f'charge_{k}'],'soc0':float(self.a[f'soc0_{k}'])}
    def _step(self,x,average_current,dt):
        aux=x[1:] if self.saved.state_dim==2 else x[None,1:]
        z,u=self.saved.transition(self.p,x[:1],aux,jnp.array([dt]),jnp.array([average_current]),jnp.array([average_current])); return pack_state(z,u)[0]
    def _measure(self,x,current):
        aux=x[1:] if self.saved.state_dim==2 else x[None,1:]
        return self.saved.voltage(self.p,x[:1],aux,jnp.array([current]))[0]
    def initial(self,p): return initial_state(self.saved.config,p['soc0'])
    def integral_at(self,p,q):
        # Exact integral of the released piecewise-linear current signal.
        t=p['t']; i=p['i']; k=np.clip(np.searchsorted(t,q,side='right')-1,0,len(t)-2); d=np.asarray(q)-t[k]
        slope=(i[k+1]-i[k])/(t[k+1]-t[k]); return p['charge'][k]+i[k]*d+.5*slope*d*d
    def grid(self,p,dt=5.):
        t=np.arange(0,p['t'][-1]+1e-9,dt); q=self.integral_at(p,t); i=np.interp(t,p['t'],p['i']); avg=np.r_[i[0],np.diff(q)/np.diff(t)]
        return {'t':t,'i':i,'avg':avg,'dt':np.r_[0,np.diff(t)],'v':np.interp(t,p['t'],p['v']),'soc0':p['soc0'],'protocol':p}
    def open_loop(self,g,initial):
        def body(x,values):
            avg,current,dt=values; nx=self._step(x,avg,dt); return nx,jnp.r_[nx,self._measure(nx,current)]
        fn=jax.jit(lambda x,av,i,dt:jax.lax.scan(body,x,(av,i,dt))[1])
        tail=np.asarray(fn(jnp.asarray(initial),jnp.asarray(g['avg'][1:]),jnp.asarray(g['i'][1:]),jnp.asarray(g['dt'][1:])))
        first=np.r_[initial,float(self.measure(jnp.asarray(initial),g['i'][0]))]
        return np.vstack((first,tail))
    def report(self,task,metrics,lines,arrays=None):
        out=self.output(task); save_json(out/'metrics.json',metrics); (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        if arrays is not None: np.savez_compressed(out/'trajectory_predictions.npz',**arrays)
        return out

def filter_profile(ctx,g,initial,voltage_sigma=.02,soc_process_per_s=1e-5,polarization_process_per_s=1e-4):
    """Assimilate original observation timestamps; reporting interpolation is never an input."""
    reporting_time=g['t']
    if 'protocol' in g:
        protocol=g['protocol']; source=protocol['t']<=reporting_time[-1]
        observation_time=protocol['t'][source]; observation_voltage=protocol['v'][source]
        merged=np.unique(np.r_[reporting_time,observation_time]); charge=ctx.integral_at(protocol,merged); current=np.interp(merged,protocol['t'],protocol['i'])
        observation_index=np.searchsorted(merged,observation_time); available=np.zeros(len(merged),bool); available[observation_index]=True; values=np.zeros(len(merged)); values[observation_index]=observation_voltage
        work={'t':merged,'i':current,'avg':np.r_[current[0],np.diff(charge)/np.diff(merged)],'dt':np.r_[0,np.diff(merged)],'v':values}
        reporting_index=np.searchsorted(merged,reporting_time)
    else:
        work=g; available=np.ones(len(work['t']),bool); observation_index=np.arange(len(work['t'])); observation_time=work['t']; observation_voltage=work['v']; reporting_index=observation_index
    n=ctx.saved.state_dim
    initial_std=np.array([.02]+[.02]*(n-1)); process_std=np.array([soc_process_per_s]+[polarization_process_per_s]*(n-1))
    if 'hysteresis_state' in ctx.saved.state_names: initial_std[-1]=.2; process_std[-1]=1e-3
    R=voltage_sigma**2; P0=jnp.diag(jnp.asarray(initial_std**2)); eye=jnp.eye(n)
    def body(carry,values):
        x,P=carry; avg,current,dt,observed,has_observation=values
        xp=ctx._step(x,avg,dt); F=jax.jacfwd(ctx._step,argnums=0)(x,avg,dt)
        Q=jnp.diag(jnp.asarray(process_std**2))*dt
        pp=F@P@F.T+Q; predicted=ctx._measure(xp,current); H=jax.jacfwd(ctx._measure,argnums=0)(xp,current)
        S=H@pp@H+R; K=pp@H/S; innovation=observed-predicted; nx=xp+K*innovation
        A=eye-jnp.outer(K,H); np_=A@pp@A.T+jnp.outer(K,K)*R; np_=.5*(np_+np_.T)
        nx=jnp.where(has_observation,nx,xp); np_=jnp.where(has_observation,np_,pp)
        posterior=ctx._measure(nx,current)
        return (nx,np_),(nx,np_,predicted,posterior,innovation,S,H)
    fn=jax.jit(lambda x,av,i,dt,v,flag:jax.lax.scan(body,(x,P0),(av,i,dt,v,flag))[1])
    result=fn(jnp.asarray(initial),jnp.asarray(work['avg'][1:]),jnp.asarray(work['i'][1:]),jnp.asarray(work['dt'][1:]),jnp.asarray(work['v'][1:]),jnp.asarray(available[1:])); result=jax.tree_util.tree_map(np.asarray,result)
    x,P,prior,posterior,innovation,S,H=result; first=float(ctx.measure(jnp.asarray(initial),work['i'][0]))
    all_state=np.vstack((initial,x)); all_covariance=np.concatenate((np.asarray(P0)[None],P)); all_prior=np.r_[first,prior]; all_posterior=np.r_[first,posterior]
    unfiltered=ctx.open_loop(work,initial)
    return {'state':all_state[reporting_index],'covariance':all_covariance[reporting_index],'prior_voltage':all_prior[reporting_index],'posterior_voltage':all_posterior[reporting_index],'innovation':np.r_[np.nan,innovation][observation_index],'innovation_variance':np.r_[np.nan,S][observation_index],'measurement_jacobian':np.vstack((np.asarray(ctx.H(jnp.asarray(initial),work['i'][0])),H))[reporting_index],'observation_time_s':observation_time,'observation_voltage':observation_voltage,'observation_prior_voltage':all_prior[observation_index],'observation_posterior_voltage':all_posterior[observation_index],'observation_openloop_voltage':unfiltered[observation_index,-1]}

def parameter_copy(ctx,name,factor):
    p=dict(ctx.p)
    if name=='capacity':
        target=ctx.capacity*factor
        if not 160<target<220: raise ValueError('Capacity perturbation outside parameterization')
        if 'capacity_fixed' in p: p['capacity_fixed']=jnp.array(target)
        else:
            q=(target-160)/60; p['capacity_logit']=jnp.array(np.log(q/(1-q)))
    elif name=='capacitance': p['capacitance_raw']=jnp.array(np.log(np.expm1((ctx.c1*factor-1)/1e5)))
    elif name=='branch_resistance':
        value=float(.001*jax.nn.softplus(p['r1_raw'])+2e-5); p['r1_raw']=jnp.array(np.log(np.expm1((value*factor-2e-5)/.001)))
    elif name=='slow_resistance':
        value=float(.001*jax.nn.softplus(p['r2_raw'])+1e-7); p['r2_raw']=jnp.array(np.log(np.expm1((value*factor-1e-7)/.001)))
    elif name=='slow_capacitance':
        value=float(1e6*jax.nn.softplus(p['c2_raw'])+1); p['c2_raw']=jnp.array(np.log(np.expm1((value*factor-1)/1e6)))
    elif name=='series_resistance': p['resistance_raw']=jnp.array(np.log(np.expm1((ctx.r0*factor-1e-7)/.001)))
    elif name=='hysteresis':
        target=ctx.h*factor/.05
        if abs(target)>=1: raise ValueError('Hysteresis perturbation outside parameterization')
        p['hysteresis_raw']=jnp.array(np.arctanh(target))
    else: raise ValueError(name)
    return p
def predict_parameters(ctx,p,protocol,initial_soc=None,initial_u=0.):
    fn=jax.jit(ctx.saved.rollout)
    state=initial_state(ctx.saved.config,protocol['soc0'] if initial_soc is None else initial_soc,initial_u); aux=state[1:] if ctx.saved.state_dim==2 else state[None,1:]
    return np.asarray(fn(p,jnp.array([state[0]],dtype=jnp.float64),jnp.asarray(aux,dtype=jnp.float64),jnp.asarray(protocol['i'][None]),jnp.asarray(np.r_[0,np.diff(protocol['t'])][None])))[0]
