"""Measured-profile reliability, residual tails, and physical-domain checks."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import numpy as np
import jax,jax.numpy as jnp
from study_common import *

def block_interval(t,error,seed=42,seconds=300,repeats=500):
    blocks=np.floor(t/seconds).astype(int); ids=np.unique(blocks); sums=np.array([np.sum(error[blocks==k]**2) for k in ids]); counts=np.array([np.sum(blocks==k) for k in ids]); rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(ids),(repeats,len(ids))); values=np.sqrt(sums[samples].sum(1)/counts[samples].sum(1)); return np.quantile(values,[.025,.975]).tolist()

def run(ctx):
    out=ctx.output('reliability'); arrays={}; records=[]; calibration=[]
    for p in ctx.protocols('validation'):
        pred=ctx.saved.predict(p['i'],p['t'],p['soc0']); calibration.extend(abs(pred[:,-1]-p['v']).tolist())
    band=float(np.quantile(calibration,.95))
    for p in ctx.protocols('test'):
        pred=ctx.saved.predict(p['i'],p['t'],p['soc0']); err=pred[:,-1]-p['v']; stem=p['file'][:-4]
        ocv=np.interp(pred[:,0],ctx.saved.ox,ctx.saved.oy); algebraic=pred[:,-1]-np.asarray(circuit_voltage(ctx.saved.config,ctx.saved.ox,ctx.saved.oy,ctx.p,jnp.asarray(pred[:,:-1]),jnp.asarray(p['i'])))
        # Static comparator shares capacity/R0/H but omits RC polarization.
        static=ocv-ctx.r0*p['i']-ctx.h*np.tanh(p['i']/.25)
        by_regime={label:error_metrics(err[mask]) for label,mask in {'charge':p['i']<-.25,'rest':abs(p['i'])<=.25,'discharge':p['i']>.25}.items()}
        by_voltage={label:error_metrics(err[mask]) for label,mask in {'below_3V':p['v']<3.,'3_to_3.2V':(p['v']>=3)&(p['v']<3.2),'3.2_to_3.4V':(p['v']>=3.2)&(p['v']<3.4),'above_3.4V':p['v']>=3.4}.items()}
        charge=pred[:,0]-(p['soc0']-p['charge']/(ctx.capacity*3600)); g=ctx.grid(p,5.); refined=ctx.open_loop(g,ctx.initial(p)); native_at_grid=np.column_stack([np.interp(g['t'],p['t'],pred[:,k]) for k in range(pred.shape[1])])
        variants=[]
        for dz,du in [(0,0),(-.02,0),(.02,0),(0,-.02),(0,.02)]:
            value=ctx.saved.predict(p['i'],p['t'],p['soc0']+dz,du); variants.append({'initial_soc_offset':dz,'initial_polarization_offset_V':du,**error_metrics(value[:,-1]-p['v']),'soc_violation_fraction':float(np.mean((value[:,0]<0)|(value[:,0]>1)))})
        record={'protocol':p['file'],**error_metrics(err),'time_weighted_rmse_V':float(np.sqrt(np.trapezoid(err**2,p['t'])/(p['t'][-1]-p['t'][0]))),'300s_block_bootstrap_rmse_interval_V':block_interval(p['t'],err),'by_current_regime':by_regime,'by_measured_voltage':by_voltage,'static_no_RC_comparator':error_metrics(static-p['v']),'soc_min':float(pred[:,0].min()),'soc_max':float(pred[:,0].max()),'soc_violation_fraction':float(np.mean((pred[:,0]<0)|(pred[:,0]>1))),'algebraic_max_abs_residual_V':float(np.max(abs(algebraic))),'charge_balance_max_abs_residual':float(np.max(abs(charge))),'validation_calibrated_95pct_absolute_error_halfwidth_V':band,'test_empirical_band_coverage':float(np.mean(abs(err)<=band)),'5s_refined_vs_native_voltage_rmse_V':rmse(refined[:,-1]-native_at_grid[:,-1]),'initialization_sensitivity':variants}
        records.append(record); arrays.update({stem+'__time_s':p['t'],stem+'__current_A':p['i'],stem+'__exact_voltage_V':p['v'],stem+'__prediction':pred,stem+'__static_no_RC_voltage_V':static,stem+'__voltage_error_V':err,stem+'__algebraic_residual_V':algebraic,stem+'__refined_time_s':g['t'],stem+'__refined_prediction':refined})
        fig,ax=plt.subplots(2,2,figsize=(11,7)); ax[0,0].hist(err*1000,bins=100); ax[0,0].set_xlabel('Voltage error (mV)'); ax[0,0].set_ylabel('Measurements'); ax[0,1].scatter(p['i'][::20],err[::20]*1000,s=2); ax[0,1].set_xlabel('Current (A)'); ax[0,1].set_ylabel('Error (mV)'); ax[1,0].scatter(p['v'][::20],err[::20]*1000,s=2); ax[1,0].set_xlabel('Measured voltage (V)'); ax[1,0].set_ylabel('Error (mV)'); ax[1,1].plot(p['t']/3600,pred[:,0]); ax[1,1].axhline(1,color='r',ls='--'); ax[1,1].set_xlabel('Time (h)'); ax[1,1].set_ylabel('Inferred SOC'); fig.tight_layout(); fig.savefig(out/(stem+'_diagnostics.png'),dpi=140); plt.close(fig)
    metrics={'protocols':records,'error_band_calibration':'Absolute full-profile residual quantile from validation experiments; empirical coverage only, no formal coverage guarantee.','bootstrap_interpretation':'Temporal block bootstrap of a fixed model/profile, not uncertainty across cells or trained seeds.','soc_truth_available':False,'inequality_projection_added':False}
    lines=['# Reliability of measured-profile predictions','','| Protocol | RMSE (mV) | Time-weighted RMSE (mV) | 95% absolute error (mV) | >100 mV fraction | SOC violations |','|---|---:|---:|---:|---:|---:|']
    for r in records: lines.append(f"| {r['protocol']} | {r['rmse_V']*1000:.3f} | {r['time_weighted_rmse_V']*1000:.3f} | {r['p95_abs_error_V']*1000:.3f} | {r['fraction_above_100mV']:.5f} | {r['soc_violation_fraction']:.5f} |")
    lines+=['','Constraint consistency and finite numerical predictions do not certify voltage or SOC accuracy. SOC/polarization have no measured reference. Initialization offsets are sensitivity scenarios, not posterior confidence intervals. Voltage-region and charge/discharge stratification identify where application errors concentrate. Numerical refinement preserves the exact integral of the released piecewise-linear current; any change from the native irregular-step forecast measures discretization sensitivity.','',metrics['error_band_calibration'],metrics['bootstrap_interpretation']]
    metrics.update(variant=ctx.saved.config.get('variant','daehn'),circuit_equality_enforced=ctx.saved.config.get('variant')!='voltage_mlp',state_order=ctx.saved.state_names)
    lines+=['','For voltage_mlp, the circuit residual is an unenforced reference discrepancy. Physical parameter interpretations are conditional; its voltage readout does not use R0/H directly.']
    return ctx.report('reliability',metrics,lines,arrays)
if __name__=='__main__':
    args=parser(__doc__).parse_args(); print(run(Context(args.model_dir,args.study)))
