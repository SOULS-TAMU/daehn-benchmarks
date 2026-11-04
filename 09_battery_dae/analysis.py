"""Evaluate saved battery models on untouched, full experimental profiles."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import SavedBatteryModel,scalar_parameters,circuit_voltage,continuous_rhs
CASE=_CASE_DIR

def analyze(model_dir,output_dir):
    saved=SavedBatteryModel(model_dir); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    meta=json.loads((CASE/'data'/'metadata.json').read_text()); records=[]; arrays={}; parameters=saved.parameters
    capacity,c1,r0,h=map(float,scalar_parameters(parameters))
    total=[]
    with np.load(CASE/'data'/'data.npz') as a:
        for protocol in meta['protocols']:
            if protocol['split']!='test': continue
            k=protocol['id']; t=a[f't_{k}']; current=a[f'i_{k}']; observed=a[f'v_{k}']; soc0=float(a[f'soc0_{k}'])
            start=time.perf_counter(); pred=saved.predict(current,t,soc0,0.); elapsed=time.perf_counter()-start
            # Independent Kirchhoff check on returned states; no latent-state labels exist.
            alg=pred[:,-1]-np.asarray(circuit_voltage(saved.config,saved.ox,saved.oy,parameters,jnp.asarray(pred[:,:-1]),jnp.asarray(current)))
            resistance=np.asarray(jax.jit(saved.r1)(parameters,jnp.asarray(pred[:,0]),jnp.asarray(current)))
            rhs=np.asarray(continuous_rhs(saved.config,parameters,jnp.asarray(pred[:,:-1]),jnp.asarray(current),jnp.asarray(resistance)))
            derivative=np.gradient(pred[:,:-1],t,axis=0,edge_order=2)
            good=np.ones(len(t),dtype=bool); good[[0,-1]]=False
            # Exclude neighborhoods of discrete current changes from FD residual summary.
            jump=np.flatnonzero(abs(np.diff(current))>.5)
            for idx in jump: good[max(0,idx-1):min(len(t),idx+3)]=False
            residual=derivative-rhs
            charge_balance=pred[:,0]-(soc0-a[f'charge_{k}']/(capacity*3600))
            err=pred[:,-1]-observed; total.extend(err.tolist())
            record={'protocol':protocol['file'],'measurements':len(t),'voltage_rmse_V':float(np.sqrt(np.mean(err**2))),'voltage_mae_V':float(np.mean(abs(err))),'voltage_max_abs_error_V':float(np.max(abs(err))),'voltage_bias_V':float(np.mean(err)),'algebraic_max_abs_residual_V':float(np.max(abs(alg))),'charge_balance_max_abs_residual':float(np.max(abs(charge_balance))),'soc_min':float(pred[:,0].min()),'soc_max':float(pred[:,0].max()),'soc_bound_violation_fraction':float(np.mean((pred[:,0]<0)|(pred[:,0]>1))),'R1_min_ohm':float(resistance.min()),'R1_max_ohm':float(resistance.max()),'continuous_fd_residual_rmse_away_from_current_jumps':np.sqrt(np.mean(residual[good]**2,axis=0)).tolist() if good.any() else None,'continuous_fd_samples':int(good.sum()),'inference_seconds_including_compilation':elapsed,'initial_soc_from_initial_voltage':soc0,'initial_polarization_assumed_V':0.,'evaluation':'open-loop full experiment; only initial voltage used to initialize SOC; future measured voltage never used as input'}
            records.append(record); stem=Path(protocol['file']).stem
            arrays.update({stem+'__time':t,stem+'__current_A':current,stem+'__exact_voltage_V':observed,stem+'__prediction':pred,stem+'__algebraic_residual_V':alg,stem+'__continuous_fd_residual':residual,stem+'__differential_rhs':rhs,stem+'__R1_ohm':resistance})
            fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True); ax[0].plot(t/3600,current); ax[0].set_ylabel('Current (A)')
            ax[1].plot(t/3600,observed,label='measured'); ax[1].plot(t/3600,pred[:,-1],label='DAEHN'); ax[1].set_ylabel('Voltage (V)'); ax[1].legend()
            ax[2].plot(t/3600,1000*err); ax[2].set_ylabel('Voltage error (mV)'); ax[2].set_xlabel('Time (h)'); fig.tight_layout(); fig.savefig(output_dir/(stem+'.png'),dpi=140); plt.close(fig)
    np.savez_compressed(output_dir/'trajectory_predictions.npz',**arrays)
    metrics={'task':'battery_voltage_forecasting','test_voltage_rmse_V':float(np.sqrt(np.mean(np.array(total)**2))),'capacity_estimate_Ah':capacity,'capacitance_estimate_F':c1,'series_resistance_estimate_ohm':r0,'hysteresis_estimate_V':h,'epochs_executed':saved.config['epochs'],'selected_epoch':saved.config['selected_epoch'],'protocols':records,'limitations':['This evaluation does not certify downstream forecast accuracy or convergence.','SOC and polarization are latent: no measured ground-truth SOC or SOH accuracy is claimed.','Data describe one cell at 25 C; no cross-cell, temperature or aging generalization is established.','LFP flat OCV makes initial SOC inference uncertain.','Current is interpolated linearly for charge integration; RC step freezes its averaged current and midpoint resistance. Temporal convergence must be checked before downstream certification.','Capacity/RC parameter identifiability and uncertainty are not established from a single run.','Full-profile test prediction starts with polarization=0; residual initial relaxation/hysteresis may contribute to error.','The algebraic voltage equation is enforced exactly; this does not guarantee voltage accuracy.']}
    metrics.update(variant=saved.config.get('variant','daehn'),activation=saved.config.get('activation','tanh'),state_order=saved.state_names,circuit_equality_enforced=saved.config.get('variant')!='voltage_mlp')
    if saved.config.get('variant')=='voltage_mlp':
        metrics['limitations']=[s for s in metrics['limitations'] if not s.startswith('The algebraic voltage equation')]
        metrics['limitations'].append('Circuit equality residual is a diagnostic discrepancy for the unconstrained readout, not an enforced invariant. Its states/parameters are not independently measured physical states.')
    (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Battery DAE evaluation','',f"Held-out full-profile voltage RMSE: {metrics['test_voltage_rmse_V']:.6g} V.",'','| Protocol | Measurements | Voltage RMSE (V) | Voltage MAE (V) | Algebraic max residual (V) | SOC violations |','|---|---:|---:|---:|---:|---:|']
    for r in records: lines.append(f"| {r['protocol']} | {r['measurements']} | {r['voltage_rmse_V']:.6g} | {r['voltage_mae_V']:.6g} | {r['algebraic_max_abs_residual_V']:.6g} | {r['soc_bound_violation_fraction']:.6g} |")
    lines+=['','Estimated constitutive parameters: '+json.dumps({k:v for k,v in metrics.items() if k.endswith('_estimate_Ah') or k.endswith('_estimate_F') or k.endswith('_estimate_ohm') or k.endswith('_estimate_V')}),'','Limitations:','']+[ '- '+s for s in metrics['limitations']]
    (output_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8'); return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path); args=p.parse_args()
    model_dir=args.model_dir.resolve(); base=CASE.parent/'results'/CASE.name
    if not model_dir.is_relative_to(base/'model'): p.error('model-dir must be inside this case model branch')
    out=args.output_dir or base/'output'/model_dir.relative_to(base/'model')
    if not out.resolve().is_relative_to(base/'output'): p.error('output-dir must be inside this case output branch')
    print(json.dumps(analyze(model_dir,out),indent=2))
