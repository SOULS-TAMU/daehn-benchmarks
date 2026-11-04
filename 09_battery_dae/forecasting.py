"""Rolling scenario forecasts with causal initialization and cheap comparators."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *
HORIZONS=np.array([30,60,300,900,1800])
METHODS=['persistence','load_adjusted_persistence','daehn_openloop_known_current','daehn_ekf_known_current','daehn_ekf_persistent_current']

def forecast_function(ctx,batch_size=32):
    _,_,voltage,transition,_=make_model(ctx.saved.config,ctx.saved.ox,ctx.saved.oy,batch_size)
    @jax.jit
    def forecast(p,state,avg,current):
        def body(x,values):
            average,endpoint=values
            aux=x[:,1] if ctx.saved.state_dim==2 else x[:,1:]
            z,u=transition(p,x[:,0],aux,jnp.full((batch_size,),5.),average,average)
            nx=pack_state(z,u); v=voltage(p,z,u,endpoint)
            return nx,jnp.concatenate((nx,v[:,None]),axis=-1)
        return jax.lax.scan(body,state,(avg.T,current.T))[1].transpose((1,0,2))
    return forecast

def predictions_for_protocol(ctx,p,fn):
    g=ctx.grid(p,5.); native=ctx.open_loop(g,ctx.initial(p)); estimate=filter_profile(ctx,g,ctx.initial(p)); L=int(HORIZONS[-1]/5)
    origins=np.arange(60,len(g['t'])-L,60) # 300 s burn-in and spacing.
    selectors=(HORIZONS/5).astype(int)-1; rows={m:[] for m in METHODS}; state_bounds={m:[] for m in METHODS if m.startswith('daehn')}; truth=[]
    for start in range(0,len(origins),32):
        ids=origins[start:start+32]; n=len(ids); padded=np.pad(ids,(0,32-n),mode='edge'); future=padded[:,None]+np.arange(1,L+1)[None,:]
        avg=g['avg'][future]; current=g['i'][future]; i0=g['i'][padded]
        # Cheap comparators receive only the last actual voltage observation.
        available=np.searchsorted(p['t'],g['t'][padded],side='right')-1; v0=p['v'][available]; observed_current=p['i'][available]
        openloop=np.asarray(fn(ctx.p,jnp.asarray(native[padded,:-1]),jnp.asarray(avg),jnp.asarray(current)))[:n]
        ekf=np.asarray(fn(ctx.p,jnp.asarray(estimate['state'][padded]),jnp.asarray(avg),jnp.asarray(current)))[:n]
        persistent=np.repeat(i0[:,None],L,axis=1)
        unknown=np.asarray(fn(ctx.p,jnp.asarray(estimate['state'][padded]),jnp.asarray(persistent),jnp.asarray(persistent)))[:n]
        truth.extend(g['v'][future[:n,selectors]].tolist())
        rows['persistence'].extend(np.repeat(v0[:n,None],len(HORIZONS),axis=1).tolist())
        adjusted=v0[:,None]-ctx.r0*(current[:,selectors]-observed_current[:,None])-ctx.h*(np.tanh(current[:,selectors]/.25)-np.tanh(observed_current[:,None]/.25))
        rows['load_adjusted_persistence'].extend(adjusted[:n].tolist())
        for name,value in [('daehn_openloop_known_current',openloop),('daehn_ekf_known_current',ekf),('daehn_ekf_persistent_current',unknown)]:
            rows[name].extend(value[:,selectors,-1].tolist()); bad=(value[:,:,0]<0)|(value[:,:,0]>1); cumulative=np.maximum.accumulate(bad,axis=1); state_bounds[name].extend(cumulative[:,selectors].tolist())
    return g['t'][origins],np.asarray(truth),{k:np.asarray(v) for k,v in rows.items()},{k:np.asarray(v) for k,v in state_bounds.items()}

def run(ctx):
    out=ctx.output('forecasting'); fn=forecast_function(ctx); calibration={m:[[] for _ in HORIZONS] for m in METHODS}; arrays={}
    for p in ctx.protocols('validation'):
        _,truth,values,_=predictions_for_protocol(ctx,p,fn)
        for name in METHODS:
            for k in range(len(HORIZONS)): calibration[name][k].extend(abs(values[name][:,k]-truth[:,k]).tolist())
    bands={name:[float(np.quantile(v,.95)) for v in columns] for name,columns in calibration.items()}; records=[]
    for p in ctx.protocols('test'):
        origins,truth,values,bounds=predictions_for_protocol(ctx,p,fn); stem=p['file'][:-4]; arrays.update({stem+'__origin_time_s':origins,stem+'__horizon_s':HORIZONS,stem+'__exact_voltage_V':truth})
        for name,pred in values.items():
            arrays[stem+'__'+name+'__prediction_V']=pred
            if name in bounds: arrays[stem+'__'+name+'__domain_exit']=bounds[name]
            for k,horizon in enumerate(HORIZONS):
                err=pred[:,k]-truth[:,k]; row={'protocol':p['file'],'method':name,'horizon_seconds':int(horizon),**error_metrics(err),'validation_calibrated_error_halfwidth_V':bands[name][k],'empirical_test_band_coverage':float(np.mean(abs(err)<=bands[name][k]))}
                if name in bounds: row['forecast_domain_exit_fraction']=float(bounds[name][:,k].mean())
                records.append(row)
        fig,ax=plt.subplots(1,3,figsize=(15,4))
        for name in METHODS[:-1]:
            vals=[r for r in records if r['protocol']==p['file'] and r['method']==name]; ax[0].plot(HORIZONS,[r['rmse_V']*1000 for r in vals],marker='o',label=name.replace('daehn_',''))
        ax[0].set_xscale('log'); ax[0].set_xlabel('Forecast horizon (s)'); ax[0].set_ylabel('Voltage RMSE (mV)'); ax[0].legend(fontsize=7)
        for name in ['daehn_ekf_known_current','daehn_ekf_persistent_current']:
            vals=[r for r in records if r['protocol']==p['file'] and r['method']==name]; ax[1].plot(HORIZONS,[r['rmse_V']*1000 for r in vals],marker='o',label=name.replace('daehn_ekf_',''))
        ax[1].set_xscale('log'); ax[1].set_xlabel('Forecast horizon (s)'); ax[1].set_ylabel('Load-assumption effect: RMSE (mV)'); ax[1].legend(fontsize=7)
        k=3; ax[2].scatter(truth[:,k],values['daehn_ekf_known_current'][:,k],s=5,label='15 min forecast'); low=min(truth[:,k].min(),values['daehn_ekf_known_current'][:,k].min()); high=max(truth[:,k].max(),values['daehn_ekf_known_current'][:,k].max()); ax[2].plot([low,high],[low,high],'k--'); ax[2].set_xlabel('Measured future voltage (V)'); ax[2].set_ylabel('Predicted future voltage (V)'); ax[2].legend(); fig.tight_layout(); fig.savefig(out/(stem+'_horizon_accuracy.png'),dpi=140); plt.close(fig)
    metrics={'horizons_seconds':HORIZONS,'records':records,'origin_spacing_seconds':300,'burn_in_seconds':300,'known_current_interpretation':'Released future current is supplied as a known scenario; no future voltage is used. Not a forecast of unknown future demand.','persistent_current_interpretation':'Future load assumed equal to current at origin; reports combined demand-assumption and voltage-model error.','band_interpretation':'Horizon/method-specific 95th absolute residual quantile from validation protocols. Empirical error band, not a formal 95% prediction guarantee.','domain_interpretation':'Raw forecasts retained; crossing SOC outside [0,1] flags unreliable-domain forecasts without clipping or inequality projection.'}
    lines=['# Rolling voltage-forecast application','','Every method uses identical origins and targets. EKF initialization uses observations only through the forecast origin. The posterior at an origin is not scored as a future prediction. Current integrals are preserved on a 5 s grid.','',metrics['known_current_interpretation'],metrics['persistent_current_interpretation'],'','| Protocol | Method | Horizon (s) | RMSE (mV) | >100 mV fraction | Empirical band coverage | Domain exit fraction |','|---|---|---:|---:|---:|---:|---:|']
    for r in records: lines.append(f"| {r['protocol']} | {r['method']} | {r['horizon_seconds']} | {r['rmse_V']*1000:.3f} | {r['fraction_above_100mV']:.4f} | {r['empirical_test_band_coverage']:.4f} | {r.get('forecast_domain_exit_fraction','NA')} |")
    lines+=['',metrics['band_interpretation'],metrics['domain_interpretation'],'','Comparators: persistence holds the last actual voltage observation available at the origin; load-adjusted persistence updates its instantaneous series-resistance/offset drop from the load at that observation, holding OCV and polarization fixed. No interpolated reporting-grid voltage is supplied to a predictor. Future target voltages are interpolated only for evaluation. Neither comparator is a separately trained model.']
    return ctx.report('forecasting',metrics,lines,arrays)
if __name__=='__main__':
    args=parser(__doc__).parse_args(); print(run(Context(args.model_dir,args.study)))
