"""Voltage-threshold time and deliverable-energy predictions from saved models."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *

def crossing(t,v,threshold):
    ids=np.flatnonzero((v[:-1]>threshold)&(v[1:]<=threshold))
    if v[0]<=threshold: return float(t[0])
    if not len(ids): return None
    k=ids[0]; return float(t[k]+(threshold-v[k])*(t[k+1]-t[k])/(v[k+1]-v[k]))
def energy_to(t,current,voltage,end):
    if end is None: return None
    before=t<end; tt=np.r_[t[before],end]; ii=np.r_[current[before],np.interp(end,t,current)]; vv=np.r_[voltage[before],np.interp(end,t,voltage)]
    return float(np.trapezoid(np.maximum(ii,0)*vv,tt)/3600)
def domain_exit(t,z):
    bad=np.flatnonzero((z<0)|(z>1)); return float(t[bad[0]]) if len(bad) else None

def run(ctx):
    out=ctx.output('discharge_energy'); records=[]; scenarios=[]; arrays={}; cutoffs=[3.2,3.,2.8]
    for p in ctx.protocols():
        if not p['file'].startswith('CCCV_') or '_dis' not in p['file']: continue
        pred=ctx.saved.predict(p['i'],p['t'],p['soc0']); z=pred[:,0]; static=np.interp(z,ctx.saved.ox,ctx.saved.oy)-ctx.r0*p['i']-ctx.h*np.tanh(p['i']/.25)
        stem=p['file'][:-4]; arrays.update({stem+'__time_s':p['t'],stem+'__current_A':p['i'],stem+'__exact_voltage_V':p['v'],stem+'__prediction':pred,stem+'__static_no_RC_voltage_V':static}); exit_time=domain_exit(p['t'],z)
        for threshold in cutoffs:
            measured=crossing(p['t'],p['v'],threshold)
            for name,v in [('daehn',pred[:,-1]),('static_no_RC',static)]:
                predicted=crossing(p['t'],v,threshold); valid=predicted is not None and (exit_time is None or predicted<=exit_time)
                e_true=energy_to(p['t'],p['i'],p['v'],measured); e_model=energy_to(p['t'],p['i'],v,predicted)
                records.append({'protocol':p['file'],'original_split':p['split'],'method':name,'voltage_threshold_V':threshold,'measured_crossing_s':measured,'predicted_crossing_s':predicted,'crossing_error_s':None if measured is None or predicted is None else predicted-measured,'crossing_relative_error':None if measured is None or predicted is None else (predicted-measured)/max(measured,1),'observed_deliverable_energy_Wh':e_true,'predicted_deliverable_energy_Wh':e_model,'energy_relative_error':None if e_true is None or e_model is None else (e_model-e_true)/e_true,'energy_error_Wh_at_same_measured_crossing':None if measured is None else energy_to(p['t'],p['i'],v,measured)-e_true,'soc_domain_exit_s':exit_time,'prediction_within_soc_domain':valid,'status':'domain_exit_before_event' if predicted is not None and not valid else 'crossing_observed' if measured is not None and predicted is not None else 'censored_no_crossing_in_record','claim':'Voltage-threshold reserve/EOD planning demonstration; thresholds are analysis choices, not manufacturer limits.'})
        for parameter in ['initial_soc','capacity','series_resistance','capacitance']:
            for amount in [-.02,.02] if parameter=='initial_soc' else [-.05,.05]:
                value=predict_parameters(ctx,ctx.p,p,initial_soc=p['soc0']+amount) if parameter=='initial_soc' else predict_parameters(ctx,parameter_copy(ctx,parameter,1+amount),p)
                event=crossing(p['t'],value[:,-1],2.8); exit_=domain_exit(p['t'],value[:,0]); scenarios.append({'protocol':p['file'],'parameter':parameter,'perturbation':amount,'2.8V_crossing_s':event,'deliverable_energy_Wh':energy_to(p['t'],p['i'],value[:,-1],event),'soc_domain_exit_s':exit_,'within_domain':event is not None and (exit_ is None or event<=exit_),'interpretation':'Deterministic sensitivity scenario, not a statistical confidence interval.'})
        fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].plot(p['t']/3600,p['v'],label='measured'); ax[0].plot(p['t']/3600,pred[:,-1],label='DAEHN'); ax[0].plot(p['t']/3600,static,label='no RC',alpha=.7)
        for limit in cutoffs: ax[0].axhline(limit,ls='--',color='grey',lw=.7)
        ax[0].set_xlabel('Time (h)'); ax[0].set_ylabel('Voltage (V)'); ax[0].legend(); rows=[r for r in records if r['protocol']==p['file'] and r['method']=='daehn']; ax[1].bar([str(r['voltage_threshold_V']) for r in rows],[r['crossing_error_s']/60 if r['crossing_error_s'] is not None else 0 for r in rows]); ax[1].set_xlabel('Threshold (V)'); ax[1].set_ylabel('Crossing-time error (min)'); fig.tight_layout(); fig.savefig(out/(stem+'_cutoff.png'),dpi=140); plt.close(fig)
    # A held-out mixed-current energy-accounting task, independent of cutoff choice.
    mixed=[]
    for p in ctx.protocols('test'):
        pred=ctx.saved.predict(p['i'],p['t'],p['soc0']); discharge=np.maximum(p['i'],0); charge=np.maximum(-p['i'],0)
        obs_dis=float(np.trapezoid(discharge*p['v'],p['t'])/3600); pred_dis=float(np.trapezoid(discharge*pred[:,-1],p['t'])/3600); obs_chg=float(np.trapezoid(charge*p['v'],p['t'])/3600); pred_chg=float(np.trapezoid(charge*pred[:,-1],p['t'])/3600)
        mixed.append({'protocol':p['file'],'observed_discharge_energy_Wh':obs_dis,'predicted_discharge_energy_Wh':pred_dis,'discharge_energy_relative_error':(pred_dis-obs_dis)/obs_dis,'observed_charge_energy_Wh':obs_chg,'predicted_charge_energy_Wh':pred_chg,'charge_energy_relative_error':(pred_chg-obs_chg)/obs_chg,'interpretation':'Full-profile charge/discharge energy accounting, not an EOD trajectory. Domain violations are reported by reliability analysis.'})
    metrics={'threshold_predictions':records,'sensitivity_scenarios':scenarios,'heldout_energy_accounting':mixed,'future_current':'Measured current supplied as the known discharge scenario; all future voltage labels withheld from initialization/prediction.','split_limitation':'18 A and 180 A full-discharge protocols were training data; 50 A was checkpoint-selection validation. Their cutoff results demonstrate application mechanics, not independent held-out EOD validation.','cutoff_interpretation':'3.2, 3.0 and 2.8 V are reserve/voltage-threshold planning choices. A cutoff is an event; no voltage or SOC inequality projection has been added.','event_interpolation':'First downward crossing, linear interpolation between neighboring released samples. Censored cases are retained.'}
    lines=['# Remaining-discharge and energy application','',metrics['split_limitation'],metrics['future_current'],'','| Protocol | Split | Threshold (V) | Measured time (min) | Predicted time (min) | Error (min) | Energy error (%) | Status |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        if r['method']!='daehn': continue
        fmt=lambda v:'censored' if v is None else f'{v:.3f}'
        lines.append(f"| {r['protocol']} | {r['original_split']} | {r['voltage_threshold_V']} | {fmt(None if r['measured_crossing_s'] is None else r['measured_crossing_s']/60)} | {fmt(None if r['predicted_crossing_s'] is None else r['predicted_crossing_s']/60)} | {fmt(None if r['crossing_error_s'] is None else r['crossing_error_s']/60)} | {fmt(None if r['energy_relative_error'] is None else 100*r['energy_relative_error'])} | {r['status']} |")
    lines+=['',metrics['cutoff_interpretation'],'','Independent mixed-profile energy accounting:']
    for r in mixed: lines.append(f"- {r['protocol']}: discharge-energy error {100*r['discharge_energy_relative_error']:.3f}%, charge-energy error {100*r['charge_energy_relative_error']:.3f}%.")
    lines+=['','Capacity, initial SOC, R0 and C1 scenarios expose fragility of event predictions. Their ranges are not uncertainty bounds. Any event occurring after an SOC-domain exit must not be treated as a physically validated forecast.']
    return ctx.report('discharge_energy',metrics,lines,arrays)
if __name__=='__main__':
    args=parser(__doc__).parse_args(); print(run(Context(args.model_dir,args.study)))
