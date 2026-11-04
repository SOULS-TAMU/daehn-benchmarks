"""Run the complete reusable battery case study against one saved checkpoint."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import json,hashlib,time
from study_common import *
import reliability,state_estimation,forecasting,discharge_application,identifiability

def main():
    p=parser(__doc__); p.add_argument('--tasks',nargs='+',choices=['reliability','state_estimation','forecasting','discharge_energy','identifiability'],default=['reliability','state_estimation','forecasting','discharge_energy','identifiability']); args=p.parse_args(); ctx=Context(args.model_dir,args.study)
    checksum=hashlib.sha256((ctx.model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(); jobs={'reliability':reliability.run,'state_estimation':state_estimation.run,'forecasting':forecasting.run,'discharge_energy':discharge_application.run,'identifiability':identifiability.run}; timings={}
    for name in args.tasks:
        print('START',name,flush=True); start=time.perf_counter(); result=jobs[name](ctx); timings[name]=time.perf_counter()-start; print('DONE',name,round(timings[name],2),'s',result,flush=True)
    if checksum!=hashlib.sha256((ctx.model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(): raise RuntimeError('Saved model changed')
    documents={}
    for name in jobs:
        path=ctx.root/name/'metrics.json'
        if path.exists(): documents[name]=json.loads(path.read_text())
    summary={'model_dir':str(ctx.model_dir),'checkpoint_sha256':checksum,'epochs_executed':ctx.saved.config['epochs'],'selected_epoch':ctx.saved.config['selected_epoch'],'tasks_completed':list(documents),'task_wall_seconds':timings,'inequality_projection_added':False,'retraining_performed':False,'results':documents}
    summary.update(variant=ctx.saved.config.get('variant','daehn'),activation=ctx.saved.config.get('activation','tanh'),state_order=ctx.saved.state_names,training_configuration=ctx.saved.config)
    save_json(ctx.root/'metrics.json',summary)
    lines=['# Battery DAE case-study campaign','',f"Checkpoint: {ctx.model_dir}",f"Training epochs: {ctx.saved.config['epochs']}; selected epoch: {ctx.saved.config['selected_epoch']}.",'','The checkpoint is read-only. All tasks are repeatable with a longer-trained compatible checkpoint. No inequality projection has been added.','']
    lines+=['Variant: '+ctx.saved.config.get('variant','daehn')+'; activation: '+ctx.saved.config.get('activation','tanh')+'; states: '+', '.join(ctx.saved.state_names)+'.','', 'Forecast method keys retain the legacy daehn_ prefix for compatibility; they refer to the selected checkpoint variant. Physical state accuracy remains unverified.','']
    if 'reliability' in documents:
        lines+=['## Reliability','']
        for r in documents['reliability']['protocols']: lines.append(f"- {r['protocol']}: voltage RMSE {r['rmse_V']*1000:.3f} mV, 95th absolute error {r['p95_abs_error_V']*1000:.3f} mV; SOC violations {100*r['soc_violation_fraction']:.3f}%.")
    if 'state_estimation' in documents:
        lines+=['','## Causal state-estimation application','']
        for r in documents['state_estimation']['real_experiments']: lines.append(f"- {r['protocol']}: open-loop voltage RMSE {1000*r['openloop_voltage_error']['rmse_V']:.3f} mV; EKF one-step prior RMSE {1000*r['causal_one_step_prior_voltage_error']['rmse_V']:.3f} mV. Real SOC accuracy is unverified.")
    if 'forecasting' in documents:
        lines+=['','## Rolling forecast application','','| Protocol | Horizon | EKF + known current RMSE (mV) | Persistence RMSE (mV) | Load-adjusted persistence RMSE (mV) |','|---|---:|---:|---:|---:|']
        rows=documents['forecasting']['records']
        for protocol in sorted(set(r['protocol'] for r in rows)):
            for horizon in [60,300,900,1800]:
                value=lambda name:next(r['rmse_V']*1000 for r in rows if r['protocol']==protocol and r['horizon_seconds']==horizon and r['method']==name)
                lines.append(f"| {protocol} | {horizon} s | {value('daehn_ekf_known_current'):.3f} | {value('persistence'):.3f} | {value('load_adjusted_persistence'):.3f} |")
        lines+=['','Known-current forecasts use the released future current as a scenario input. The persistent-current variant quantifies demand-assumption error. Empirical residual bands have no formal coverage guarantee.']
        lines+=['','Short-horizon superiority is not established: persistence or load-adjusted persistence can outperform the DAEHN forecast. The model advantage must be reported by horizon and protocol, not as a blanket improvement.']
    if 'discharge_energy' in documents:
        lines+=['','## Discharge reserve and energy application','',documents['discharge_energy']['split_limitation']]
        for r in documents['discharge_energy']['threshold_predictions']:
            if r['method']=='daehn' and r['original_split']=='validation': lines.append(f"- {r['protocol']}, threshold {r['voltage_threshold_V']} V: crossing error {None if r['crossing_error_s'] is None else round(r['crossing_error_s']/60,3)} min; status {r['status']}.")
        for r in documents['discharge_energy']['heldout_energy_accounting']: lines.append(f"- {r['protocol']}: observed discharge energy {r['observed_discharge_energy_Wh']:.3f} Wh; modeled discharge-energy error {100*r['discharge_energy_relative_error']:.3f}%.")
        candidates=[r for r in documents['discharge_energy']['threshold_predictions'] if r['method']=='daehn' and r['crossing_error_s'] is not None]
        if candidates:
            worst=max(candidates,key=lambda r:abs(r['crossing_error_s'])); lines.append(f"- Largest tested crossing failure: {worst['protocol']}, {worst['voltage_threshold_V']} V, error {worst['crossing_error_s']/60:.3f} min. Accurate integrated energy does not establish reliable first-crossing decisions.")
    lines+=['','## Evidence needed for a stronger case study','','1. Retrain for longer epochs and multiple seeds, keeping protocol splits fixed.','2. Compare learned R1 with a separately calibrated conventional ECM and an unconstrained learned voltage model. Current cheap comparators do not replace those trained baselines.','3. Resolve capacity/initialization/charge-accounting mismatch before interpreting SOC or EOD as physically reliable.','4. Obtain independent SOC/capacity labels and a new full-discharge test protocol for measured state/EOD validation.','5. Check calibration across load regimes, numerical refinement, and added diffusion/hysteresis model adequacy.','6. Report forecast horizon, known/unknown current assumptions, domain exits, censored events, and sensitivity with every claimed benefit.','','Detailed outputs:']
    for name in documents: lines.append(f"- [{name}]({name}/report.md): metrics.json, trajectory_predictions.npz and plots.")
    (ctx.root/'report.md').write_text('\n'.join(lines),encoding='utf-8'); save_json(ctx.root/'campaign_manifest.json',{'schema_version':1,'checkpoint_sha256':checksum,'dataset_sha256':ctx.meta['data_sha256'],'model_dir':str(ctx.model_dir),'tasks':args.tasks,'future_voltage_input':False,'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CASE.glob('*.py')}})
    print('CAMPAIGN REPORT',ctx.root/'report.md',flush=True)
if __name__=='__main__': main()
