"""Audit and summarize a completed battery training/application suite without retraining."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,hashlib,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CASE=_CASE_DIR
BASE=CASE.parent/'results'/CASE.name

def read(path): return json.loads(Path(path).read_text())
def stats(values):
    a=np.asarray(values,float)
    return {'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else None,'min':float(a.min()),'max':float(a.max()),'n':len(a)}
def fmt(s,scale=1): return f"{s['mean']*scale:.2f} ± {s['std']*scale:.2f}" if s['std'] is not None else f"{s['mean']*scale:.2f}"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite-manifest',type=Path,required=True); p.add_argument('--study',default='case_study_v2')
    p.add_argument('--name',default='completed_suite_review'); args=p.parse_args()
    if Path(args.name).name!=args.name or Path(args.study).name!=args.study: p.error('Names must be one folder component')
    suite=read(args.suite_manifest); expected=len(set(suite['settings']['models']))*len(set(suite['settings']['seeds']))
    if len(suite['runs'])!=expected or any(r['status'] not in ('trained','analyzed') for r in suite['runs']): raise ValueError('Training suite is incomplete')
    if {(r['model'],r['seed']) for r in suite['runs']}!={(v,s) for v in suite['settings']['models'] for s in suite['settings']['seeds']}: raise ValueError('Missing or repeated model/seed pair')
    results=[]; reference=None; dataset=suite['dataset_sha256']
    if hashlib.sha256((CASE/'data'/'data.npz').read_bytes()).hexdigest()!=dataset: raise ValueError('Dataset checksum mismatch')
    for run in suite['runs']:
        model=Path(run['model_dir']); relative=model.relative_to(BASE/'model'); campaign=BASE/'output'/args.study/relative
        config=read(model/'run_config.json'); manifest=read(model/'checkpoint_manifest.json'); verification=read(campaign/'verification.json')
        for key in ['epochs','learning_rate','batch_size','window_length','activation','validate_every']:
            if config[key]!=suite['settings'][key]: raise ValueError('Training setting differs from requested suite: '+key)
        if config['hidden_sizes']!=[suite['settings']['hidden_size']]*suite['settings']['hidden_layers']: raise ValueError('Network configuration differs from suite')
        checksum=hashlib.sha256((model/'parameter_tree.pkl').read_bytes()).hexdigest()
        if checksum!=manifest['checkpoint_sha256'] or config['data_sha256']!=dataset: raise ValueError('Model/data hash mismatch')
        if verification['status']!='passed' or verification['issues']: raise ValueError('Campaign verification failed')
        with np.load(campaign/'forecasting'/'trajectory_predictions.npz') as a:
            target={k:a[k] for k in a.files if k.endswith(('__exact_voltage_V','__origin_time_s','__horizon_s'))}
        if reference is None: reference=target
        elif target.keys()!=reference.keys() or any(not np.array_equal(target[k],reference[k]) for k in target): raise ValueError('Forecast targets do not match')
        tasks={name:read(campaign/name/'metrics.json') for name in ['reliability','forecasting','state_estimation','discharge_energy','identifiability']}
        tails=[]
        with np.load(campaign/'reliability'/'trajectory_predictions.npz') as a:
            for protocol in ['halfcycles.csv','loadprofile.csv']:
                stem=protocol[:-4]; prediction=a[stem+'__prediction']; observed=a[stem+'__exact_voltage_V']; error=prediction[:,-1]-observed; square=error**2
                masks={'above_100mV':abs(error)>.1,'measured_above_3.4V':observed>=3.4,'soc_outside_domain':(prediction[:,0]<0)|(prediction[:,0]>1)}
                tails.append({'protocol':protocol,**{name:{'measurement_fraction':float(mask.mean()),'fraction_of_total_squared_error':float(square[mask].sum()/square.sum())} for name,mask in masks.items()}})
        original=BASE/'output'/relative; physical=read(original/'metrics.json'); history=read(original/'training_history.json')
        last=history[-1]['validation_scaled_voltage_mse']; early=min(history,key=lambda x:abs(x['epoch']-2500))['validation_scaled_voltage_mse']
        with (model/'parameter_tree.pkl').open('rb') as f: parameters=pickle.load(f)
        extras={}
        if 'r2_raw' in parameters:
            r2=float(.001*np.logaddexp(0,parameters['r2_raw'])+1e-7); c2=float(1e6*np.logaddexp(0,parameters['c2_raw'])+1)
            extras={'slow_R2_ohm':r2,'slow_C2_F':c2,'slow_time_constant_s':r2*c2}
        results.append({'variant':run['model'],'seed':run['seed'],'model_dir':str(model),'campaign':str(campaign),'checkpoint_sha256':checksum,'selected_epoch':config['selected_epoch'],'training_seconds':run['wall_seconds'],'late_validation_improvement_fraction':(early-last)/early,'first_validation_mse':history[0]['validation_scaled_voltage_mse'],'last_validation_mse':last,'best_validation_mse':min(x['validation_scaled_voltage_mse'] for x in history),'config':config,'physical_parameters':{k:v for k,v in physical.items() if '_estimate_' in k},'extra_parameters':extras,'tail_diagnostics':tails,'tasks':tasks})
    variants=list(suite['settings']['models']); protocols=['halfcycles.csv','loadprofile.csv']; rows=[]; horizons=[]; events=[]; training=[]
    for variant in variants:
        runs=[r for r in results if r['variant']==variant]
        training.append({'variant':variant,'selected_epochs':[r['selected_epoch'] for r in runs],'seconds':stats([r['training_seconds'] for r in runs]),'late_validation_improvement':stats([r['late_validation_improvement_fraction'] for r in runs]),'capacity_Ah':stats([r['physical_parameters']['capacity_estimate_Ah'] for r in runs]),'conditional_sensitivity_condition':stats([r['tasks']['identifiability']['sensitivity_condition_number'] for r in runs])})
        for protocol in protocols:
            rr=[next(x for x in r['tasks']['reliability']['protocols'] if x['protocol']==protocol) for r in runs]
            se=[next(x for x in r['tasks']['state_estimation']['real_experiments'] if x['protocol']==protocol) for r in runs]
            ee=[next(x for x in r['tasks']['discharge_energy']['heldout_energy_accounting'] if x['protocol']==protocol) for r in runs]
            row={'variant':variant,'protocol':protocol}
            for key in ['rmse_V','time_weighted_rmse_V','p95_abs_error_V','max_abs_error_V','fraction_above_100mV','soc_violation_fraction','soc_max','algebraic_max_abs_residual_V','charge_balance_max_abs_residual','5s_refined_vs_native_voltage_rmse_V','test_empirical_band_coverage']:
                row[key]=stats([x[key] for x in rr])
            row.update(ekf_prior_rmse_V=stats([x['causal_one_step_prior_voltage_error']['rmse_V'] for x in se]),filter_openloop_rmse_V=stats([x['openloop_voltage_error']['rmse_V'] for x in se]),posterior_soc_violation_fraction=stats([x['posterior_soc_violation_fraction'] for x in se]),discharge_energy_relative_error=stats([x['discharge_energy_relative_error'] for x in ee]),charge_energy_relative_error=stats([x['charge_energy_relative_error'] for x in ee]))
            rows.append(row)
            tail_rows=[next(x for x in r['tail_diagnostics'] if x['protocol']==protocol) for r in runs]
            row['error_tail_squared_error_fraction']=stats([x['above_100mV']['fraction_of_total_squared_error'] for x in tail_rows])
            row['high_voltage_squared_error_fraction']=stats([x['measured_above_3.4V']['fraction_of_total_squared_error'] for x in tail_rows])
            row['outside_soc_squared_error_fraction']=stats([x['soc_outside_domain']['fraction_of_total_squared_error'] for x in tail_rows])
            for horizon in [30,60,300,900,1800]:
                for method in ['persistence','load_adjusted_persistence','daehn_openloop_known_current','daehn_ekf_known_current','daehn_ekf_persistent_current']:
                    fs=[next(x for x in r['tasks']['forecasting']['records'] if x['protocol']==protocol and x['horizon_seconds']==horizon and x['method']==method) for r in runs]
                    record={'variant':variant,'protocol':protocol,'horizon_s':horizon,'method':method,'rmse_V':stats([x['rmse_V'] for x in fs]),'empirical_band_coverage':stats([x['empirical_test_band_coverage'] for x in fs]),'band_halfwidth_V':stats([x['validation_calibrated_error_halfwidth_V'] for x in fs])}
                    if 'forecast_domain_exit_fraction' in fs[0]: record['domain_exit_fraction']=stats([x['forecast_domain_exit_fraction'] for x in fs])
                    horizons.append(record)
        for protocol in ['CCCV_18A_dis.csv','CCCV_50A_dis.csv','CCCV_180A_dis.csv']:
            for threshold in [3.2,3.,2.8]:
                es=[next(x for x in r['tasks']['discharge_energy']['threshold_predictions'] if x['method']=='daehn' and x['protocol']==protocol and x['voltage_threshold_V']==threshold) for r in runs]
                valid=[x for x in es if x['crossing_error_s'] is not None]
                events.append({'variant':variant,'protocol':protocol,'threshold_V':threshold,'split':es[0]['original_split'],'censored':len(es)-len(valid),'domain_invalid':sum(not x['prediction_within_soc_domain'] for x in es),'crossing_error_min':stats([x['crossing_error_s']/60 for x in valid]) if valid else None,'energy_relative_error':stats([x['energy_relative_error'] for x in valid if x['energy_relative_error'] is not None]) if valid else None})
    destination=BASE/'output'/'case_study_comparisons'/args.name; destination.mkdir(parents=True,exist_ok=True)
    capacity_bounds=[]
    meta=read(CASE/'data'/'metadata.json')
    with np.load(CASE/'data'/'data.npz') as a:
        for protocol in meta['protocols']:
            if protocol['split']!='test': continue
            key=protocol['id']; soc=float(a[f'soc0_{key}']); charge=a[f'charge_{key}']; bound=max(float(np.max(-charge)/(3600*(1-soc))),float(np.max(charge)/(3600*soc)))
            capacity_bounds.append({'protocol':protocol['file'],'assumed_initial_SOC':soc,'minimum_capacity_Ah_for_SOC_domain_under_current_accounting':bound,'interpretation':'Conditional on supplied initial SOC, unit charge efficiency and unbiased released current, not independently measured cell capacity.'})
    payload={'audited_training_runs':len(results),'campaigns_passed':len(results),'matched_forecast_targets':True,'dataset_sha256':dataset,'inequality_projection_added':False,'seed_statistic':'Mean ± sample standard deviation over three training seeds; not uncertainty across cells/protocols.','performance':rows,'forecasting':horizons,'events':events,'training':training,'conditional_capacity_domain_diagnostics':capacity_bounds,'runs':results}
    (destination/'metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Completed battery suite: results audit and analysis','',f'{len(results)}/{expected} training runs completed. All {len(results)} campaign verification files passed; checkpoint/dataset hashes and forecast origins/horizons/targets were independently checked. No retraining or inequality projection was performed.','',payload['seed_statistic'],'','## Held-out prediction accuracy','','| Variant | Half-cycle RMSE (mV) | Load-profile RMSE (mV) | Half-cycle EKF prior (mV) | Load-profile EKF prior (mV) |','|---|---:|---:|---:|---:|']
    find=lambda v,p:next(r for r in rows if r['variant']==v and r['protocol']==p)
    for v in variants:
        a,b=find(v,protocols[0]),find(v,protocols[1]); lines.append(f"| {v} | {fmt(a['rmse_V'],1000)} | {fmt(b['rmse_V'],1000)} | {fmt(a['ekf_prior_rmse_V'],1000)} | {fmt(b['ekf_prior_rmse_V'],1000)} |")
    lines+=['','## Forecast horizon and supplied-load assumption','','| Variant | Protocol | 1 min EKF (mV) | 1 min load-adjusted persistence (mV) | 30 min EKF (mV) | 30 min open-loop (mV) | 30 min load-adjusted persistence (mV) | 30 min persistent-load EKF (mV) |','|---|---|---:|---:|---:|---:|---:|---:|']
    f=lambda v,p,h,m:next(r for r in horizons if r['variant']==v and r['protocol']==p and r['horizon_s']==h and r['method']==m)
    for v in variants:
        for protocol in protocols:
            values=[f(v,protocol,h,m)['rmse_V'] for h,m in [(60,'daehn_ekf_known_current'),(60,'load_adjusted_persistence'),(1800,'daehn_ekf_known_current'),(1800,'daehn_openloop_known_current'),(1800,'load_adjusted_persistence'),(1800,'daehn_ekf_persistent_current')]]
            lines.append('| '+v+' | '+protocol+' | '+' | '.join(fmt(x,1000) for x in values)+' |')
    lines+=['','Known future current is supplied as a scenario, not predicted demand. The load-adjusted comparator uses each checkpoint\'s fitted R0/H; it is checkpoint-dependent. Plain persistence is common across all models.','', '## Energy and reserve events','','| Variant | Half-cycle discharge energy error (%) | Load-profile discharge energy error (%) | 50 A, 3.2 V crossing error (min) | 180 A, 3.2 V crossing error (min) | 180 A, 3.2 V energy-to-event error (%) |','|---|---:|---:|---:|---:|---:|']
    e=lambda v,p:next(r for r in events if r['variant']==v and r['protocol']==p and r['threshold_V']==3.2)
    for v in variants:
        a,b=find(v,protocols[0]),find(v,protocols[1]); va,hi=e(v,'CCCV_50A_dis.csv'),e(v,'CCCV_180A_dis.csv')
        lines.append(f"| {v} | {fmt(a['discharge_energy_relative_error'],100)} | {fmt(b['discharge_energy_relative_error'],100)} | {fmt(va['crossing_error_min']) if va['crossing_error_min'] else 'censored'} | {fmt(hi['crossing_error_min']) if hi['crossing_error_min'] else 'censored'} | {fmt(hi['energy_relative_error'],100) if hi['energy_relative_error'] else 'censored'} |")
    lines+=['','These are held-out mixed-profile throughput totals, not remaining capacity labels. The 50 A discharge was checkpoint-selection validation and 18/180 A were training; their event errors are application demonstrations, not independent EOD validation.','', '## Reliability and calibration','','| Variant | Protocol | P95 absolute error (mV) | >100 mV measurements (%) | Raw SOC violations (%) | Max SOC, seed mean | 5 s refinement discrepancy (mV) | Full-profile band coverage (%) | 30 min EKF band coverage (%) | 30 min EKF SOC-domain exits (%) |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        ff=f(r['variant'],r['protocol'],1800,'daehn_ekf_known_current')
        lines.append(f"| {r['variant']} | {r['protocol']} | {r['p95_abs_error_V']['mean']*1000:.2f} | {r['fraction_above_100mV']['mean']*100:.3f} | {r['soc_violation_fraction']['mean']*100:.3f} | {r['soc_max']['mean']:.4f} | {r['5s_refined_vs_native_voltage_rmse_V']['mean']*1000:.3f} | {r['test_empirical_band_coverage']['mean']*100:.1f} | {ff['empirical_band_coverage']['mean']*100:.1f} | {ff['domain_exit_fraction']['mean']*100:.3f} |")
    lines+=['','Empirical validation-residual bands are not formal 95% intervals. Three training seeds and one cell at one temperature cannot establish population reliability. Exact circuit/charge consistency and smaller voltage errors do not prove measured SOC accuracy.','', '## Optimization and conditional identifiability','','| Variant | Selected epochs | Training min/run | Capacity (Ah) | Validation improvement from ~2500 to final (%) | Parameter-sensitivity condition |','|---|---|---:|---:|---:|---:|']
    for r in training: lines.append(f"| {r['variant']} | {r['selected_epochs']} | {fmt(r['seconds'],1/60)} | {fmt(r['capacity_Ah'])} | {fmt(r['late_validation_improvement'],100)} | {r['conditional_sensitivity_condition']['mean']:.3g} |")
    lines+=['','Detached-OCV is a training-gradient ablation, not an equal-quality converged baseline. Near-final selection does not prove convergence. Global-parameter sensitivities hold neural weights and initial states fixed; no parameter recovery or confidence claim follows. Added RC branch and hysteresis mechanisms are proposed extensions, not the published diffusion PDE.','', '## Interpretation limits','','Do not select a universal winning model from these test protocols. Report performance separately by profile, horizon, seed, load-information assumption and application. The unconstrained readout has approximately twice as many parameters, so comparisons do not isolate projection alone. Independent SOC/capacity labels, independent discharge tests, temperature/cell transfer, and calibrated uncertainty remain missing.','', 'metrics.json stores all per-seed task results and aggregate diagnostics; plots show seed variability, forecast horizons, optimization behavior and event fragility.']
    lines+=['','## Rare-error concentration','','| Variant | Protocol | Share of squared error from >100 mV residuals (%) | Share from measured voltage >=3.4 V (%) | Share from SOC outside [0,1] (%) |','|---|---|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['variant']} | {r['protocol']} | {r['error_tail_squared_error_fraction']['mean']*100:.1f} | {r['high_voltage_squared_error_fraction']['mean']*100:.1f} | {r['outside_soc_squared_error_fraction']['mean']*100:.1f} |")
    lines+=['','These overlapping strata diagnose error concentration; their squared-error shares must not be added or interpreted as causal attribution.','', '## Conditional charge-accounting diagnostic','']
    for r in capacity_bounds: lines.append(f"- {r['protocol']}: a capacity of at least {r['minimum_capacity_Ah_for_SOC_domain_under_current_accounting']:.2f} Ah is needed to keep SOC inside [0,1] under the current initial-SOC and charge-accounting assumptions. This is not an independent capacity measurement.")
    if set(['daehn','ecm','voltage_mlp','fixed_capacity','detached_ocv','two_rc','dynamic_hysteresis','two_rc_hysteresis']).issubset(variants):
        baseline,conventional=find('daehn',protocols[0]),find('ecm',protocols[0]); primary_load=find('daehn',protocols[1]); combined_load=find('two_rc_hysteresis',protocols[1])
        gain=100*(1-baseline['rmse_V']['mean']/conventional['rmse_V']['mean']); load_gain=100*(1-combined_load['rmse_V']['mean']/primary_load['rmse_V']['mean'])
        reference_forecast=f('daehn',protocols[0],1800,'daehn_ekf_known_current'); cheap=f('daehn',protocols[0],1800,'load_adjusted_persistence')
        band=reference_forecast['band_halfwidth_V']['mean']*1000
        narrative=['# Scientific interpretation','',f'The completed suite supports a useful physically structured voltage-surrogate case study, with explicit limits on state accuracy, reserve-event accuracy and uncertainty. All {expected} runs and their campaigns completed. Model/data hashes, requested training configurations and shared forecast targets passed the review. Nothing was retrained or projected onto new inequalities.','', '## Learned constitutive modeling has measurable value','',f'DAEHN reduces half-cycle RMSE from {conventional["rmse_V"]["mean"]*1000:.2f} to {baseline["rmse_V"]["mean"]*1000:.2f} mV ({gain:.1f}%) relative to the constant-resistance ECM. Load-profile improvement is smaller. The stable three-seed result supports learning a state/current-dependent resistance; it does not isolate a causal benefit of the projection algorithm. The ECM has five parameters versus roughly 33,500 in DAEHN and trains much faster.','', '## Memory mechanisms improve different objectives','',f'Dynamic hysteresis has the best seed-mean half-cycle accuracy among the structured formulations, while two-RC plus hysteresis has the best full load-profile RMSE: {combined_load["rmse_V"]["mean"]*1000:.2f} mV, {load_gain:.1f}% below the reference DAEHN. The combined extension worsens half-cycle RMSE relative to DAEHN. There is no universal winner. Extra RC time constants and all fitted values are recorded in metrics.json; they are model-conditioned estimates, not independently recovered electrochemical parameters.','', '## Useful forecasts require a stated load scenario','',f'At 30 minutes, DAEHN half-cycle EKF forecasts have RMSE {reference_forecast["rmse_V"]["mean"]*1000:.2f} mV versus {cheap["rmse_V"]["mean"]*1000:.2f} mV for its load-adjusted persistence comparator. Dynamic hysteresis further improves the half-cycle forecast. At one minute, inexpensive load-adjusted persistence remains better. Holding the future current fixed causes large errors, especially on half cycles; this is a supplied-load scenario application, not autonomous demand forecasting.','', 'The EKF improves causal one-step voltage errors, particularly for the added-state formulations. It is not uniformly beneficial for longer forecasts: on the load profile, the combined extension has a better 30-minute open-loop forecast than its EKF-initialized forecast. State assimilation may fit local voltage while changing a latent state in a way that hurts future accuracy. Filter assumptions should be revisited on validation data, not tuned to the held-out test. Real SOC truth remains unavailable.','', '## Integrated energy and threshold decisions must be evaluated separately','', 'The reference DAEHN has about -0.55% and -0.06% discharge-throughput error on the two test profiles. The conventional ECM is already extremely accurate in integrated load-profile energy, so energy integration alone does not justify the neural model. Errors of opposite signs can cancel.','', 'Reserve-event fragility remains important. Reference DAEHN predicts the 180 A, 3.2 V crossing roughly 5.45 minutes early, corresponding to about -19.6% energy-to-that-event error. Dynamic hysteresis predicts the 50 A, 3.2 V crossing about 10 minutes early despite better average voltage errors on the test load profile. The combined extension has the same type of failure. These discharge protocols were used for training or checkpoint-selection validation; no independent held-out EOD claim is supported.','', '## Reliability limits are scientifically informative','',f'For DAEHN half cycles, {baseline["fraction_above_100mV"]["mean"]*100:.2f}% of measurements have errors above 100 mV but contribute {baseline["error_tail_squared_error_fraction"]["mean"]*100:.1f}% of total squared error. Episodes with SOC outside the physical domain account for a large overlapping share. This is association, not proof that imposing bounds would fix the voltage errors. Refinement discrepancies remain small relative to measured errors, making the model/data assumptions a more prominent concern than the tested discretization change.','', 'Structured models still exceed SOC=1, with peak half-cycle SOC around 1.04. Their fitted capacity remains near 191.5 Ah, while 199.16 Ah would be needed for half-cycle SOC feasibility under the current initialization and unit-efficiency accounting. Initial SOC, current bias, efficiency, effective capacity and constitutive inadequacy need diagnosis; the computed bound is not a measured capacity. Fixed capacity changes voltage performance little and does not resolve this issue. No SOC accuracy or capacity-recovery claim follows from a stable fitted value.','',f'Forecast bands have high test coverage but are broad: the DAEHN 30-minute absolute-error halfwidth is about {band:.0f} mV, compared with RMSE near 25 mV on half cycles and 14 mV on the load profile. Full-profile bands under-cover DAEHN half cycles at about 89%. Coverage and interval sharpness must both be reported; the present bands are not calibrated 95% operational confidence statements.','', '## Optimization findings','', 'The unconstrained voltage readout occasionally wins on half cycles, but varies strongly across seeds and performs poorly on the test load profile. It has roughly twice as many parameters and a different readout, so this is evidence about the complete structured formulation, not projection alone.','', 'Detached OCV gradients produce selected epochs 1, 1 and 3. Its validation curves initially deteriorate and end above their starting error. The saved outputs therefore represent an early-checkpoint gradient ablation, not a converged 3000-epoch competitor. Do not use it to claim that every unconstrained or alternative optimizer must fail.','', 'Most other variants select near-final checkpoints and continue improving on validation. Three thousand epochs do not establish convergence. Parameter-sensitivity results also show confounding; zero columns for R0/H in voltage_mlp explain its enormous condition number and should not be interpreted as a numerical failure of the whole neural model.','', '## A defensible case-study claim','', 'A literature-based, physically structured battery DAE with learned constitutive behavior can improve voltage prediction and longer-horizon supplied-load forecasts relative to a calibrated conventional circuit. Memory extensions improve selected operating regimes. Exact algebraic consistency and low integrated-energy error remain insufficient for valid latent SOC, reliable reserve crossings or sharp calibrated uncertainty.','', 'The next priorities are independent SOC/capacity and full-discharge validation, training-window/initial-state sensitivity for slow memory, diagnosis of high-voltage/relaxation tail episodes, and validation-only filter/interval calibration. Inequalities remain a separate later experiment. Keep conclusions limited to this cell, temperature, split and supplied-load information.','', 'See report.md for seed aggregates and metrics.json for full per-seed provenance and diagnostics.']
        (destination/'interpretation.md').write_text('\n'.join(narrative),encoding='utf-8')
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    colors=plt.get_cmap('tab10').colors
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    for j,protocol in enumerate(protocols):
        means=[find(v,protocol)['rmse_V']['mean']*1000 for v in variants]; errors=[find(v,protocol)['rmse_V']['std']*1000 for v in variants]
        ax[j].bar(range(len(variants)),means,yerr=errors,capsize=3,color=colors[:len(variants)]); ax[j].set_xticks(range(len(variants)),variants,rotation=45,ha='right'); ax[j].set_ylabel('Voltage RMSE (mV), seed mean ± SD'); ax[j].set_title(protocol)
    fig.tight_layout(); fig.savefig(destination/'accuracy_seed_variability.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for j,protocol in enumerate(protocols):
        for v in ['daehn','ecm','two_rc','dynamic_hysteresis','two_rc_hysteresis']:
            ax[j].plot([30,60,300,900,1800],[f(v,protocol,h,'daehn_ekf_known_current')['rmse_V']['mean']*1000 for h in [30,60,300,900,1800]],marker='o',label=v)
        ax[j].plot([30,60,300,900,1800],[f('daehn',protocol,h,'load_adjusted_persistence')['rmse_V']['mean']*1000 for h in [30,60,300,900,1800]],ls='--',label='DAEHN load-adjusted persistence'); ax[j].set_xscale('log'); ax[j].set_xlabel('Horizon (s), supplied future current'); ax[j].set_ylabel('Mean voltage RMSE (mV)'); ax[j].set_title(protocol); ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(destination/'forecast_horizons.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for v in variants:
        for r in [x for x in results if x['variant']==v]:
            history=read(BASE/'output'/Path(r['model_dir']).relative_to(BASE/'model')/'training_history.json')
            ax[0 if v=='detached_ocv' else 1].plot([x['epoch'] for x in history],[np.sqrt(x['validation_scaled_voltage_mse'])*50 for x in history],label=f"{v}:{r['seed']}",alpha=.7)
    for a in ax: a.set_xlabel('Epoch'); a.set_ylabel('Conditional validation voltage RMSE (mV)'); a.legend(fontsize=5,ncol=2)
    ax[0].set_title('Detached OCV gradient'); ax[1].set_title('Other formulations'); fig.tight_layout(); fig.savefig(destination/'validation_learning_curves.png',dpi=150); plt.close(fig)
    print(destination/'report.md')

if __name__=='__main__': main()
