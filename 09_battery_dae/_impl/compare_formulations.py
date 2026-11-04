"""Compare trained baselines/formulations; report budgets and application failures."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign-dir',type=Path,action='append',required=True); p.add_argument('--name',default='baseline_formulation_comparison'); args=p.parse_args()
    if Path(args.name).name!=args.name: p.error('name must be a folder name')
    reference=None; dataset=None; rows=[]; configs=[]
    for directory in args.campaign_dir:
        directory=directory.resolve()
        if not directory.is_relative_to((BASE/'output').resolve()): p.error('Campaign must be inside case outputs')
        report=json.loads((directory/'metrics.json').read_text()); manifest=json.loads((directory/'campaign_manifest.json').read_text()); config=json.loads((Path(report['model_dir'])/'run_config.json').read_text())
        if dataset is None: dataset=manifest['dataset_sha256']
        if dataset!=manifest['dataset_sha256']: raise ValueError('Different dataset hashes')
        with np.load(directory/'forecasting'/'trajectory_predictions.npz') as a:
            targets={k:a[k] for k in a.files if k.endswith(('__origin_time_s','__exact_voltage_V','__horizon_s'))}
        if reference is None: reference=targets
        elif reference.keys()!=targets.keys() or any(not np.array_equal(reference[k],targets[k]) for k in targets): raise ValueError('Different forecast origins, horizons or targets')
        config.update(campaign=str(directory),model_dir=report['model_dir']); configs.append(config)
        results=report['results']; forecast=results['forecasting']['records']; energy=results['discharge_energy']['heldout_energy_accounting']
        events=[r for r in results['discharge_energy']['threshold_predictions'] if r['method']=='daehn']
        for r in results['reliability']['protocols']:
            e=next(e for e in energy if e['protocol']==r['protocol']); f=next(f for f in forecast if f['protocol']==r['protocol'] and f['method']=='daehn_ekf_known_current' and f['horizon_seconds']==1800)
            rows.append({'variant':config.get('variant','daehn'),'seed':config['seed'],'activation':config.get('activation','tanh'),'hidden_sizes':config['hidden_sizes'],'epochs':config['epochs'],'selected_epoch':config['selected_epoch'],'parameter_count':config.get('parameter_count'),'protocol':r['protocol'],'native_voltage_rmse_V':r['rmse_V'],'forecast_1800s_rmse_V':f['rmse_V'],'forecast_1800s_domain_exit_fraction':f['forecast_domain_exit_fraction'],'discharge_energy_relative_error':e['discharge_energy_relative_error'],'soc_violation_fraction':r['soc_violation_fraction'],'circuit_residual_max_V':r['algebraic_max_abs_residual_V'],'model_dir':report['model_dir'],'worst_cutoff_error_s':max((abs(x['crossing_error_s']) for x in events if x['crossing_error_s'] is not None),default=None),'censored_cutoff_count':sum(x['crossing_error_s'] is None for x in events)})
    out=BASE/'output'/'case_study_comparisons'/args.name; out.mkdir(parents=True,exist_ok=True)
    matched_fields=['hidden_sizes','activation','epochs','learning_rate','batch_size','window_length','steps_per_epoch','capacity_prior_weight','soc_bound_weight','validate_every']
    groups={}
    grouping_fields=matched_fields+['fixed_capacity_Ah','hysteresis_rate','hysteresis_relaxation_seconds','validate_every']
    for row in rows:
        config=next(c for c in configs if c['model_dir']==row['model_dir'])
        signature=hashlib.sha256(json.dumps({k:config.get(k) for k in grouping_fields},sort_keys=True).encode()).hexdigest()[:12]
        row['configuration_group']=signature; groups.setdefault((row['variant'],row['protocol'],signature),[]).append(row)
    aggregates=[{'variant':v,'protocol':p,'configuration_group':signature,'runs':len(g),'distinct_seeds':len(set(x['seed'] for x in g)),'native_rmse_mean_V':float(np.mean([x['native_voltage_rmse_V'] for x in g])),'native_rmse_std_V':float(np.std([x['native_voltage_rmse_V'] for x in g],ddof=1)) if len(g)>1 else None,'forecast_rmse_mean_V':float(np.mean([x['forecast_1800s_rmse_V'] for x in g]))} for (v,p,signature),g in groups.items()]
    budgets_matched=all(all(c.get(k)==configs[0].get(k) for k in matched_fields) for c in configs)
    save_json(out/'metrics.json',{'rows':rows,'aggregate_by_variant_protocol':aggregates,'configurations':configs,'matched_dataset_and_forecast_targets':True,'common_training_settings':budgets_matched,'matched_setting_fields':matched_fields,'inequality_projection_added':False,'interpretation':'Formulation and learned-readout comparisons; architecture/state dimensions and parameter counts differ. Not a pure causal projection ablation. Same optimizer budgets do not guarantee convergence. No measured SOC labels; discharge cutoff records are training/validation demonstrations.'})
    lines=['# Trained baseline and formulation comparisons','',f'Common requested training settings: {budgets_matched}. Exact test origins, horizons and targets agree. Full run configurations are stored in metrics.json.','', '| Variant | Seed | Activation | Hidden sizes | Parameters | Epochs / selected | Protocol | Native RMSE (mV) | 30 min forecast (mV) | Discharge energy error (%) | SOC violations (%) |','|---|---:|---|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['variant']} | {r['seed']} | {r['activation']} | {r['hidden_sizes']} | {r['parameter_count']} | {r['epochs']} / {r['selected_epoch']} | {r['protocol']} | {1000*r['native_voltage_rmse_V']:.3f} | {1000*r['forecast_1800s_rmse_V']:.3f} | {100*r['discharge_energy_relative_error']:.3f} | {100*r['soc_violation_fraction']:.3f} |")
    lines+=['','Known future current is supplied for these forecasts. Consult individual campaigns for short horizons, persistent-current forecasts, error bands, censored events and domain-invalid forecasts.','', 'Circuit residuals for voltage_mlp are unenforced discrepancies. Constant-R1 ECM has no hidden network. Two-RC is a slow-relaxation surrogate, not an implementation of the later paper\'s diffusion PDE. Dynamic hysteresis is a proposed extension. Fixed-capacity and detached-gradient variants are ablations.','', 'Cutoff-error statistics include training/validation profiles and must not be described as independent EOD validation. Parameter-count and state-dimension differences preclude a pure projection-effect claim. Seed aggregates are descriptive and are meaningful only for otherwise matched configurations.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(12,4)); labels=[f"{r['variant']}:{r['seed']}\n{r['protocol'].split('.')[0]}" for r in rows]
    ax[0].bar(range(len(rows)),[1000*r['native_voltage_rmse_V'] for r in rows]); ax[1].bar(range(len(rows)),[1000*r['forecast_1800s_rmse_V'] for r in rows])
    for a in ax: a.set_xticks(range(len(rows)),labels,rotation=90,fontsize=6); a.set_ylabel('Voltage RMSE (mV)')
    ax[0].set_title('Full held-out profiles'); ax[1].set_title('30 min forecasts with supplied future current'); fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=140); plt.close(fig)
    print(out/'report.md')

if __name__=='__main__': main()
