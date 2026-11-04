"""Summarize completed physical audits without changing saved training results."""
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parent
if not (ROOT/'results').exists(): ROOT=Path(r'D:\Projects\daehn-benchmarks')
FIELDS={'01_lotka_volterra':['invariant_max_error','phase_rmse','period','reference_period','nonpositive_fraction'], '02_mass_spring':['energy_max_error','amplitude_rmse','phase_rmse','frequency_error'],'03_two_body':['angular_momentum_max_error','independent_energy_max_error','energy_drift','radial_rmse','orbital_phase_rmse','endpoint_closure','reference_endpoint_closure'],'04_robot_arm':['end_effector_path_rmse','joint_velocity_rmse','path_velocity_consistency_rmse','minimum_jacobian_singular_value','maximum_jacobian_condition'],'05_rigid_body':['energy_max_error','momentum_squared_max_error','precession_phase_rmse','minimum_constraint_singular_value','maximum_constraint_condition'],'06_cstr':['temperature_rmse','negative_concentration_fraction','terminal_window_rmse','terminal_velocity_norm'],'07_catalytic_oxidation':['log_concentration_rmse','internal_log_y1_rmse','concentration_decode_clip_fraction','unclipped_y1_rmse','nonpositive_fraction','conversion_error']}
def main():
    overview=['# Saved-model physical case analyses','', 'Independent dense reference integration on the saved time interval. Median metrics across five matched seeds; catalytic results are separated by saved dataset. Inverse fit is a reconstruction study. No retraining or MPC rerun.','']
    summary={}
    for case,fields in FIELDS.items():
        folder=ROOT/'results'/case/'output'/'physical_analysis'; manifests=list(folder.glob('*/*/*/metrics.json')); groups={}; contracts={}; metadata={}; audits=[]
        for file in manifests:
            data=json.loads(file.read_text()); key=data.get('trajectory_data_sha256','legacy')
            if 'delta_sweep' in file.parts:
                cfg=json.loads((Path(data['variants'][0]['run'])/'run_config.json').read_text()); key += ':delta='+str(cfg.get('delta'))
            groups.setdefault(key,[]).extend(data['variants']); contracts[key]=data['contract']
            with np.load(file.parent/'trajectory_predictions.npz') as a: parameters=a['physical_parameters'][0].tolist()
            metadata[key]={'time_window':data['time_window'],'first_physical_parameter_row':parameters,'grid_points':data['grid_points'],'reference_vs_saved_interpolation_max':data.get('reference_vs_saved_interpolation_max')}
        if not groups: continue
        lines=['# '+case+' physical findings','', 'All evaluations use the saved model input contract. Independent DOP853 reference starts from the first saved trajectory state; this is an interpolation-window test, with no new initial conditions or parameters.','']
        for dataset,rows in groups.items():
            label=str(dataset)[:12]+(' ('+dataset.split(':',1)[1]+')' if ':' in dataset else '')
            lines+=['Saved evaluation window: '+str(metadata[dataset]['time_window'])+'. First physical parameter row: '+str(metadata[dataset]['first_physical_parameter_row'])+'.','']
            lines+=['## Dataset '+label,'','| Variant | n | State RMSE | Differential mean norm | Algebraic mean norm | Full-map derivative RMSE | Inference seconds/batch |','|---|---:|---:|---:|---:|---:|---:|']
            models={}
            for name in sorted(set(v['variant'] for v in rows)):
                vals=[v for v in rows if v['variant']==name]; aggregate={}
                for field in ['state_rmse','differential_violation','algebraic_violation','full_map_derivative_rmse','grid_total_differential_rmse_interior','inference_time_seconds_per_batch',*fields,'constraint_minimum_singular_value','constraint_maximum_condition','differentiated_constraint_rmse','projection_failure_fraction','mean_projection_iterations','auxiliary_vs_full_map_derivative_rmse']:
                    numbers=[v.get(field) for v in vals if isinstance(v.get(field),(int,float))]; aggregate[field]=float(np.median(numbers)) if numbers else None
                models[name]=aggregate
                lines.append('| '+name+' | '+str(len(vals))+' | '+' | '.join(f'{aggregate[k]:.6g}' for k in ['state_rmse','differential_violation','algebraic_violation','full_map_derivative_rmse','inference_time_seconds_per_batch'])+' |')
            lines+=['','| Diagnostic | '+ ' | '.join(models)+' |','|---|'+'---:|'*len(models)]
            for field in fields+['constraint_minimum_singular_value','constraint_maximum_condition','differentiated_constraint_rmse','grid_total_differential_rmse_interior','projection_failure_fraction','auxiliary_vs_full_map_derivative_rmse']:
                lines.append('| '+field.replace('_',' ')+' | '+' | '.join('unavailable' if m[field] is None else f'{m[field]:.6g}' for m in models.values())+' |')
            dv=[v for v in rows if v['variant']=='daehn']
            if dv:
                worst=max(dv,key=lambda v:v['state_rmse']); best=min(dv,key=lambda v:v['state_rmse']); lines+=['',f"DAEHN state RMSE ranges from {best['state_rmse']:.6g} (seed {best['seed']}) to {worst['state_rmse']:.6g} (seed {worst['seed']}). The worst run's algebraic mean norm is {worst['algebraic_violation']:.6g}; this must be considered separately from its state error."]
                if 'mlp' in models and 'pinn' in models:
                    dm=models['daehn']['state_rmse']; lines.append(f"DAEHN median RMSE is {dm/models['mlp']['state_rmse']:.3g} times the raw MLP median and {dm/models['pinn']['state_rmse']:.3g} times the raw PINN median.")
                    for raw in ('mlp','pinn'):
                        ratio=models[raw+'_projected']['state_rmse']/models[raw]['state_rmse']; lines.append(f"Applying the saved projection to {raw.upper()} changes its median state RMSE by {(ratio-1)*100:.2f}%. Projected {raw.upper()} median remains {models[raw+'_projected']['state_rmse']/dm:.3g} times DAEHN's median.")
                else: lines.append('This saved projection-step sweep has only DAEHN checkpoints; raw-baseline comparisons are in the main matched-dataset group.')
            # Audit saved contracts and compare each model only to the matching seed/dataset.
            pairing=[]
            for seed in sorted(set(v['seed'] for v in rows)):
                seedrows=[v for v in rows if v['seed']==seed]; configs=[]
                for row in seedrows:
                    if row['variant'].endswith('_projected'): continue
                    manifest=json.loads((Path(row['run'])/'checkpoint_manifest.json').read_text()); config=json.loads((Path(row['run'])/'run_config.json').read_text())
                    configs.append({'variant':row['variant'],'trajectory_hash':manifest.get('trajectory_data_sha256'),'delta':config.get('delta'),'tolerance':config.get('tolerance'),'max_iterations':config.get('max_iterations'),'derivative_steps':manifest.get('derivative_steps')})
                pairing.append({'seed':seed,'models':configs,'same_saved_trajectory':len(set(v['trajectory_hash'] for v in configs))==1,'same_projection_hyperparameters':len(set(str((v['delta'],v['tolerance'],v['max_iterations'],v['derivative_steps'])) for v in configs))==1})
            lines+=['','Saved-trajectory hashes and projection settings checked for paired seeds. See metrics.json for pairing audit.','']
            if all(v['same_saved_trajectory'] for v in pairing): lines.append('All available raw and DAEHN models in this group share the identical saved trajectory.')
            if not all(v['same_projection_hyperparameters'] for v in pairing): lines.append('Projection configurations differ between some saved models; baseline projection comparisons are not a single universal projection ablation.')
            summary.setdefault(case,{})[dataset]={'models':models,'pairing':pairing,'contract':contracts[dataset],'evaluation':metadata[dataset]}
            lines.append('')
        lines+=['## Interpretation and limitations','', 'Algebraic/invariant residuals do not establish trajectory accuracy. Continuous residuals use derivatives of the full projected state map; auxiliary derivative agreement is reported separately. Differentiated constraints use finite differences along the evaluation grid and can include endpoint/discontinuity effects. Projection convergence flags apply to every reported projected variant; a poor or nonconverged baseline must not be read as a successful physical correction.','', 'Period estimates require at least two detected maxima; unavailable period estimates reflect the finite saved window. Extrema and peak timing are grid-based. Inference timing comes from the existing warmed evaluation helper and excludes initial compilation; no dedicated standalone Jacobian timing was performed. Constraint conditioning is measured from physical constraint Jacobians, not the full projection KKT matrix. Projection iterations, solver residuals, and failures provide separate solver diagnostics.','']
        if case=='03_two_body': lines+=['The enforced projection constraint is angular momentum only. Energy diagnostics are independent; endpoint closure is compared with reference endpoint closure because the stored interval may omit a complete orbit.']
        if case=='05_rigid_body': lines+=['Precession phase is the angle of the omega1–omega2 trajectory. Spatial orientation/attitude is not a model output and was not reconstructed.']
        if case=='06_cstr': lines+=['Temperature is the benchmark dimensionless theta variable. Concentrations are checked for negativity. Terminal-window accuracy/velocity describe transients under time-varying inputs, not an equilibrium experiment. No asserted upper physical bound is imposed unless supplied by the saved model.','The saved derivative helper holds physical parameters fixed while varying time. The additional grid_total_differential_rmse_interior differentiates along the actual prescribed input profile and excludes endpoints and neighborhoods of input transitions at t=20,25,40. Its finite-grid truncation error must be considered.']
        if case=='07_catalytic_oxidation': lines+=['The first network output is log(y1). Primary state metrics use the training script physical decoder exp(clip(log_y1,-60,5)). The older plotting registry uses unclipped exponentiation; the unclipped y1 error and clipping fraction are reported separately to expose pathological internal outputs. Both algebraic residual components and per-species relative errors are in run-level metrics. Each parameter regime is a separately trained time-to-trajectory model, not a parameter-conditioned model.']
        (folder/'report.md').write_text('\n'.join(lines)); (folder/'metrics.json').write_text(json.dumps(summary[case],indent=2))
        overview+=['## '+case,'',f"[Detailed report]({(folder/'report.md').as_posix()})",'']
        for dataset,group in summary[case].items():
            overview.append('Dataset '+dataset[:12]+(' '+dataset.split(':',1)[1] if ':' in dataset else '')+': '+', '.join(name+' state RMSE='+f"{metrics['state_rmse']:.6g}" for name,metrics in group['models'].items()))
        overview.append('')
    inverse_folder=ROOT/'results'/'01_lotka_volterra'/'output'/'physical_analysis'/'inverse'
    if (inverse_folder/'metrics.json').exists():
        im=json.loads((inverse_folder/'metrics.json').read_text()); summary['inverse_lotka_volterra']=im
        overview+=['## Inverse Lotka–Volterra','',f"[Detailed report]({(inverse_folder/'report.md').as_posix()})",'',f"Maximum relative coefficient error: {max(im['parameter_relative_error'])*100:.6g}%. Dense reconstruction RMSE: {im['dense_state_rmse']:.6g}. Continuous residual RMSE: {im['continuous_residual_rmse']:.6g}. Both integral design matrices have rank 2. Only a saved DAEHN inverse checkpoint is available; all observations were used in fitting."]
    dest=ROOT/'results'/'physical_analysis'; dest.mkdir(exist_ok=True); (dest/'metrics.json').write_text(json.dumps(summary,indent=2)); (dest/'report.md').write_text('\n'.join(overview)); print('\n'.join(overview))
if __name__=='__main__': main()
