"""Local observability, constitutive surfaces and parameter-sensitivity diagnostics."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *

def run(ctx):
    out=ctx.output('identifiability'); z=np.linspace(.001,.999,201); currents=np.array([-180.,-50.,0.,50.,180.]); records=[]; sigma=np.zeros((len(currents),len(z))); condition=sigma.copy(); resistance=sigma.copy(); slope=sigma.copy()
    for k,current in enumerate(currents):
        for j,soc in enumerate(z):
            state=jnp.asarray(initial_state(ctx.saved.config,soc)); F=np.asarray(ctx.F(state,current,5.)); H=np.asarray(ctx.H(state,current)); scales=np.array([.05]*ctx.saved.state_dim)
            if 'hysteresis_state' in ctx.saved.state_names: scales[-1]=.2
            normalized=np.vstack([H@np.linalg.matrix_power(F,k) for k in range(ctx.saved.state_dim)])@np.diag(scales)/.02; sv=np.linalg.svd(normalized,compute_uv=False); sigma[k,j]=sv[-1]; condition[k,j]=sv[0]/max(sv[-1],1e-30); slope[k,j]=H[0]
        resistance[k]=np.asarray(jax.jit(ctx.saved.r1)(ctx.p,jnp.asarray(z),jnp.full((len(z),),current)))
        records.append({'current_A':current,'minimum_normalized_observability_singular_value':float(sigma[k].min()),'median_condition_number':float(np.median(condition[k])),'maximum_condition_number':float(condition[k].max()),'median_OCV_slope_V_per_SOC':float(np.median(slope[k]))})
    readout_slope=slope.copy()
    physical_slope=np.asarray(jax.vmap(jax.grad(lambda zz:jnp.interp(zz,jnp.asarray(ctx.saved.ox),jnp.asarray(ctx.saved.oy))))(jnp.asarray(z)))
    slope=np.tile(physical_slope,(len(currents),1))
    for row in records:
        k=list(currents).index(row['current_A']); row['median_OCV_slope_V_per_SOC']=float(np.median(slope[k])); row['median_model_voltage_SOC_derivative']=float(np.median(readout_slope[k]))
    names=['capacity','capacitance','series_resistance','hysteresis']; all_sensitivity=[]; parameter_rows=[]; arrays={'soc_grid':z,'current_grid_A':currents,'R1_ohm':resistance,'OCV_slope_V_per_SOC':slope,'model_voltage_SOC_derivative':readout_slope,'normalized_observability_min_singular_value':sigma,'observability_condition':condition}
    if 'r1_raw' in ctx.p: names+=['branch_resistance']
    if 'r2_raw' in ctx.p: names+=['slow_resistance','slow_capacitance']
    for p in ctx.protocols('validation'):
        columns=[]; base=ctx.saved.predict(p['i'],p['t'],p['soc0'])
        for name in names:
            plus=predict_parameters(ctx,parameter_copy(ctx,name,1.01),p); minus=predict_parameters(ctx,parameter_copy(ctx,name,.99),p); columns.append((plus[:,-1]-minus[:,-1])/2) # volts for one percentage-point parameter change.
            parameter_rows.append({'protocol':p['file'],'parameter':name,'voltage_rms_change_for_one_percent_parameter_change_V':rmse(columns[-1]),'plus_1pct_voltage_rmse_V':rmse(plus[:,-1]-p['v']),'minus_1pct_voltage_rmse_V':rmse(minus[:,-1]-p['v'])})
        S=np.column_stack(columns); ids=np.linspace(0,len(S)-1,min(1000,len(S)),dtype=int); all_sensitivity.append(S[ids]/.02); arrays[p['file'][:-4]+'__time_s']=p['t']; arrays[p['file'][:-4]+'__voltage_sensitivity_per_percentage_point']=S
    S=np.vstack(all_sensitivity); sv=np.linalg.svd(S,compute_uv=False); norms=np.linalg.norm(S,axis=0); normalized=S/np.maximum(norms,1e-30); gram=normalized.T@normalized
    metrics={'two_sample_local_observability':records,'state_scaling_for_observability':[.05,.05],'measurement_scaling_V':.02,'conditional_parameter_sensitivity':parameter_rows,'parameter_order':names,'sensitivity_singular_values':sv,'sensitivity_condition_number':float(sv[0]/max(sv[-1],1e-30)),'sensitivity_column_cosine_similarity':gram,'identifiability_interpretation':'Local to the fitted parameter values, neural R1 law and assumed initial states. Not a proof of global identifiability, parameter recovery or a statistical confidence interval.','soc_truth_available':False,'measurement_information':'OCV slope and the two-sample linearized observability matrix expose weak SOC information on the LFP plateau.'}
    arrays.update(parameter_sensitivity_matrix_scaled=S,parameter_column_cosine_similarity=gram)
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for k,current in enumerate(currents): ax[0].plot(z,resistance[k]*1000,label=f'{current:g} A'); ax[1].semilogy(z,sigma[k],label=f'{current:g} A')
    ax[0].set_xlabel('SOC'); ax[0].set_ylabel('R1 (mOhm)'); ax[0].legend(fontsize=7); ax[1].set_xlabel('SOC'); ax[1].set_ylabel('Local observability min singular value'); im=ax[2].imshow(gram,vmin=-1,vmax=1,cmap='coolwarm'); ax[2].set_xticks(range(len(names)),names,rotation=45,ha='right'); ax[2].set_yticks(range(len(names)),names); fig.colorbar(im,ax=ax[2]); fig.tight_layout(); fig.savefig(out/'observability_sensitivity.png',dpi=140); plt.close(fig)
    lines=['# State/parameter identifiability diagnostics','',metrics['identifiability_interpretation'],'',metrics['measurement_information'],'','The two-sample matrix is [H; HF] at fixed current and dt=5 s. It is scaled by representative state perturbations and a 20 mV discrepancy scale. It does not represent a complete experiment observability Gramian.','',f"Conditional global-parameter sensitivity condition number: {metrics['sensitivity_condition_number']:.6g}.",'','Sensitivity calculations use only validation protocols. Neural weights and initial SOC are held fixed; allowing them to vary can further weaken identifiability. No parameter uncertainty interval is inferred from this diagnostic.','', '| Parameter | Protocol | RMS voltage response to 1% change (mV) |','|---|---|---:|']
    for r in parameter_rows: lines.append(f"| {r['parameter']} | {r['protocol']} | {r['voltage_rms_change_for_one_percent_parameter_change_V']*1000:.5f} |")
    metrics.update(state_order=ctx.saved.state_names,state_scaling_for_observability=scales,observability_samples=ctx.saved.state_dim,variant=ctx.saved.config.get('variant','daehn'))
    lines+=['',f'Extended variants use {ctx.saved.state_dim} local samples [H; HF; ...] to match their state dimension. The stored legacy two_sample_local_observability key denotes these rows for compatibility. Fixed capacity perturbations are hypothetical scenarios, not a fitted-capacity uncertainty. voltage_mlp R0/H are not used in its inference readout; its state observability is conditional on learned features, not physical SOC validation.']
    return ctx.report('identifiability',metrics,lines,arrays)
if __name__=='__main__':
    args=parser(__doc__).parse_args(); print(run(Context(args.model_dir,args.study)))
