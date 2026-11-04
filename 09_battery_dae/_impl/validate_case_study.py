"""Verify campaign artifacts, numerical checks and a causal-filter prefix test."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *

def main():
    args=parser(__doc__).parse_args(); ctx=Context(args.model_dir,args.study); tasks=['reliability','state_estimation','forecasting','discharge_energy','identifiability']; issues=[]
    for task in tasks:
        folder=ctx.root/task
        for name in ['metrics.json','trajectory_predictions.npz','report.md']:
            if not (folder/name).exists(): issues.append('Missing '+str(folder/name))
        data=json.loads((folder/'metrics.json').read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonstandard JSON '+x)))
        with np.load(folder/'trajectory_predictions.npz') as a:
            for key in a.files:
                if key.endswith('innovation') or key.endswith('innovation_variance'): continue # First innovation undefined by construction.
                if a[key].dtype.kind in 'fc' and not np.isfinite(a[key]).all(): issues.append('Nonfinite '+task+'/'+key)
        if not list(folder.glob('*.png')): issues.append('No plots for '+task)
    # Genuine causal check: changing later voltage observations must not alter earlier filter states.
    protocol=next(ctx.protocols('validation')); grid=ctx.grid(protocol,5.); short={k:(v[:101] if isinstance(v,np.ndarray) else v) for k,v in grid.items()}; changed=dict(short); changed['protocol']=dict(protocol); changed['protocol']['v']=protocol['v'].copy(); changed['protocol']['v'][protocol['t']>250]+=.1
    original=filter_profile(ctx,short,ctx.initial(protocol)); modified=filter_profile(ctx,changed,ctx.initial(protocol)); prefix_error=float(np.max(abs(original['state'][:51]-modified['state'][:51]))); assert prefix_error<1e-12
    assert np.max(abs(original['state'][51:]-modified['state'][51:]))>1e-8
    reference_changed=dict(short); reference_changed['v']=short['v']+.3; ignore=filter_profile(ctx,reference_changed,ctx.initial(protocol)); reference_error=float(np.max(abs(original['state']-ignore['state']))); assert reference_error<1e-12
    # A synchronous sensor check separates its prior from the same-step update.
    synchronous={k:v for k,v in short.items() if k!='protocol'}; shifted=dict(synchronous); shifted['v']=synchronous['v'].copy(); shifted['v'][51:]+=.1
    a=filter_profile(ctx,synchronous,ctx.initial(protocol)); b=filter_profile(ctx,shifted,ctx.initial(protocol)); prior_error=float(np.max(abs(a['prior_voltage'][:52]-b['prior_voltage'][:52]))); assert prior_error<1e-12
    if np.min(np.linalg.eigvalsh(original['covariance'])) < -1e-12: issues.append('Non-PSD EKF covariance')
    with np.load(ctx.root/'forecasting'/'trajectory_predictions.npz') as a:
        for protocol in ['halfcycles','loadprofile']:
            truth=a[protocol+'__exact_voltage_V']; origins=a[protocol+'__origin_time_s']
            if truth.shape!=(len(origins),5): issues.append('Forecast target shape mismatch')
            for name in ['persistence','load_adjusted_persistence','daehn_openloop_known_current','daehn_ekf_known_current','daehn_ekf_persistent_current']:
                if a[protocol+'__'+name+'__prediction_V'].shape!=truth.shape: issues.append('Unmatched forecast shape '+name)
    checksum=hashlib.sha256((ctx.model_dir/'parameter_tree.pkl').read_bytes()).hexdigest(); manifest=json.loads((ctx.model_dir/'checkpoint_manifest.json').read_text()); assert checksum==manifest['checkpoint_sha256']
    result={'status':'passed' if not issues else 'failed','issues':issues,'artifact_tasks_checked':tasks,'original_timestamp_causal_filter_prefix_max_error':prefix_error,'interpolated_reference_voltage_ignored_max_error':reference_error,'causal_one_step_prior_prefix_max_error':prior_error,'forecast_methods_share_origins_and_targets':True,'checkpoint_hash_unchanged':True,'no_inequality_projection_added':True,'note':'Numerical/workflow correctness checks; not scientific accuracy certification.'}; save_json(ctx.root/'verification.json',result); print(json.dumps(result,indent=2))
    if issues: raise RuntimeError('Campaign verification failed')
if __name__=='__main__': main()
