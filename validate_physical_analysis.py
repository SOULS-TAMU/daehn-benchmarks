"""Verify the saved-model audit coverage, paired grids, hashes, and JSON output."""
from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent
def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value
def main():
    manifests=[p for p in (ROOT/'results').rglob('checkpoint_manifest.json') if '08_cstr_mpc' not in p.parts]
    checked=0; variants=0; dataset_grids={}; issues=[]; case_counts={}
    for mp in manifests:
        manifest=json.loads(mp.read_text()); run=mp.parent; case=next(p for p in run.parents if p.parent==ROOT/'results'); dest=case/'output'/'physical_analysis'
        actual=hashlib.sha256((run/manifest['checkpoint_file']).read_bytes()).hexdigest()
        if actual!=manifest['checkpoint_sha256']: issues.append('checkpoint changed: '+str(run))
        if manifest.get('task')=='inverse':
            if not (dest/'inverse'/'metrics.json').exists(): issues.append('missing inverse analysis')
            continue
        path=dest/run.relative_to(case/'model'); metrics_path=path/'metrics.json'
        if not metrics_path.exists(): issues.append('missing metrics: '+str(path)); continue
        data=json.loads(metrics_path.read_text()); checked+=1; variants+=len(data['variants']); case_counts[case.name]=case_counts.get(case.name,0)+1
        if data.get('analysis_version')!=4: issues.append('old schema: '+str(path))
        with np.load(path/'trajectory_predictions.npz') as source: arrays={k:source[k] for k in source.files}
        arrays['exact_physical']=arrays['exact'].copy()
        if manifest['example']==17: arrays['exact_physical'][:,0]=np.exp(np.clip(arrays['exact'][:,0],-60,5))
        for row in data['variants']:
            arrays[row['variant']+'__physical_prediction']=arrays[row['variant']+'__prediction'].copy()
            if manifest['example']==17: arrays[row['variant']+'__physical_prediction'][:,0]=np.exp(np.clip(arrays[row['variant']+'__prediction'][:,0],-60,5))
        np.savez_compressed(path/'trajectory_predictions.npz',**arrays)
        with np.load(path/'trajectory_predictions.npz') as arch:
            coord=arch['coordinates']; exact=arch['exact']; key=(case.name,data['trajectory_data_sha256'])
            if key in dataset_grids:
                previous=dataset_grids[key]
                if not np.array_equal(previous[0],coord) or not np.allclose(previous[1],exact,rtol=1e-12,atol=1e-12): issues.append('paired grid/reference differs: '+str(path))
            else: dataset_grids[key]=(coord,exact)
            for row in data['variants']:
                row['mse_derivative']=row['full_map_derivative_rmse']**2
                pred=arch[row['variant']+'__prediction']
                if pred.shape!=exact.shape or not np.isfinite(pred).all(): issues.append('invalid prediction: '+str(path))
                if 'grid_total_differential_rmse_interior' not in row: issues.append('missing grid residual: '+str(path))
                if manifest['example']==17 and 'concentration_decode_clip_fraction' not in row: issues.append('missing decoding diagnostic: '+str(path))
        if not (path/'comparison.png').exists(): issues.append('missing comparison plot: '+str(path))
        metrics_path.write_text(json.dumps(clean(data),indent=2,allow_nan=False))
    result={'checkpoint_hashes_checked':len(manifests),'forward_checkpoints_analyzed':checked,'forward_variants_evaluated':variants,'inverse_checkpoints_analyzed':1,'case_checkpoint_counts':case_counts,'paired_dataset_grids_checked':len(dataset_grids),'issues':issues}
    output=ROOT/'results'/'physical_analysis'; output.mkdir(exist_ok=True); (output/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    for report in (ROOT/'results').glob('*/output/physical_analysis/**/*.md'):
        raw=report.read_bytes()
        try: content=raw.decode('utf-8')
        except UnicodeDecodeError: content=raw.decode('cp1252')
        report.write_text(content,encoding='utf-8')
    if issues: raise RuntimeError('Validation issues detected')
if __name__=='__main__': main()
