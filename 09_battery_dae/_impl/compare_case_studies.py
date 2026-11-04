"""Compare completed campaigns from longer training or different seeds."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from study_common import *

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign-dir',type=Path,action='append',required=True); p.add_argument('--name',default='checkpoint_comparison'); args=p.parse_args()
    if Path(args.name).name!=args.name: p.error('name must be a folder name')
    rows=[]; datasets=set(); reference=None
    for directory in args.campaign_dir:
        directory=directory.resolve()
        if not directory.is_relative_to((BASE/'output').resolve()): p.error('Campaign outside case output branch')
        manifest=json.loads((directory/'campaign_manifest.json').read_text()); datasets.add(manifest['dataset_sha256']); data=json.loads((directory/'metrics.json').read_text()); config=json.loads((Path(data['model_dir'])/'run_config.json').read_text()); rows.append({'campaign':str(directory),'epochs':data['epochs_executed'],'selected_epoch':data['selected_epoch'],'steps_per_epoch_limit':config.get('steps_per_epoch'),'checkpoint_sha256':data['checkpoint_sha256'],'results':data['results']})
        with np.load(directory/'forecasting'/'trajectory_predictions.npz') as a: targets={k:a[k] for k in a.files if k.endswith('__exact_voltage_V') or k.endswith('__origin_time_s')}
        if reference is None: reference=targets
        elif reference.keys()!=targets.keys() or any(not np.array_equal(reference[k],targets[k]) for k in reference): raise ValueError('Forecast origins/targets differ between campaigns')
    if len(datasets)!=1: raise ValueError('Campaigns use different datasets')
    destination=BASE/'output'/'case_study_comparisons'/args.name; destination.mkdir(parents=True,exist_ok=True)
    save_json(destination/'metrics.json',{'campaigns':rows,'matched_forecast_origins_and_targets':True,'dataset_sha256':next(iter(datasets)),'interpretation':'Checkpoint comparison; differences across epochs/seeds do not establish a causal projection benefit.'})
    lines=['# Battery checkpoint comparison','','Forecast origins and targets match exactly. Dataset hashes agree. Different step limits/selected epochs imply different optimization exposure; these rows are not a controlled learning-curve experiment.','', '| Checkpoint | Epochs executed | Selected epoch | Steps/epoch limit | Protocol | Native voltage RMSE (mV) | 30 min EKF scenario forecast RMSE (mV) | SOC violation (%) |','|---|---:|---:|---|---|---:|---:|---:|']
    for run in rows:
        for reliability in run['results']['reliability']['protocols']:
            forecast=next(r for r in run['results']['forecasting']['records'] if r['protocol']==reliability['protocol'] and r['method']=='daehn_ekf_known_current' and r['horizon_seconds']==1800)
            lines.append(f"| {Path(run['campaign']).name} | {run['epochs']} | {run['selected_epoch']} | {run['steps_per_epoch_limit'] or 'full epoch'} | {reliability['protocol']} | {reliability['rmse_V']*1000:.3f} | {forecast['rmse_V']*1000:.3f} | {100*reliability['soc_violation_fraction']:.4f} |")
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8'); print(destination/'report.md')
if __name__=='__main__': main()
