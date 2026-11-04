"""Summarize matched trials, with failures retained and no unequal-duration gains."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import argparse,json
from pathlib import Path
import numpy as np
from compare_prediction_models import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save-plots',action='store_true');args=p.parse_args()
    if args.save_plots:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    groups={}
    for path in sorted(ROOT.glob('*/*/*/*/*/*/seed=*/metrics.json')):
        m=json.loads(path.read_text());group=(path.parents[6].name,path.parents[5].name,m['experiment'],m['controller'],m['variant'],m['seed'])
        if m['status']=='running':continue
        m['corrected_one_step_rmse']=None
        if (path.parent/'trajectory_predictions.npz').exists():
            with np.load(path.parent/'trajectory_predictions.npz') as a:
                m['corrected_one_step_rmse']=float(np.sqrt(np.mean((a['predicted_next_states']-a['states'][1:])**2)))
        (path.parent/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
        groups.setdefault(group,[]).append((path,m))
    lines=['# Matched prediction-model control comparisons','','Short verification runs check execution only; performance conclusions require full matched trials. Prediction error below uses corrected controller predictions and therefore includes delay effects in study 05.','', '| Run | Study / objective / variant / seed | Predictor | Status | Steps | Tracking RMSE | Production | Quality outside min | Heat effort | One-step RMSE | Solve p95 s | Fallbacks |','|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group,entries in groups.items():
        for path,m in entries:
            lines.append('| {} | {} / {} / {} / {} | {} | {} | {} | {:.6f} | {:.6f} | {:.3f} | {:.6f} | {:.6f} | {:.3f} | {} |'.format(group[0],m['experiment'],m['controller'],m['variant'],m['seed'],m['predictor'],m['status'],m['completed_steps'],m.get('tracking_rmse',float('nan')),m.get('mean_production_proxy',float('nan')),m.get('quality_outside_sampled_minutes',float('nan')),m.get('mean_heat_effort',float('nan')),m['corrected_one_step_rmse'] if m['corrected_one_step_rmse'] is not None else float('nan'),m.get('solve_p95_seconds',float('nan')),m.get('fallback_steps',0)))
        entries=[(path,m) for path,m in entries if (path.parent/'trajectory_predictions.npz').exists()]
        if args.save_plots and entries:
            fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
            for path,m in entries:
                with np.load(path.parent/'trajectory_predictions.npz') as a:
                    x=a['states'];u=a['controls'];t=np.arange(len(u))*.25;tx=np.arange(len(x))*.25
                    axes[0].plot(tx,x[:,1],label=m['predictor']+' / '+m['status'])
                    axes[1].plot(t,u[:,0]*x[1:,1]);axes[2].plot(t,(u[:,1]/.6)**2);axes[3].plot(t,a['solve_seconds'])
            longest=max(entries,key=lambda item:item[1]['completed_steps'])[0]
            with np.load(longest.parent/'trajectory_predictions.npz') as a:
                t=np.arange(len(a['controls']))*.25;band=a['quality_bands']
                axes[0].step(t,a['references'],where='post',color='black',ls='--',label='Tracking reference')
                axes[0].fill_between(t,band[:,0],band[:,1],color='green',alpha=.1,step='post')
            for ax,label in zip(axes,['Product B','Production proxy','Heat effort','Solve seconds']):ax.set_ylabel(label);ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);axes[0].set_title(' / '.join(map(str,group)));axes[-1].set_xlabel('Time (minutes)')
            output=ROOT/group[0]/group[1]/'comparisons'/group[2]/group[3]/group[4]/('seed='+str(group[5]));output.mkdir(parents=True,exist_ok=True)
            fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    (ROOT/'COMPARISON_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved comparison report for',len(groups),'matched groups:',ROOT/'COMPARISON_RESULTS.md')
if __name__=='__main__':main()
