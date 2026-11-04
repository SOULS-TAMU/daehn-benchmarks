"""Read completed or partial control studies and optionally save comparison plots."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json
import numpy as np
from control_experiments import ROOT,EXPERIMENTS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['all']+list(EXPERIMENTS),default='all')
    p.add_argument('--save-plots',action='store_true');p.add_argument('--show',action='store_true')
    args=p.parse_args()
    import matplotlib
    if not args.show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in list(EXPERIMENTS) if args.experiment=='all' else [args.experiment]:
        folder=ROOT/name;entries=[]
        for path in sorted(folder.glob('*/*/seed=*/metrics.json')):
            m=json.loads(path.read_text());entries.append((path,m))
        if not entries:continue
        print(name,flush=True)
        for path,m in entries:
            print(m['controller'],m['variant'],m['seed'],m['status'],m['completed_steps'],
                  'production',round(m.get('mean_production_proxy',0),5),'quality minutes',round(m.get('quality_outside_sampled_minutes',0),3),
                  'physical violation',m.get('maximum_physical_state_violation'), 'fallbacks',m.get('fallback_steps'),flush=True)
        for variant in sorted({m['variant'] for _,m in entries}):
            fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
            plotted=False
            available=[(path,m) for path,m in entries if m['variant']==variant and (path.parent/'trajectory_predictions.npz').exists()]
            if available:
                schedule_path,schedule_m=max(available,key=lambda item:item[1]['completed_steps'])
                with np.load(schedule_path.parent/'trajectory_predictions.npz') as schedule:
                    st=np.arange(len(schedule['controls']))*schedule_m['sampling_interval_minutes']
                    axes[0].step(st,schedule['references'],where='post',color='black',ls='--',lw=1,label='Tracking reference')
                    axes[0].fill_between(st,schedule['quality_bands'][:,0],schedule['quality_bands'][:,1],step='post',color='green',alpha=.1,label='Quality band')
            for path,m in entries:
                if m['variant']!=variant or not (path.parent/'trajectory_predictions.npz').exists():continue
                with np.load(path.parent/'trajectory_predictions.npz') as a:
                    x=a['states'];u=a['controls'];r=a['references'];band=a['quality_bands'];dt=m['sampling_interval_minutes']
                    t=np.arange(len(u))*dt;tx=np.arange(len(x))*dt
                    label=m['controller']+(f" seed {m['seed']}" if name.startswith('05') else '')
                    axes[0].plot(tx,x[:,1],label=label,alpha=.8)
                    axes[1].plot(t,u[:,0]*x[1:,1],label=label,alpha=.8)
                    axes[2].plot(t,(u[:,1]/.6)**2,label=label,alpha=.8)
                    if m['status']=='physical_constraint_stop':
                        axes[0].plot(tx[-1],x[-1,1],'rx',ms=8,label=label+' stopped: physical bound')
                    elif m['status']=='rejected_requirement':
                        axes[0].axvline(tx[-1],color='gray',ls=':',lw=.8)
                    plotted=True
            axes[0].set_title(name.replace('_',' ')+' / '+variant)
            axes[0].set_ylabel('Product-B concentration');axes[1].set_ylabel('Production proxy');axes[2].set_ylabel('Heat effort (q/0.6)^2');axes[2].set_xlabel('Time (minutes)')
            for ax in axes:ax.grid(alpha=.2)
            axes[0].legend(fontsize=8,ncol=2)
            if args.save_plots and plotted:fig.savefig(folder/(variant+'_comparison.png'),dpi=180,facecolor='white')
            if args.show:plt.show()
            plt.close(fig)

if __name__=='__main__':main()