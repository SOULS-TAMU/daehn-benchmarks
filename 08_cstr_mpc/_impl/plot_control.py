"""Plot existing controller results; no retraining or controller rerun."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse
import numpy as np

CASE=_CASE_DIR

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=['tracking','disturbance','nominal'],default='tracking')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--objective',choices=['tracking','economic'],default='tracking')
    p.add_argument('--no-offset-correction',action='store_true')
    p.add_argument('--compare-original',action='store_true')
    p.add_argument('--compare-previous',action='store_true',help='Compare with the previous offset-free controller')
    p.add_argument('--terminal-cost',choices=['lqr','legacy'],default='legacy')
    p.add_argument('--save',action='store_true')
    p.add_argument('--show',action='store_true')
    args=p.parse_args()
    import matplotlib
    if not args.show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=CASE.parent/'results'/CASE.name/'output/control'/f'seed={args.seed}'
    folder=base/(args.scenario+('_economic' if args.objective=='economic' else '' if args.no_offset_correction else '_offset_free_lqr' if args.terminal_cost=='lqr' else '_offset_free'))
    with np.load(folder/'trajectory_predictions.npz') as a:
        t=a['time'];x=a['states'];ref=a['references'];baseline=a['baseline_states']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    ax.plot(t,x[:,1],color='#1261a0',linewidth=2.2,label='Economic NMPC' if args.objective=='economic' else 'Original MPC' if args.no_offset_correction else 'Offset-free MPC + LQR' if args.terminal_cost=='lqr' else 'Previous offset-free MPC')
    if args.compare_previous and not args.no_offset_correction:
        with np.load(base/(args.scenario+'_offset_free')/'trajectory_predictions.npz') as old:
            ax.plot(old['time'],old['states'][:,1],color='#b36b24',linestyle=':',linewidth=1.7,label='Previous offset-free MPC')
    elif args.compare_original and not args.no_offset_correction:
        with np.load(base/args.scenario/'trajectory_predictions.npz') as old:
            ax.plot(old['time'],old['states'][:,1],color='#b36b24',linewidth=1.7,label='Original MPC')
    else:ax.plot(t,baseline[:,1],color='#b36b24',linestyle=':',label='Fixed nominal control')
    if args.objective=='economic':
        import json
        band=json.loads((folder/'metrics.json').read_text())['economic_config']['quality_band']
        ax.axhspan(*band,alpha=.12,color='green',label='B quality band')
    ax.step(t,np.r_[ref,ref[-1]],where='post',color='#222222',linestyle='--',linewidth=1.7,label='Economic operating target' if args.objective=='economic' else 'Setpoint')
    ax.set(xlabel='Time (minutes)',ylabel='Product-B concentration (benchmark units)',title='CSTR economic production control' if args.objective=='economic' else 'CSTR product-B '+('disturbance rejection' if args.scenario=='disturbance' else 'setpoint tracking'),xlim=(t[0],t[-1]))
    ax.grid(alpha=.2);ax.legend(frameon=False)
    if args.save:
        dest=folder/(args.scenario+'.png');fig.savefig(dest,dpi=220,facecolor='white');print(dest)
    if args.show:plt.show()
    plt.close(fig)

if __name__=='__main__':main()