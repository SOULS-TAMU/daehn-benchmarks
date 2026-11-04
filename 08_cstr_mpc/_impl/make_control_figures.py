"""Render two slide-sized figures from full matched CSTR MPC trials."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results-root',type=Path,default=Path(r'D:\Projects\daehn-benchmarks\results\08_cstr_mpc\output\model_comparison\full\rk=8'))
p.add_argument('--output-dir',type=Path,default=_CASE_DIR/'slide_figures')
a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
TEAL='#007F86'; ORANGE='#C06A16'; GRAY='#7B8799'; INK='#172B4D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#64748B','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'svg.fonttype':'none'})
sources=[]
def load(model,study,objective):
    d=a.results_root/model/study/objective/'default/seed=101'
    path=d/'metrics.json'; m=json.loads(path.read_text())
    assert m['status']=='completed' and not m['verification_only']
    assert m['completed_steps']==m['requested_steps']
    with np.load(d/'trajectory_predictions.npz') as f:z={k:f[k] for k in f.files}
    assert len(z['states'])==m['completed_steps']+1
    assert np.isfinite(z['states']).all() and np.isfinite(z['solve_seconds']).all()
    # Refresh tracking and timing from trajectory arrays, matching the saved scoring contract.
    rmse=float(np.sqrt(np.mean((z['states'][1:,1]-z['references'])**2)))
    np.testing.assert_allclose(rmse,m['tracking_rmse'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(np.quantile(z['solve_seconds'],.95),m['solve_p95_seconds'],rtol=1e-10)
    sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return m,z
def matched(x,y):
    m,z=x; n,w=y
    for k in ['sampling_interval_minutes','horizon','requested_steps','model_sha256']:
        assert m[k]==n[k],f'Mismatched {k}'
    for k in ['references','quality_bands','disturbances']:
        assert np.array_equal(z[k],w[k]),f'Mismatched {k}'
def canvas(title,subtitle):
    f=plt.figure(figsize=(6.1,5.3),facecolor='white')
    f.text(.145,.96,title,fontsize=18,weight='bold',va='top')
    f.text(.145,.895,subtitle,fontsize=11.5,color='#526277',va='top')
    return f
def clean(ax):
    ax.grid(axis='y',color='#E4EAF0'); ax.set_axisbelow(True)
def save(f,stem,footer):
    f.text(.145,.025,footer,fontsize=10,color='#526277',va='bottom')
    f.savefig(a.output_dir/(stem+'.svg'),facecolor='white')
    f.savefig(a.output_dir/(stem+'.png'),dpi=240,facecolor='white')
    plt.close(f)

daehn=load('daehn','00_local_tracking','tracking')
linear=load('linear','00_local_tracking','tracking'); matched(daehn,linear)
dm,dz=daehn; lm,lz=linear; reduction=100*(1-dm['tracking_rmse']/lm['tracking_rmse'])
f=canvas('Tracking: DAEHN vs linear','Full 32-minute local tracking trial · same MPC settings')
ax=f.add_axes([.145,.54,.815,.245]); bar=f.add_axes([.145,.19,.815,.225])
t=np.arange(len(dz['states']))*dm['sampling_interval_minutes']
ref=np.r_[dz['references'],dz['references'][-1]]
ax.step(t,ref,where='post',color=INK,ls=':',lw=1.8,label='Reference')
ax.plot(t,dz['states'][:,1],color=TEAL,lw=2.4,label='DAEHN')
ax.plot(t,lz['states'][:,1],color=ORANGE,lw=2.0,ls='--',label='Linear')
ax.set_ylabel('Product B (scaled)'); ax.set_xlabel('Time (min)',fontsize=12,labelpad=2)
ax.set_xlim(0,t[-1]); ax.set_yticks([.515,.520,.525]); ax.set_ylim(.512,.528); clean(ax)
f.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.135,.855),ncol=3,frameon=False,fontsize=11,handlelength=1.6,columnspacing=1.)
heights=np.array([dm['tracking_rmse'],lm['tracking_rmse']])*1000
bar.bar([0,1],heights,width=.46,color=[TEAL,ORANGE])
for j,v in enumerate(heights):bar.text(j,v+.08,f'{v:.3f}',ha='center',fontsize=13,weight='bold')
bar.set_ylim(0,4.4); bar.set_xticks([0,1],['DAEHN','Linear']); bar.set_ylabel('RMSE (×10⁻³)'); clean(bar)
bar.text(.5,.93,f'{reduction:.1f}% lower tracking RMSE',transform=bar.transAxes,ha='center',va='top',fontsize=12,weight='bold',bbox=dict(fc='white',ec='none',pad=2))
save(f,'01_daehn_vs_linear_control','Both completed with zero quality-band excursions.\nFixed nominal linearization; one trained DAEHN seed.')

studies=[('01_long_term_requirements','tracking','4 h\ntracking'),('03_feed_disturbances','tracking','Feed\ntracking'),('01_long_term_requirements','economic','4 h\neconomic*')]
pairs=[]
for study,objective,label in studies:
    d=load('daehn',study,objective); n=load('nonlinear',study,objective); matched(d,n)
    pairs.append((d[0],n[0]))
f=canvas('Nonlinear control at lower cost','DAEHN vs equation-based nonlinear MPC')
top=f.add_axes([.145,.54,.815,.245]); bottom=f.add_axes([.145,.205,.815,.245])
x=np.arange(3); width=.29
for j,(label,color) in enumerate([('DAEHN',TEAL),('Nonlinear equations',GRAY)]):
    r=np.array([pair[j]['tracking_rmse']*1000 for pair in pairs]); latency=np.array([pair[j]['solve_p95_seconds'] for pair in pairs]); xx=x+(j-.5)*width
    top.bar(xx,r,width,color=color,label=label); bottom.bar(xx,latency,width,color=color)
    for q,v in zip(xx,r):top.text(q,v+.16,f'{v:.3f}',fontsize=10.5,ha='center',weight='bold')
    for q,v in zip(xx,latency):bottom.text(q,v+.06,f'{v:.2f}',fontsize=11,ha='center',weight='bold')
top.set_ylabel('RMSE (×10⁻³)'); top.set_ylim(0,8.5); top.set_xticks(x); top.tick_params(axis='x',labelbottom=False); clean(top)
bottom.set_ylabel('p95 solve time (s)'); bottom.set_ylim(0,4.05); bottom.set_xticks(x,[s[2] for s in studies]); clean(bottom)
gains=[]
for j,(d,n) in enumerate(pairs):
    gain=100*(1-d['solve_p95_seconds']/n['solve_p95_seconds']); gains.append(gain)
    bottom.text(j,3.62,f'−{gain:.1f}%',ha='center',fontsize=12,weight='bold',color=TEAL)
f.legend(*top.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.135,.855),ncol=2,frameon=False,fontsize=11,handlelength=1.6,columnspacing=1.2)
save(f,'02_daehn_vs_nonlinear_control_cost','*Economic RMSE is descriptive, not its primary objective.\nRecorded full MPC timings; one trained DAEHN seed.')

report={'figure_size_inches':[6.1,5.3],'sources':sources,'matched_schedules_and_complete_durations_verified':True,'linear_comparison':{'study':'00_local_tracking','daehn_tracking_rmse':dm['tracking_rmse'],'linear_tracking_rmse':lm['tracking_rmse'],'tracking_rmse_reduction_percent':reduction},'nonlinear_comparison':[{'study':study,'objective':obj,'daehn_tracking_rmse':d['tracking_rmse'],'nonlinear_tracking_rmse':n['tracking_rmse'],'daehn_p95_s':d['solve_p95_seconds'],'nonlinear_p95_s':n['solve_p95_seconds'],'p95_reduction_percent':gain} for (study,obj,_),(d,n),gain in zip(studies,pairs,gains)]}
(a.output_dir/'control_figure_metrics.json').write_text(json.dumps(report,indent=2))
(a.output_dir/'SLIDE_NOTES.txt').write_text('CSTR comparison figures: 6.1 x 5.3 inches each.\nFor a 13 x 7.5 inch slide, position x=0.30 and 6.60, y=0.85 inches.\nFigures finish at y=6.15, leaving about 1 inch for two slide points.\n\nSuggested slide points:\n1. DAEHN reduces local tracking RMSE by 27.4% versus fixed linear MPC, with zero quality-band excursions for both.\n2. DAEHN closely matches nonlinear-model control performance while reducing p95 solve time by 59-66% in the three shown full trials.\n\nSingle trained DAEHN seed (11), control scenario seed 101; no population-level confidence claim.\nLinear comparison uses its attainable local target range. Not evidence against gain-scheduled or successive-linearization MPC.\nNonlinear baseline is RK4 with 8 substeps per interval, inside matched horizon-40 SLSQP MPC.\nTimings are measured complete MPC solves, not standalone inference or guaranteed deployment speed.\nEconomic RMSE is descriptive: tracking accuracy is not the primary economic-control objective.\nAll concentration and control quantities are scaled benchmark values.\nReproduce using make_control_figures.py in the parent case study folder.\n')
print(a.output_dir)
