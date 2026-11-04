from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import solve_ivp,cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CASE=Path(__file__).resolve().parent
sys.path.insert(0,str(CASE))
import inverse
import jax.numpy as jnp

def main():
    run,config,parameters=inverse.load_saved('daehn',42)
    with np.load(run/'evaluation_data.npz') as a: old=a['coordinates'][:,0]; observed=a['targets']
    with np.load(CASE/'data/data.npz') as a: reference=a['parameters'][0,:4]
    t=np.linspace(old[0],old[-1],2001)
    def rhs(_,y):
        a,b,g,d=reference; return [a*y[0]-b*y[0]*y[1],-g*y[1]+d*y[0]*y[1]]
    sol=solve_ivp(rhs,(old[0],old[-1]),observed[0],t_eval=t,method='DOP853',rtol=1e-11,atol=1e-13)
    if not sol.success: raise RuntimeError(sol.message)
    payload=inverse.predict_payload(parameters,t[:,None],sol.y.T,config)
    coeff=np.asarray(parameters['coefficients']); y=payload['prediction']; exact=sol.y.T
    dt=old-old[0]; ix=cumulative_trapezoid(observed[:,0],old,initial=0); iy=cumulative_trapezoid(observed[:,1],old,initial=0)
    designs=[np.column_stack((dt,-iy)),np.column_stack((-dt,ix))]; response=[np.log(observed[:,0]/observed[0,0]),np.log(observed[:,1]/observed[0,1])]
    ident=[]
    for a,b in zip(designs,response):
        sv=np.linalg.svd(a,compute_uv=False); fit=np.linalg.lstsq(a,b,rcond=None)[0]; normalized=a/np.linalg.norm(a,axis=0)
        ident.append({'rank':int(np.linalg.matrix_rank(a)),'singular_values':sv.tolist(),'condition':float(sv[0]/sv[-1]),'column_normalized_condition':float(np.linalg.cond(normalized)),'observed_integral_fit':fit.tolist(),'integral_residual_rmse':float(np.sqrt(np.mean((a@fit-b)**2)))})
    a,b,g,d=reference; positive=(y>0).all(1); invariant=np.full(len(y),np.nan); invariant[positive]=d*y[positive,0]-g*np.log(y[positive,0])+b*y[positive,1]-a*np.log(y[positive,1]); h0=d*exact[0,0]-g*np.log(exact[0,0])+b*exact[0,1]-a*np.log(exact[0,1])
    metrics={'estimated_parameters':dict(zip(inverse.NAMES,map(float,coeff))),'generating_parameters':dict(zip(inverse.NAMES,map(float,reference))),'parameter_relative_error':(abs(coeff-reference)/abs(reference)).tolist(),'dense_state_rmse':float(np.sqrt(np.mean((y-exact)**2))),'continuous_residual_rmse':float(np.sqrt(np.mean(payload['differential_residual']**2))),'positivity_violation_fraction':float(np.mean(~positive)),'generating_invariant_max_error':float(np.nanmax(abs(invariant-h0))),'identifiability':ident,'available_saved_baselines':[],'limitations':['Only saved DAEHN inverse checkpoint exists; PINN inverse baseline unavailable.','All CSV observations used in fitting: reconstruction, not holdout generalization.','Integral-stabilized variant; not a convergence test of the notebook-only loss.','Design rank describes structural identifiability on observed populations; not a statistical confidence interval.','Generating coefficients read only in analysis.','Dense reference independently integrated from first observed state.']}
    dest=CASE.parent/'results'/CASE.name/'output'/'physical_analysis'/'inverse'; dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'trajectory_predictions.npz',**payload); (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].plot(t,exact); ax[0].plot(t,y,'--'); ax[0].set_xlabel('Time'); ax[0].set_ylabel('Populations'); ax[1].plot(t,np.linalg.norm(y-exact,axis=1)); ax[1].set_ylabel('Reconstruction error'); fig.tight_layout(); fig.savefig(dest/'comparison.png',dpi=150)
    (dest/'report.md').write_text('# Inverse Lotka–Volterra analysis\n\nMaximum relative coefficient error: '+str(float(np.max(abs(coeff-reference)/abs(reference))))+'\n\nDense state RMSE: '+str(metrics['dense_state_rmse'])+'\n\n'+ '\n\n'.join(metrics['limitations'])+'\n\nObserved integral design diagnostics: '+json.dumps(ident))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
