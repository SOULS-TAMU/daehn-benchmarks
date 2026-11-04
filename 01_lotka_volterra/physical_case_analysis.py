"""Read-only saved-model physical audit; outputs go to separate analysis branch."""
from pathlib import Path
import sys, json, time, hashlib, argparse
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CASE=Path(__file__).resolve().parent
sys.path.insert(0,str(CASE.parent/'_shared'))
from postprocessing.case_registry import build_inference_trainer
from case_study_utils import load_parameter_checkpoint, evaluate_trainer_predictions
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)

def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))
def maximum(x): return float(np.max(np.abs(x)))
def physical(y,ex):
    y=np.array(y,copy=True)
    if ex==17: y[:,0]=np.exp(np.clip(y[:,0],-60,5))
    return y
def phase(y): return np.unwrap(np.arctan2(y[:,1],y[:,0]))
def features(t,y,z,p,ex,module,deriv):
    m={'component_rmse':np.sqrt(np.mean((y-z)**2,axis=0)).tolist(), 'component_relative_l2':(np.linalg.norm(y-z,axis=0)/np.maximum(np.linalg.norm(z,axis=0),1e-30)).tolist(), 'minimum_components':y.min(axis=0).tolist(),'maximum_components':y.max(axis=0).tolist(),'final_state_error':float(np.linalg.norm(y[-1]-z[-1]))}
    peaks=find_peaks(y[:,0],prominence=max(np.ptp(z[:,0])*0.05,1e-10))[0]; ref=find_peaks(z[:,0],prominence=max(np.ptp(z[:,0])*0.05,1e-10))[0]
    m.update(peak_times=t[peaks].tolist(),reference_peak_times=t[ref].tolist(),period=float(np.mean(np.diff(t[peaks]))) if len(peaks)>1 else None,reference_period=float(np.mean(np.diff(t[ref]))) if len(ref)>1 else None,peak_timing_error=float(t[peaks[0]]-t[ref[0]]) if len(peaks) and len(ref) else None,extrema_error=np.maximum(abs(y.max(0)-z.max(0)),abs(y.min(0)-z.min(0))).tolist())
    if ex==12:
        m.update(energy_max_error=maximum(.5*np.sum(y*y,axis=1)-p[:,0]),amplitude_rmse=rms(np.linalg.norm(y,axis=1)-np.linalg.norm(z,axis=1)),phase_rmse=rms(phase(y)-phase(z)),frequency_error=float((phase(y)[-1]-phase(y)[0]-phase(z)[-1]+phase(z)[0])/(t[-1]-t[0])))
    if ex==13:
        r=np.linalg.norm(y[:,:2],axis=1); rz=np.linalg.norm(z[:,:2],axis=1)
        energy=.5*np.sum(y[:,2:]**2,axis=1)-p[:,0]/r; ez=.5*np.sum(z[:,2:]**2,axis=1)-p[:,0]/rz
        m.update(angular_momentum_max_error=maximum(y[:,0]*y[:,3]-y[:,1]*y[:,2]-p[:,1]),independent_energy_max_error=maximum(energy-ez),energy_drift=maximum(energy-energy[0]),radial_rmse=rms(r-rz),orbital_phase_rmse=rms(phase(y)-phase(z)),endpoint_closure=float(np.linalg.norm(y[-1]-y[0])),reference_endpoint_closure=float(np.linalg.norm(z[-1]-z[0])))
    if ex==11:
        positive=np.all(y>0,axis=1); inv=np.full(len(y),np.nan); a,b,g,d,h=p.T
        inv[positive]=d[positive]*y[positive,0]-g[positive]*np.log(y[positive,0])+b[positive]*y[positive,1]-a[positive]*np.log(y[positive,1])-h[positive]
        m.update(nonpositive_fraction=float(np.mean(~positive)),invariant_max_error=float(np.nanmax(abs(inv))))
        center=np.array([p[0,2]/p[0,3],p[0,0]/p[0,1]])
        m['phase_rmse']=rms(phase(y-center)-phase(z-center))
    if ex==15:
        jac=np.array([module.end_effector_jacobian(v) for v in y]); sv=np.linalg.svd(jac,compute_uv=False)
        ee=np.array([module.end_effector(v) for v in y]); exact=np.array([module.end_effector(v) for v in z]); velocity=np.array([module.target_velocity(v) for v in t])
        m.update(end_effector_path_rmse=rms(ee-exact),joint_velocity_rmse=rms(deriv-np.gradient(z,t,axis=0)),path_velocity_consistency_rmse=rms(np.einsum('nij,nj->ni',jac,deriv)-velocity),minimum_jacobian_singular_value=float(sv[:,-1].min()),maximum_jacobian_condition=float(np.max(sv[:,0]/sv[:,-1])))
    if ex==16:
        inertia=p[:,:3]; energy=.5*np.sum(inertia*y*y,1); momentum=np.sum((inertia*y)**2,1)
        jac=np.stack((inertia*y,2*inertia**2*y),axis=1); sv=np.linalg.svd(jac,compute_uv=False)
        m.update(energy_max_error=maximum(energy-p[:,3]),momentum_squared_max_error=maximum(momentum-p[:,4]),precession_phase_rmse=rms(phase(y)-phase(z)),minimum_constraint_singular_value=float(sv[:,-1].min()),maximum_constraint_condition=float(np.max(sv[:,0]/sv[:,-1])))
    if ex==8:
        m.update(negative_concentration_fraction=float(np.mean(y[:,:3]<0)),temperature_rmse=rms(y[:,3]-z[:,3]),terminal_window_rmse=rms(y[-max(2,len(y)//10):]-z[-max(2,len(y)//10):]),terminal_velocity_norm=float(np.linalg.norm(deriv[-1])))
    if ex==17:
        good=(y>0)&(z>0)
        m.update(nonpositive_fraction=float(np.mean(y<=0)),log_concentration_rmse=rms(np.log(y[good])-np.log(z[good])),relative_concentration_rmse_by_component=np.sqrt(np.mean(((y-z)/np.maximum(abs(z),1e-30))**2,0)).tolist(),conversion_error=abs(float(1-y[-1,0]/y[0,0]-(1-z[-1,0]/z[0,0]))))
    return m

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int); parser.add_argument('--study'); parser.add_argument('--points',type=int,default=801); args=parser.parse_args()
    base=CASE.parent/'results'/CASE.name; study=args.study or ('parameter_pairs' if CASE.name.startswith('07') else 'reference'); out=base/'output'/'physical_analysis'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for manifest_path in sorted((base/'model'/study).rglob('checkpoint_manifest.json')):
        run=manifest_path.parent; manifest=json.loads(manifest_path.read_text()); seed=manifest['seed']
        if args.seed and seed!=args.seed: continue
        dest=out/study/manifest['model']/run.name; dest.mkdir(parents=True,exist_ok=True)
        cached=None
        if (dest/'metrics.json').exists():
            saved=json.loads((dest/'metrics.json').read_text())
            if saved['grid_points']==args.points:
                if saved.get('analysis_version')==4 and (int(manifest['example'])!=17 or 'concentration_decode_clip_fraction' in saved['variants'][0]): rows.extend(saved['variants']); continue
                cached=saved
        print('START',CASE.name,manifest['model'],seed,run.name,flush=True)
        if hashlib.sha256((run/manifest['checkpoint_file']).read_bytes()).hexdigest()!=manifest['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch: '+str(run))
        trainer,projected,manifest,config,data,module=build_inference_trainer(run); params=load_parameter_checkpoint(run); ex=int(manifest['example'])
        with np.load(run/'trajectory_data.npz') as arch: original={k:arch[k] for k in arch.files}
        old=original['coordinates'][:,0]; order=np.argsort(old); old=old[order]; states=original['targets'][order]; ders=original['derivative_targets'][order]
        system=getattr(trainer,'physical_system',trainer.compiled.system)
        state_names=[s.name for s in system.differential_states]
        spec_names=[s.state.name for s in trainer.compiled.discretization.derivatives]
        derivative_order=[state_names.index(n) for n in spec_names]
        state_ders=ders[:,np.argsort(derivative_order)]
        t=np.linspace(old[0],old[-1],args.points); coords=t[:,None]; p=np.column_stack([np.interp(t,old,original['physical_parameters'][order,i]) for i in range(original['physical_parameters'].shape[1])])
        if ex==8:
            phi,q=module.input_profile(t); ca,th=module.disturbance_profile(t); p=np.column_stack((phi,q,ca,th))
        # Hermite interpolation uses supplied exact derivatives for differential states.
        exact=np.column_stack([CubicHermiteSpline(old,states[:,i],state_ders[:,i])(t) if i<ders.shape[1] else np.interp(t,old,states[:,i]) for i in range(states.shape[1])]); ed=np.column_stack([CubicHermiteSpline(old,states[:,i],state_ders[:,i]).derivative()(t) for i in range(ders.shape[1])])
        interpolated=exact.copy()
        nd=ders.shape[1]
        system=getattr(trainer,'physical_system',trainer.compiled.system)
        def parameter_at(q):
            if ex==8: return np.array([*module.input_profile(q),*module.disturbance_profile(q)])
            return np.array([np.interp(q,old,original['physical_parameters'][order,i]) for i in range(p.shape[1])])
        def expand(s,q):
            if ex==8: return np.concatenate((s,np.asarray(module.reaction_rates(s))))
            if ex==17:
                k=parameter_at(q)[1]; c=np.exp(np.clip(s[0],-60,5)); return np.array([s[0],k*c/(1+k*c),1/(1+k*c)])
            return s
        residual=jax.jit(system.differential_equations.function)
        def rhs(q,s):
            par=parameter_at(q)
            if ex==17: return np.array([-par[0]*par[1]/(1+par[1]*np.exp(np.clip(s[0],-60,5)))])
            return -np.asarray(residual(jnp.array([q]),jnp.asarray(expand(s,q)),jnp.zeros(nd),jnp.asarray(par)))
        solution=solve_ivp(rhs,(old[0],old[-1]),states[0,:nd],t_eval=t,method='DOP853',rtol=1e-10,atol=1e-12)
        if not solution.success: raise RuntimeError(solution.message)
        exact=np.array([expand(s,q) for q,s in zip(t,solution.y.T)]); ed=np.array([rhs(q,s) for q,s in zip(t,solution.y.T)])[:,derivative_order]
        reference_discrepancy=maximum(exact-interpolated)
        variants=[]; archives={'coordinates':coords,'physical_parameters':p,'exact':exact,'exact_physical':physical(exact,ex),'exact_derivatives':ed}
        modes=[('daehn',True)] if projected else [(manifest['model'],False),(manifest['model']+'_projected',True)]
        for name,use_projection in modes:
            start=time.perf_counter()
            if cached:
                with np.load(dest/'trajectory_predictions.npz') as archive: payload={k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith(name+'__')}
                metrics=next(v.copy() for v in cached['variants'] if v['variant']==name)
                metrics['mse_data']=float(np.mean((payload['prediction']-exact)**2))
                metrics['relative_state_error']=float(np.linalg.norm(payload['prediction']-exact)/np.linalg.norm(exact))
                metrics['mse_derivative']=float(np.mean((payload['model_derivatives']-ed)**2))
            else:
                payload,metrics=evaluate_trainer_predictions(trainer,params,coordinates=coords,physical_parameters=p,targets=exact,derivative_targets=ed,projected=use_projection,model_name='daehn' if use_projection else name)
            y=payload['prediction']; d=payload['model_derivatives']; yp=physical(y,ex); zp=physical(exact,ex)
            archives[name+'__physical_prediction']=yp
            metrics['mse_derivative']=float(np.mean((d-ed)**2))
            if ex==17:
                metrics['concentration_decode_clip_fraction']=float(np.mean((y[:,0]<-60)|(y[:,0]>5)))
                metrics['internal_log_y1_rmse']=rms(y[:,0]-exact[:,0])
                metrics['unclipped_y1_rmse']=rms(np.exp(np.clip(y[:,0],-350,350))-np.exp(np.clip(exact[:,0],-350,350)))
            metrics.update(features(t,yp,zp,p,ex,module,d),variant=name,seed=seed,run=str(run),checkpoint_sha256=manifest['checkpoint_sha256'],evaluation_wall_seconds=time.perf_counter()-start,algebraic_component_max=np.max(abs(payload['algebraic_residual']),axis=0).tolist(),differential_component_rmse=np.sqrt(np.mean(payload['differential_residual']**2,0)).tolist(),state_rmse=rms(yp-zp),full_map_derivative_rmse=rms(d-ed),differentiated_constraint_rmse=rms(np.gradient(payload['algebraic_residual'],t,axis=0)) if payload['algebraic_residual'].shape[1] else None)
            if 'converged' in payload: metrics['projection_failure_fraction']=float(np.mean(~payload['converged']))
            if 'projected_derivatives' in payload: metrics['auxiliary_vs_full_map_derivative_rmse']=rms(payload['projected_derivatives']-d)
            jf=jax.jit(jax.vmap(jax.jacfwd(system.algebraic_equalities.function,argnums=1)))
            jac=np.asarray(jf(jnp.asarray(coords),jnp.asarray(y),jnp.asarray(d),jnp.asarray(p)))
            if jac.shape[1]:
                sv=np.linalg.svd(jac,compute_uv=False); metrics['constraint_minimum_singular_value']=float(sv[:,-1].min()); metrics['constraint_maximum_condition']=float(np.max(sv[:,0]/np.maximum(sv[:,-1],1e-30)))
            grid_derivative=np.gradient(y[:,:nd],t,axis=0,edge_order=2)[:,derivative_order]
            grid_residual=np.asarray(jax.jit(jax.vmap(system.differential_equations.function))(jnp.asarray(coords),jnp.asarray(y),jnp.asarray(grid_derivative),jnp.asarray(p)))
            interior=np.ones(len(t),dtype=bool); interior[[0,-1]]=False
            if ex==8:
                for transition in (20,25,40): interior &= abs(t-transition)>2*np.max(np.diff(t))
            metrics['grid_total_differential_rmse_interior']=rms(grid_residual[interior])
            archives[name+'__grid_total_differential_residual']=grid_residual
            metrics['dense_between_saved_points_fraction']=float(np.mean(np.min(abs(t[:,None]-old[None,:]),axis=1)>1e-10))
            variants.append(metrics); rows.append(metrics)
            archives.update({name+'__'+k:v for k,v in payload.items() if k not in ('coordinates','physical_parameters','exact','exact_derivatives')})
        np.savez_compressed(dest/'trajectory_predictions.npz',**archives)
        limitations=['Time-only backbone: no novel initial-condition or recursive-flow claims.','In-window dense grid; no extrapolation.','Independent DOP853 reference from first saved trajectory state, rtol=1e-10, atol=1e-12; varying physical inputs interpolated from saved arrays.','Differentiated constraint residual uses grid finite differences, including endpoints.','Projection uses each saved trainer configuration; baseline comparison is paired to its own configuration.']
        (dest/'metrics.json').write_text(json.dumps({'analysis_version':4,'derivative_order':spec_names,'contract':manifest['backbone'],'grid_points':len(t),'trajectory_data_sha256':manifest.get('trajectory_data_sha256'),'reference_vs_saved_interpolation_max':reference_discrepancy,'time_window':[float(t[0]),float(t[-1])],'limitations':limitations,'variants':variants},indent=2,allow_nan=True))
        fig,axes=plt.subplots(2,1,figsize=(9,7)); axes[0].plot(t,zp[:,0],label='reference',color='black')
        for v in variants:
            name=v['variant']; pred=physical(archives[name+'__prediction'],ex); axes[0].plot(t,pred[:,0],label=name); axes[1].plot(t,np.linalg.norm(pred-zp,axis=1),label=name)
        if ex==17: axes[0].set_yscale('log')
        axes[0].set_ylabel('State component 1'); axes[1].set_ylabel('State error norm'); axes[1].set_xlabel('Coordinate'); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(dest/'comparison.png',dpi=140); plt.close(fig)
        print('DONE',[(v['variant'],v['state_rmse']) for v in variants],flush=True)
    (out/'metrics.json').write_text(json.dumps({'runs':rows},indent=2,allow_nan=True))
    lines=['# '+CASE.name+' physical analysis','',f'{len(rows)} variant evaluations. Independent dense DOP853 reference; all models evaluated at identical points within each saved dataset.','', '| Variant | Count | Median state RMSE | Median differential violation | Median algebraic violation |','|---|---:|---:|---:|---:|']
    for name in sorted(set(v['variant'] for v in rows)):
        vals=[v for v in rows if v['variant']==name]; lines.append(f"| {name} | {len(vals)} | {np.median([v['state_rmse'] for v in vals]):.6g} | {np.median([v['differential_violation'] for v in vals]):.6g} | {np.median([v['algebraic_violation'] for v in vals]):.6g} |")
    lines+=['','Constraint accuracy and trajectory accuracy must be interpreted separately. Projection failure fractions, extrema, phase/period availability and case-specific physical diagnostics are in each run metrics file. A null period means too few detected peaks in the saved window.','', 'No retraining. No control campaign. No conditioned-flow or extrapolation tests. Comparisons across catalytic parameter regimes must use the per-run parameter arrays. The CSTR terminal window is a transient diagnostic, not proof of equilibrium. Rigid-body phase refers to the angular-velocity plane, not reconstructed spatial attitude.']
    (out/'report.md').write_text('\n'.join(lines))
if __name__=='__main__': main()
