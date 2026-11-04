"""Receding-horizon product-B controller using a saved DAEHN flow model.

The controller receives measured state/disturbances; reference equations are
used only by the separate simulation harness. No model retraining is performed.
"""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse, json, time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import solve_discrete_are
from surrogate import SavedFlowModel
from model import integrate_interval, admissible_state, state_constraint_values

CASE = _CASE_DIR
RESULTS = CASE.parent / 'results' / CASE.name
DEFAULT_MODEL = RESULTS / 'model' / 'flow' / 'daehn' / 'seed=11'

class MPCController:
    """Single shooting, full-state feedback, measured disturbance persistence.

    step returns the first move and diagnostics. Failed optimization can use a
    rechecked feasible shifted plan; otherwise it raises rather than silently
    applying an unverified action. This is simulation-grade failure handling.
    """
    def __init__(self, model_dir=DEFAULT_MODEL, horizon=None, max_iterations=100,
                 constraint_tolerance=1e-7, state_margin=0.0, offset_correction=True, bias_gain=0.3, terminal_cost="legacy", target_selection="legacy", objective_mode="tracking", quality_band=(0.48,0.50), production_weight=1.0, heat_weight=0.05, flow_weight=0.05, quality_penalty=1000.0, quality_margin=1e-4, minimum_production=0.0, shortfall_penalty=100.0, prediction_model=None):
        self.model = SavedFlowModel(model_dir) if prediction_model is None else prediction_model
        self.config = self.model.config['plant_config']
        c = self.config; m = c['mpc']
        # Build the cached predictor outside JAX tracing.
        self.model(np.r_[c['nominal_state'],c['nominal_control'],c['nominal_disturbance']])
        self.horizon = int(m['prediction_horizon'] if horizon is None else horizon)
        if self.horizon < 1 or max_iterations < 1 or not np.isfinite(state_margin) or state_margin < 0 or state_margin >= 2 or constraint_tolerance <= 0:
            raise ValueError('Invalid controller settings')
        if not np.isclose(self.model.config['sampling_interval'],c['sampling_interval']):
            raise ValueError('Model and controller sampling intervals disagree')
        self.max_iterations = max_iterations
        self.tolerance = constraint_tolerance
        if minimum_production<0 or shortfall_penalty<0:raise ValueError('Invalid production-demand settings')
        if objective_mode not in {'tracking','economic'}:raise ValueError('Unknown objective mode')
        if not 0<=quality_band[0]<quality_band[1]<=c['state_bounds'][1][1]:raise ValueError('Invalid quality band')
        if not 0<=quality_margin<(quality_band[1]-quality_band[0])/2:raise ValueError('Invalid quality margin')
        if min(production_weight,heat_weight,flow_weight,quality_penalty)<0 or quality_penalty<=0:raise ValueError('Invalid economic weights')
        if objective_mode=='economic' and (not offset_correction or terminal_cost!='legacy'):raise ValueError('Economic mode requires offset correction and the legacy nonlinear terminal cost')
        self.objective_mode=objective_mode
        self.economic={'quality_band':list(quality_band),'production_weight':production_weight,'heat_weight':heat_weight,'flow_weight':flow_weight,'quality_penalty':quality_penalty,'quality_margin':quality_margin,'minimum_production':minimum_production,'shortfall_penalty':shortfall_penalty}
        self.plan = None
        if not 0 < bias_gain <= 1: raise ValueError('bias_gain must be in (0,1]')
        self.offset_correction = bool(offset_correction)
        if terminal_cost not in {'lqr','legacy'} or target_selection not in {'continuous','legacy'}:
            raise ValueError('Unknown terminal cost or target selection')
        self.terminal_cost = terminal_cost if self.offset_correction else 'legacy'
        self.target_selection = target_selection
        self._terminal_info = {}
        self._target_switches = 0
        self.bias_gain = float(bias_gain)
        self.bias = np.zeros(4)
        self._previous_raw_prediction = None
        self._previous_context = None
        self._steady_target = None
        self._state_margin = state_margin
        n=self.horizon; ts=c['sampling_interval']
        # Check both midpoint and endpoint predictions without clipping states.
        bounds=jnp.asarray(c['state_bounds']); move=jnp.asarray(c['control_move_limits'])
        nominal=jnp.asarray(c['nominal_control'])
        def rollout(flat,x,d,bias):
            controls=flat.reshape((n,2))
            def body(state,values):
                u,disturbance=values
                inputs=jnp.concatenate((state,u,disturbance))
                middle=self.model.predict_jax(inputs,tau=ts/2)[:4]+0.5*bias
                following=self.model.predict_jax(inputs)[:4]+bias
                return following,(following,middle)
            _,(states,middle)=jax.lax.scan(body,x,(controls,d))
            return states,middle
        def objective(flat,x,previous,d,reference,bias,target_state,target_control,terminal_matrix):
            states,_=rollout(flat,x,d,bias); controls=flat.reshape((n,2))
            deltas=controls-jnp.concatenate((previous[None,:],controls[:-1]),axis=0)
            tracking=((states[:,1]-reference)/m['concentration_tracking_scale'])**2
            thermal=m['temperature_tracking_weight']*((states[:,3]-target_state[3])/m['temperature_tracking_scale'])**2
            deviation=jnp.sum(((controls-target_control)/jnp.asarray(m['input_deviation_scales']))**2,axis=1)
            changes=jnp.sum((deltas/move)**2,axis=1)
            regularization=1e-6*jnp.sum(((states[:,jnp.array([0,2])]-target_state[jnp.array([0,2])])/0.1)**2,axis=1) if self.terminal_cost=='lqr' else jnp.zeros(n)
            if self.objective_mode=='economic':
                lower,upper=self.economic['quality_band'];lower+=quality_margin;upper-=quality_margin
                violation=jnp.maximum(lower-states[:,1],0)+jnp.maximum(states[:,1]-upper,0)
                shortfall=jnp.maximum(minimum_production-controls[:,0]*states[:,1],0)
                stage=shortfall_penalty*shortfall**2-production_weight*controls[:,0]*states[:,1]+heat_weight*(controls[:,1]/0.6)**2+flow_weight*(controls[:,0]/2)**2+quality_penalty*(violation/0.1)**2
                return jnp.sum(stage+m['input_move_weight']*changes)+m['terminal_tracking_weight']*(tracking[-1]+thermal[-1])
            state_error=(states[-1]-target_state).at[1].set(states[-1,1]-reference[-1])
            end_error=jnp.concatenate((state_error,controls[-1]-target_control))
            terminal=end_error@terminal_matrix@end_error
            return jnp.sum(tracking+thermal+regularization+m['input_deviation_weight']*deviation+m['input_move_weight']*changes)+terminal
        def constraints(flat,x,d,bias):
            states,middle=rollout(flat,x,d,bias)
            values=jnp.stack((middle,states),axis=1)
            # Margin tightens thermal upper/lower limits, not concentration
            # positivity or the inventory invariant near boundary states.
            lo=bounds[:,0].at[3].add(state_margin)
            hi=bounds[:,1].at[3].add(-state_margin)
            return jnp.concatenate(((values-lo).reshape(-1),(hi-values).reshape(-1),(c['inventory_upper']-jnp.sum(values[...,:3],axis=-1)).reshape(-1)))
        def equilibrium(values,d,bias,reference):
            xs,us=values[:4],values[4:]
            predicted=self.model.predict_jax(jnp.concatenate((xs,us,d)))[:4]+bias
            return jnp.concatenate((predicted-xs,jnp.asarray([xs[1]-reference])))
        def target_cost(values,anchor):
            if self.objective_mode=='economic':
                return -production_weight*values[4]*values[1]+heat_weight*(values[5]/0.6)**2+flow_weight*(values[4]/2)**2
            return m['temperature_tracking_weight']*(values[3]/m['temperature_tracking_scale'])**2+m['input_deviation_weight']*jnp.sum(((values[4:]-nominal)/jnp.asarray(m['input_deviation_scales']))**2)
        def approach_cost(target,x,previous,d,bias,reference):
            def advance(carry,_):
                state,last=carry
                u=last+jnp.clip(target[4:]-last,-move,move)
                following=self.model.predict_jax(jnp.concatenate((state,u,d)))[:4]+bias
                cost=((following[1]-reference)/m['concentration_tracking_scale'])**2+m['input_move_weight']*jnp.sum(((u-last)/move)**2)
                violation=jnp.concatenate((bounds[:,0]-following,following-bounds[:,1],jnp.asarray([jnp.sum(following[:3])-c['inventory_upper']])))
                cost+=1e6*jnp.sum(jnp.maximum(violation,0)**2)
                return (following,u),cost
            _,costs=jax.lax.scan(advance,(x,previous),None,length=n)
            return jnp.sum(costs)
        self._approach_cost=jax.jit(approach_cost)
        self._flow_jacobian=jax.jit(jax.jacfwd(lambda value:self.model.predict_jax(value)[:4]))
        self._equilibrium=jax.jit(equilibrium)
        self._equilibrium_jacobian=jax.jit(jax.jacfwd(equilibrium))
        self._target_cost=jax.jit(jax.value_and_grad(target_cost))
        self._rollout=jax.jit(rollout)
        self._cost_gradient=jax.jit(jax.value_and_grad(objective))
        self._constraints=jax.jit(constraints)
        self._constraint_jacobian=jax.jit(jax.jacfwd(constraints))
        self._bounds=Bounds(np.tile(np.asarray(c['control_bounds'])[:,0],n),np.tile(np.asarray(c['control_bounds'])[:,1],n))
        self._difference=np.eye(2*n)
        self._difference[2:,:-2]-=np.eye(2*(n-1))

    def steady_target(self,disturbance,reference,anchor=None,previous=None):
        """Feasible surrogate equilibrium with exact B target, corrected for bias."""
        c=self.config
        if not c['state_bounds'][1][0] <= reference <= c['state_bounds'][1][1]:
            raise ValueError('Requested B setpoint is outside the operating bounds')
        anchor=np.asarray(c['nominal_state'] if anchor is None else anchor)
        initial=np.r_[c['nominal_state'],c['nominal_control']] if self._steady_target is None else self._steady_target.copy()
        b=np.vstack((np.asarray(c['state_bounds']),np.asarray(c['control_bounds'])))
        b[3,0]+=self._state_margin; b[3,1]-=self._state_margin
        if self.objective_mode=='economic': b[1]=np.asarray(self.economic['quality_band'])+np.array([self.economic['quality_margin'],-self.economic['quality_margin']])
        initial=np.clip(initial,b[:,0],b[:,1])
        dj,bj,rj=map(jnp.asarray,(disturbance,self.bias,reference))
        def objective(a):
            value,gradient=self._target_cost(jnp.asarray(a),jnp.asarray(anchor))
            return float(value),np.asarray(gradient)
        def eq(a):
            value=np.asarray(self._equilibrium(jnp.asarray(a),dj,bj,rj))
            return value[:4] if self.objective_mode=='economic' else value
        def jac(a):
            value=np.asarray(self._equilibrium_jacobian(jnp.asarray(a),dj,bj,rj))
            return value[:4] if self.objective_mode=='economic' else value
        inventory=np.r_[np.ones(3),np.zeros(3)][None,:]
        guesses=[initial]
        # Explore both flow branches of the nonmonotone B steady-state map.
        for phi,ca,cc in [(0.6,0.18,0.22),(1.6,0.4,0.1)]:
            guesses.append(np.clip(np.array([ca,reference,cc,anchor[3],phi,c['nominal_control'][1]]),b[:,0],b[:,1]))
        feasible=[]
        warm_candidate=None
        target_constraints=[{'type':'eq','fun':eq,'jac':jac},LinearConstraint(inventory,-np.inf,c['inventory_upper'])]
        if self.objective_mode=='economic' and self.economic['minimum_production']>0:
            target_constraints.append({'type':'ineq','fun':lambda a:a[4]*a[1]-self.economic['minimum_production'],
                                       'jac':lambda a:np.array([0.,a[4],0.,0.,a[1],0.])})
        for guess in guesses:
            result=minimize(objective,guess,jac=True,method='SLSQP',bounds=Bounds(b[:,0],b[:,1]),
                            constraints=target_constraints,
                            options={'maxiter':200,'ftol':1e-12})
            if result.success and np.isfinite(result.x).all() and np.max(np.abs(eq(result.x)))<=1e-7 and admissible_state(result.x[:4],c) and (self.objective_mode!='economic' or result.x[4]*result.x[1]>=self.economic['minimum_production']-1e-7):
                if len(feasible)==0 and np.array_equal(guess,guesses[0]):warm_candidate=result
                if not any(np.allclose(result.x,r.x,atol=1e-5,rtol=0) for r in feasible):feasible.append(result)
        if not feasible:
            raise RuntimeError(f'No verified feasible steady target for B={reference}: {result.message}')
        previous=np.asarray(c['nominal_control'] if previous is None else previous)
        old_target=self._steady_target
        def score(r):
            value=float(self._approach_cost(jnp.asarray(r.x),jnp.asarray(anchor),jnp.asarray(previous),dj,bj,rj))+r.fun
            if self.target_selection=='continuous' and old_target is not None:
                value+=0.1*np.sum(((r.x[4:]-old_target[4:])/np.asarray(c['mpc']['input_deviation_scales']))**2)
            return value
        result=min(feasible,key=(lambda r:r.fun) if self.objective_mode=='economic' else score)
        if self.objective_mode=='tracking' and self.target_selection=='continuous' and warm_candidate is not None and old_target is not None:
            # Retain the warm branch when its approach cost is nearly as good.
            best=score(result)
            if score(warm_candidate)<=best+0.05*max(1.,abs(best)):result=warm_candidate
        if old_target is not None and np.linalg.norm(result.x[4:]-old_target[4:])>0.25:self._target_switches+=1
        self._steady_target=result.x.copy()
        return result.x[:4],result.x[4:]

    def terminal_matrix(self,state_target,input_target,disturbance):
        """LQR tail for augmented [state error, previous-input error]."""
        m=self.config['mpc']
        legacy=np.diag([0.,m['terminal_tracking_weight']/m['concentration_tracking_scale']**2,0.,m['terminal_tracking_weight']*m['temperature_tracking_weight']/m['temperature_tracking_scale']**2,0.,0.])
        if self.terminal_cost=='legacy':
            self._terminal_info={'mode':'legacy'}
            return legacy
        derivatives=np.asarray(self._flow_jacobian(jnp.asarray(np.r_[state_target,input_target,disturbance])))
        A,B=derivatives[:,:4],derivatives[:,4:6]
        Aa=np.block([[A,B],[np.zeros((2,4)),np.eye(2)]])
        Ba=np.vstack((B,np.eye(2)))
        Qx=np.diag([1e-4,1/m['concentration_tracking_scale']**2,1e-4,m['temperature_tracking_weight']/m['temperature_tracking_scale']**2])
        Ru=np.diag(m['input_deviation_weight']/np.asarray(m['input_deviation_scales'])**2)
        Rd=np.diag(m['input_move_weight']/np.asarray(self.config['control_move_limits'])**2)
        W=np.block([[Qx,np.zeros((4,2))],[np.zeros((2,4)),Ru]])
        # Stage cost uses x_(i+1) and u_i: include the state/input cross term.
        Q=Aa.T@W@Aa; R=Rd+Ba.T@W@Ba; S=Aa.T@W@Ba
        try:
            P=solve_discrete_are(Aa,Ba,Q,R,s=S);P=(P+P.T)/2
            K=np.linalg.solve(R+Ba.T@P@Ba,Ba.T@P@Aa+S.T)
            residual=Aa.T@P@Aa-P-(Aa.T@P@Ba+S)@K+Q
            relative=np.linalg.norm(residual)/max(1.,np.linalg.norm(P))
            radius=float(np.max(np.abs(np.linalg.eigvals(Aa-Ba@K))))
            if not np.isfinite(P).all() or np.linalg.eigvalsh(P).min()<-1e-6 or relative>1e-7 or radius>=1:
                raise ValueError('Invalid or nonstabilizing Riccati solution')
            self._terminal_info={'mode':'lqr','closed_loop_spectral_radius':radius,'riccati_relative_residual':float(relative)}
            return P
        except (ValueError,np.linalg.LinAlgError) as error:
            self._terminal_info={'mode':'legacy_fallback','reason':str(error)}
            return legacy

    def reset(self):
        """Reset the observer and warm starts when restarting a process loop."""
        self.plan=None; self.bias=np.zeros(4)
        self._previous_raw_prediction=None; self._previous_context=None; self._steady_target=None; self._target_switches=0

    def _initial_plan(self,previous):
        candidate=np.tile(previous,(self.horizon,1)) if self.plan is None else np.concatenate((self.plan[1:],self.plan[-1:]))
        b=np.asarray(self.config['control_bounds']); move=np.asarray(self.config['control_move_limits'])
        candidate=candidate.copy(); last=previous
        for i in range(self.horizon):
            candidate[i]=np.clip(candidate[i],np.maximum(b[:,0],last-move),np.minimum(b[:,1],last+move))
            last=candidate[i]
        return candidate

    def step(self,state,previous_control,disturbance,reference=None):
        x=np.asarray(state,float); previous=np.asarray(previous_control,float)
        d=np.asarray(disturbance,float); n=self.horizon
        if d.shape==(2,): d=np.tile(d,(n,1))
        ref=self.config['mpc']['product_B_reference'] if reference is None else reference
        ref=np.broadcast_to(np.asarray(ref,float),(n,)).copy()
        if x.shape!=(4,) or previous.shape!=(2,) or d.shape!=(n,2) or not np.isfinite(np.r_[x,previous,d.ravel(),ref]).all():
            raise ValueError('Expected finite state(4), previous input(2), disturbance(2) or forecast(N,2), reference scalar or (N,)')
        if not np.all(self.model.in_domain(np.column_stack((np.tile(x,(n,1)),np.tile(previous,(n,1)),d)))):
            raise ValueError('Measured state, previous input, or disturbance outside the training domain')
        start=time.perf_counter()
        if self.offset_correction:
            if self._previous_raw_prediction is not None:
                last_state,last_disturbance=self._previous_context
                raw=np.asarray(self.model(np.r_[last_state,previous,last_disturbance]))[:4]
                innovation=x-raw
                self.bias=(1-self.bias_gain)*self.bias+self.bias_gain*innovation
            target_state,target_control=self.steady_target(d[-1],ref[-1],x,previous)
        else:
            target_state=np.zeros(4); target_control=np.asarray(self.config['nominal_control'])
        if self.objective_mode=='economic':ref=np.full(n,target_state[1])
        terminal=self.terminal_matrix(target_state,target_control,d[-1])
        bj,txj,tuj,tj=map(jnp.asarray,(self.bias,target_state,target_control,terminal))
        seed=self._initial_plan(previous).ravel()
        xj,pj,dj,rj=map(jnp.asarray,(x,previous,d,ref))
        move=np.tile(self.config['control_move_limits'],n)
        offset=np.zeros(2*n); offset[:2]=previous
        linear=LinearConstraint(self._difference,offset-move,offset+move)
        def cost(a):
            value,gradient=self._cost_gradient(jnp.asarray(a),xj,pj,dj,rj,bj,txj,tuj,tj)
            return float(value),np.asarray(gradient)
        def cons(a): return np.asarray(self._constraints(jnp.asarray(a),xj,dj,bj))
        def jac(a): return np.asarray(self._constraint_jacobian(jnp.asarray(a),xj,dj,bj))
        def feasible(a):
            return (np.isfinite(a).all() and np.all(a>=self._bounds.lb-self.tolerance) and np.all(a<=self._bounds.ub+self.tolerance)
                    and np.max(np.abs(self._difference@a-offset)-move)<=self.tolerance
                    and np.min(cons(a))>=-self.tolerance)
        result=minimize(cost,seed,jac=True,method='SLSQP',bounds=self._bounds,
                        constraints=[linear,{'type':'ineq','fun':cons,'jac':jac}],
                        options={'maxiter':self.max_iterations,'ftol':1e-6 if self.objective_mode=='economic' else 1e-8,'disp':False})
        fallback=False
        if result.success and feasible(result.x): chosen=result.x
        elif feasible(seed): chosen=seed; fallback=True
        else: raise RuntimeError(f'MPC has no verified feasible candidate: {result.message}')
        self.plan=chosen.reshape((n,2))
        states,_=self._rollout(jnp.asarray(chosen),xj,dj,bj)
        self._previous_raw_prediction=np.asarray(self.model(np.r_[x,self.plan[0],d[0]]))[:4]
        self._previous_context=(x.copy(),d[0].copy())
        return self.plan[0].copy(),{
            'terminal_info':self._terminal_info.copy(),'effective_reference':float(ref[0]),
            'estimated_bias':self.bias.copy(),'steady_target_state':target_state.copy(),
            'steady_target_control':target_control.copy(),
            'solver_success':bool(result.success),'solver_message':str(result.message),
            'iterations':int(result.nit),'fallback_used':fallback,
            'solve_seconds':time.perf_counter()-start,'objective':cost(chosen)[0],
            'predicted_constraint_violation':float(max(0.,-np.min(cons(chosen)))),
            'predicted_next_state':np.asarray(states[0])}


def simulate(args):
    controller=MPCController(args.model_dir,args.horizon,args.max_iterations,state_margin=args.temperature_margin,offset_correction=not args.no_offset_correction,bias_gain=args.bias_gain,terminal_cost=args.terminal_cost,target_selection=args.target_selection,objective_mode=args.objective,quality_band=(args.quality_lower,args.quality_upper),production_weight=args.production_weight,heat_weight=args.heat_weight,flow_weight=args.flow_weight,quality_penalty=args.quality_penalty,quality_margin=args.quality_margin,minimum_production=args.minimum_production,shortfall_penalty=args.shortfall_penalty)
    c=controller.config; ts=c['sampling_interval']
    x=np.asarray(c['nominal_state'],float); previous=np.asarray(c['nominal_control'],float)
    baseline=x.copy(); baseline_u=previous.copy()
    states=[x.copy()]; baselines=[baseline.copy()]; controls=[]; disturbances=[]; references=[]
    predictions=[]; diagnostics=[]; dense_states=[]; stop_reason=None
    for k in range(args.steps):
        fraction=k/max(1,args.steps)
        reference=args.reference
        d=np.asarray(c['nominal_disturbance'],float)
        changed=fraction>=0.4 if args.change_time is None else k*ts>=args.change_time
        if args.scenario=='tracking' and changed: reference=args.second_reference
        if args.scenario=='disturbance' and changed: d=np.array([0.95,0.1])
        try: u,info=controller.step(x,previous,d,reference)
        except (ValueError,RuntimeError) as error:
            stop_reason=str(error); break
        reference=info['effective_reference']
        plant=integrate_interval(x,u,d,c)
        nominal=integrate_interval(baseline,baseline_u,d,c)
        states.append(plant['next_state']); baselines.append(nominal['next_state'])
        controls.append(u); disturbances.append(d); references.append(info['effective_reference'])
        predictions.append(info.pop('predicted_next_state')); diagnostics.append(info)
        dense_states.append(plant['states'])
        x=plant['next_state']; baseline=nominal['next_state']; previous=u
        if args.print_every and (k==0 or (k+1)%args.print_every==0):
            print(f"step {k+1}/{args.steps}: B={x[1]:.6f}, target={reference:.4f}, phi={u[0]:.4f}, q={u[1]:.4f}, solve={info['solve_seconds']:.3f}s",flush=True)
        if not np.all(admissible_state(plant['states'],c)):
            stop_reason='Reference plant exceeded a sampled operating constraint'; break
    if not controls: raise RuntimeError(stop_reason or 'No control intervals completed')
    states,baselines,controls,disturbances,references,predictions,dense_states=map(np.asarray,(states,baselines,controls,disturbances,references,predictions,dense_states))
    errors=states[1:,1]-references; base_errors=baselines[1:,1]-references
    times=np.asarray([v['solve_seconds'] for v in diagnostics])
    moves=np.diff(np.concatenate((np.asarray(c['nominal_control'])[None,:],controls)),axis=0)
    metrics={
        'task':'closed_loop_cstr_mpc','scenario':args.scenario,'model_dir':str(Path(args.model_dir).resolve()),
        'requested_steps':args.steps,'completed_steps':len(controls),'stop_reason':stop_reason,
        'simulation_completed':len(controls)==args.steps and stop_reason is None,
        'product_B_tracking_rmse':float(np.sqrt(np.mean(errors**2))),
        'nominal_control_tracking_rmse':float(np.sqrt(np.mean(base_errors**2))),
        'final_product_B':float(states[-1,1]),'final_reference':float(references[-1]),
        'final_tracking_error':float(errors[-1]),
        'tail_tracking_mean_error':float(errors[-min(20,len(errors)):].mean()),
        'objective_mode':controller.objective_mode,'economic_config':controller.economic if controller.objective_mode=='economic' else None,
        'offset_correction':controller.offset_correction,'bias_gain':controller.bias_gain,
        'terminal_cost':controller.terminal_cost,'target_selection':controller.target_selection,
        'terminal_lqr_fallback_steps':sum(v['terminal_info']['mode']=='legacy_fallback' for v in diagnostics),
        'target_large_change_steps':controller._target_switches,
        'one_step_state_rmse_closed_loop':float(np.sqrt(np.mean((predictions-states[1:])**2))),
        'maximum_sampled_plant_state_violation':float(np.maximum(state_constraint_values(dense_states,c),0).max()),
        'maximum_control_move_violation':float(np.maximum(np.abs(moves)-np.asarray(c['control_move_limits']),0).max()),
        'maximum_control_bound_violation':float(max(0.,np.max(np.asarray(c['control_bounds'])[:,0]-controls),np.max(controls-np.asarray(c['control_bounds'])[:,1]))),
        'fallback_steps':sum(v['fallback_used'] for v in diagnostics),
        'solver_success_steps':sum(v['solver_success'] for v in diagnostics),
        'first_solve_seconds_including_compilation':float(times[0]),
        'warm_solve_mean_seconds':float(times[1:].mean()) if len(times)>1 else None,
        'warm_solve_max_seconds':float(times[1:].max()) if len(times)>1 else None,
        'sampling_interval_seconds':ts*60,'horizon':controller.horizon,
        'temperature_margin':args.temperature_margin,'reference':args.reference,'second_reference':args.second_reference,
        'max_iterations':args.max_iterations,'change_time_minutes':args.change_time if args.change_time is not None else 0.4*args.steps*ts,'plant_config':c,
        'validation_scope':'Full-state, noise-free simulation against source equations; midpoint/endpoint prediction constraints and dense sampled plant checks; no physical deployment or stability guarantee.'}
    if controller.objective_mode=='economic':
        lower,upper=controller.economic['quality_band'];cb=dense_states[...,1]
        quality=np.maximum(lower-cb,0)+np.maximum(cb-upper,0)
        metrics.update(mean_production_proxy=float(np.mean(controls[:,0]*states[1:,1])),
                       integrated_production_proxy=float(np.sum(controls[:,0]*np.trapezoid(dense_states[...,1],dx=ts/(c['interval_nodes']-1),axis=1))),
                       final_production_proxy=float(controls[-1,0]*states[-1,1]),
                       mean_heat_effort=float(np.mean((controls[:,1]/0.6)**2)),mean_flow_effort=float(np.mean((controls[:,0]/2)**2)),
                       maximum_sampled_quality_violation=float(quality.max()),quality_endpoint_fraction_in_band=float(np.mean((states[1:,1]>=lower-1e-6)&(states[1:,1]<=upper+1e-6))),
                       tail_maximum_quality_violation=float(quality[-min(20,len(quality)):].max()))
    if args.save:
        destination=RESULTS/'output'/'control'/Path(args.model_dir).name/(args.scenario+('_economic' if controller.objective_mode=='economic' else '_offset_free_lqr' if controller.offset_correction and controller.terminal_cost=='lqr' else '_offset_free' if controller.offset_correction else ''))
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        np.savez(destination/'trajectory_predictions.npz',time=np.arange(len(states))*ts,states=states,
                 baseline_states=baselines,controls=controls,disturbances=disturbances,references=references,
                 predicted_next_states=predictions,interval_states=dense_states,solve_seconds=times,
                 solver_success=np.asarray([v['solver_success'] for v in diagnostics]),
                 fallback_used=np.asarray([v['fallback_used'] for v in diagnostics]),
                 estimated_bias=np.asarray([v['estimated_bias'] for v in diagnostics]),
                 steady_target_states=np.asarray([v['steady_target_state'] for v in diagnostics]),
                 steady_target_controls=np.asarray([v['steady_target_control'] for v in diagnostics]))
        metrics['output_dir']=str(destination)
    print(json.dumps({k:v for k,v in metrics.items() if k!='plant_config'},indent=2))
    if args.plot:
        import matplotlib.pyplot as plt
        t=np.arange(len(controls))*ts
        fig,axes=plt.subplots(3,1,sharex=True)
        axes[0].plot(t,states[1:,1],label='MPC B'); axes[0].plot(t,baselines[1:,1],label='Nominal control B')
        axes[0].step(t,references,where='post',label='Operating target')
        if controller.objective_mode=='economic':axes[0].axhspan(*controller.economic['quality_band'],alpha=.15,color='green',label='Quality band')
        axes[0].legend()
        axes[1].plot(t,states[1:,3]);axes[1].set_ylabel('theta')
        for i,name in enumerate(['phi','q']):axes[2].step(t,controls[:,i],where='post',label=name)
        axes[2].legend();axes[2].set_xlabel('Time (minutes)');plt.show()
    return 0 if metrics['simulation_completed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--horizon',type=int)
    p.add_argument('--max-iterations',type=int,default=100)
    p.add_argument('--scenario',choices=['nominal','tracking','disturbance'],default='tracking')
    p.add_argument('--reference',type=float,default=0.5)
    p.add_argument('--second-reference',type=float,default=0.48)
    p.add_argument('--change-time',type=float,help='Time in minutes for reference/feed change; default 40 percent of run')
    p.add_argument('--temperature-margin',type=float,default=0.)
    p.add_argument('--print-every',type=int,default=10)
    p.add_argument('--no-offset-correction',action='store_true',help='Reproduce the original nominal-input/zero-temperature penalty')
    p.add_argument('--bias-gain',type=float,default=0.3)
    p.add_argument('--terminal-cost',choices=['lqr','legacy'],default='legacy')
    p.add_argument('--target-selection',choices=['continuous','legacy'],default='legacy')
    p.add_argument('--objective',choices=['tracking','economic'],default='tracking')
    p.add_argument('--quality-lower',type=float,default=0.48)
    p.add_argument('--quality-upper',type=float,default=0.50)
    p.add_argument('--production-weight',type=float,default=1.0)
    p.add_argument('--heat-weight',type=float,default=0.05)
    p.add_argument('--flow-weight',type=float,default=0.05)
    p.add_argument('--quality-penalty',type=float,default=1000.0)
    p.add_argument('--quality-margin',type=float,default=1e-4)
    p.add_argument('--minimum-production',type=float,default=0.0)
    p.add_argument('--shortfall-penalty',type=float,default=100.0)
    p.add_argument('--save',action='store_true')
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    if args.steps<1: p.error('--steps must be positive')
    return simulate(args)

if __name__=='__main__': raise SystemExit(main())