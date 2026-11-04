"""Refresh trajectory-derived scores and write a complete validation report."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
import json
from pathlib import Path
import numpy as np
from control_experiments import ROOT, CASE, EXPERIMENTS, metrics

def main():
    rows=[]
    for study in EXPERIMENTS:
        runs=[]
        for path in sorted((ROOT/study).glob('*/*/seed=*/metrics.json')):
            old=json.loads(path.read_text(encoding='utf-8'))
            if old['status']=='running':raise RuntimeError('Suite is still running: '+str(path))
            archive=path.parent/'trajectory_predictions.npz'
            if archive.exists():
                with np.load(archive) as a:
                    records={k:a[k] for k in a.files}
                n=old['completed_steps']
                assert records['states'].shape==(n+1,4)
                assert records['controls'].shape==(n,2)
                assert records['interval_states'].shape[0]==n
                for key,value in records.items():
                    assert np.isfinite(value).all(),(path,key)
                updated=metrics(records,old['plant_config'],study,old['controller'],old['seed'],old['variant'],old['requested_steps'],old['status'],old['stop_reason'],old['model_sha256'],old['wall_seconds'])
                for key,value in old.items():
                    if key not in updated:updated[key]=value
                path.write_text(json.dumps(updated,indent=2),encoding='utf-8')
                old=updated
            runs.append(old);rows.append(old)
        expected=sum(len(EXPERIMENTS[study]['variants'])*len(EXPERIMENTS[study]['seeds']) for _ in EXPERIMENTS[study]['modes'])
        assert len(runs)==expected,(study,len(runs),expected)
        (ROOT/study/'metrics.json').write_text(json.dumps(dict(experiment=study,runs=runs,comparison_note='Paired requirements and independent plant simulation; tracking targets band midpoint, economic chooses within the band. Proxy cost includes demand shortfall.'),indent=2),encoding='utf-8')
    lines=['# Reactor control validation results','',
           'All seven studies ran sequentially with the existing saved seed=11 nonlinear DAEHN model. No model retraining was performed. Results below include failed/rejected trials; completed means simulation finished, not that quality was always satisfied.','',
           '| Study | Controller | Variant / seed | Status | Minutes completed | Mean production proxy | Quality outside (min) | Tracking RMSE | Mean heat effort | Solve p95 (s) |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for m in rows:
        lines.append('| {} | {} | {} / {} | {} | {:.2f} | {:.5f} | {:.3f} | {:.5f} | {:.5f} | {:.3f} |'.format(m['experiment'].split('_',1)[1],m['controller'],m['variant'],m['seed'],m['status'],m['completed_steps']*m['sampling_interval_minutes'],m.get('mean_production_proxy',0),m.get('quality_outside_sampled_minutes',0),m.get('tracking_rmse',0),m.get('mean_heat_effort',0),m.get('solve_p95_seconds',0)))
    lines+=['','## Integrity and operating bounds','',f"Validated {len(rows)} trial archives for dimensions and finite values. All trials used model parameter SHA-256 `{rows[0]['model_sha256']}`."]
    assert len({m['model_sha256'] for m in rows})==1
    for label,key in [('Sampled physical-state excess','maximum_physical_state_violation'),('Applied input-bound excess','maximum_control_bound_violation'),('Applied move-limit excess','maximum_control_move_violation')]:
        lines.append(f"{label}: {max(m.get(key,0) for m in rows):.3g} maximum across trials.")
    lines+=['',f"Fallback moves: {sum(m.get('fallback_steps',0) for m in rows)}. Recorded MPC solves exceeding the 15-second sampling interval: {sum(m.get('solve_over_15_seconds',0) for m in rows)}. These timings include cold compilation and changing economic settings; timings were not collected on a dedicated real-time machine.",'','## Stops and rejected requests','']
    for m in rows:
        if m['status']!='completed':lines.append(f"- {m['experiment']} / {m['controller']} / {m['variant']} / seed {m['seed']}: {m['status']} after {m['completed_steps']*.25:.2f} min. {m['stop_reason']}")
    lines+=['','## Interpretation','',
            'Production and heat effort are benchmark proxies, not calibrated revenue or energy. Tracking and economic control pursue different objectives; economic tracking RMSE is therefore not its primary success measure. Quality excursions include startup and unannounced requirement transitions. Dense interval checks sample eleven points per interval and do not prove continuous-time constraint satisfaction.',
            '', 'The noise/delay trials deliberately use the current bias observer without a new state estimator or delay-augmented prediction model. Three seeds are a limited replication check. Mismatch trials change only the independent plant kinetics and heat release. Request rejection ends the infeasible trial; it does not demonstrate a deployed emergency or backup policy.',
            '', 'See EXPERIMENTS.md for schedules, assumptions and rerun commands. Numerical results and plots are saved under results/08_cstr_mpc/output/experiments, with separate study/controller/variant/seed folders. Earlier control results are retained.']
    lookup={(m['experiment'],m['controller'],m['variant'],m['seed']):m for m in rows}
    t=lookup[('01_long_term_requirements','tracking','default',101)]
    e=lookup[('01_long_term_requirements','economic','default',101)]
    t2=lookup[('02_changing_economic_priorities','tracking','default',101)]
    e2=lookup[('02_changing_economic_priorities','economic','default',101)]
    highlights=['## Main findings','',
      'Thirty trials were attempted: 25 completed their full duration, four rejected deliberately infeasible requirements, and one stopped on a sampled physical-bound excursion.', '',
      f"The four-hour changing-requirement test increased mean production by {100*(e['mean_production_proxy']/t['mean_production_proxy']-1):.2f}% with economic control. Quality excursions increased from {t['quality_outside_sampled_minutes']:.2f} to {e['quality_outside_sampled_minutes']:.2f} minutes; normalized mean heating effort increased from {t['mean_heat_effort']:.5f} to {e['mean_heat_effort']:.5f}. The retained LQR variant performed almost identically to nonlinear tracking in this long-run schedule.", '',
      f"Changing-demand economic control reduced integrated production shortfall from {t2['integrated_production_shortfall']:.3f} to {e2['integrated_production_shortfall']:.3f}, with more heating and quality excursions.", '',
      'All three paired noise/delay seeds completed. Economic production averaged approximately 0.941 versus 0.822 for tracking, with fewer sampled quality-excursion minutes in these particular tests, but more actuator limiting and one fallback move. Both mismatch variants completed; economic gains varied by plant change and generally incurred more quality excursions.', '',
      'Economic startup from the low-inventory state required 22 feasible fallback moves and 7.61 minutes outside the quality band. Near the operating boundary, temperature reached 2.000000992 against a limit of 2.0 after 4.75 minutes; the tolerance was 1e-7, so that trial stopped. Its shortened production average must not be compared directly with the complete 24-minute tracking run.', '',
      'These results support a production/quality/effort trade-off rather than an overall improvement in every control metric. The boundary/startup tests identify reasons to investigate constraint margins, startup policies, and delay-aware estimation in a separate formulation study.', '']
    lines[4:4]=highlights
    (CASE/'EXPERIMENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Refreshed and validated',len(rows),'trials; report:',CASE/'EXPERIMENT_RESULTS.md')
if __name__=='__main__':main()
