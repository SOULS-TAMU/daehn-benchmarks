"""Train matched battery baselines/formulations and optionally execute their campaigns."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import argparse,json,subprocess,sys,hashlib,time
from model import VARIANTS
CASE=_CASE_DIR
RESULTS=CASE.parent/'results'/CASE.name

def main(default_models=VARIANTS):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=VARIANTS,default=list(default_models))
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    p.add_argument('--epochs',type=int,default=3000); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--window-length',type=int,default=128); p.add_argument('--hidden-size','--hidden-nodes',type=int,default=32)
    p.add_argument('--hidden-layers',type=int,default=2); p.add_argument('--activation',choices=['tanh','relu'],default='tanh')
    p.add_argument('--learning-rate',type=float,default=.001); p.add_argument('--print-every',type=int,default=50)
    p.add_argument('--validate-every',type=int,default=1)
    p.add_argument('--steps-per-epoch',type=int); p.add_argument('--capacity-prior-weight',type=float,default=.001)
    p.add_argument('--soc-bound-weight',type=float,default=10.); p.add_argument('--fixed-capacity-ah',type=float,default=191.5)
    p.add_argument('--hysteresis-rate',type=float,default=20.); p.add_argument('--hysteresis-relaxation-seconds',type=float,default=10000.)
    p.add_argument('--study',required=True); p.add_argument('--analyze',action='store_true')
    p.add_argument('--analysis-study',default='case_study_v2'); args=p.parse_args()
    if Path(args.study).name!=args.study or Path(args.analysis_study).name!=args.analysis_study: p.error('Studies must be single folder names')
    destination=RESULTS/'suites'/args.study; destination.mkdir(parents=True,exist_ok=True)
    manifest_path=destination/'suite_manifest.json'
    if manifest_path.exists(): p.error('Suite already exists; choose a new study to preserve its results')
    metadata=json.loads((CASE/'data'/'metadata.json').read_text())
    manifest={'schema_version':1,'settings':vars(args),'dataset_sha256':metadata['data_sha256'],'runs':[],
              'inequality_projection_added':False,'source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in CASE.glob('*.py')}}
    write=lambda:manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    write()
    shared=['epochs','batch_size','window_length','hidden_size','hidden_layers','activation','learning_rate','print_every','validate_every','capacity_prior_weight','soc_bound_weight','fixed_capacity_ah','hysteresis_rate','hysteresis_relaxation_seconds']
    if args.steps_per_epoch is not None: shared.append('steps_per_epoch')
    for model in dict.fromkeys(args.models):
        for seed in dict.fromkeys(args.seeds):
            branch=RESULTS/'model'/args.study/model; before=set(branch.glob('*/parameter_tree.pkl'))
            command=[sys.executable,str(CASE/'train.py'),'--model',model,'--seed',str(seed),'--study',args.study]
            for key in shared: command += ['--'+key.replace('_','-'),str(getattr(args,key))]
            row={'model':model,'seed':seed,'command':command,'status':'running'}; manifest['runs'].append(row); write()
            print(f'TRAIN {model} seed={seed}',flush=True); start=time.perf_counter()
            with (destination/f'{model}_seed{seed}_training.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=CASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
                for line in process.stdout: log.write(line); log.flush(); print(line,end='',flush=True)
                status=process.wait()
            row['wall_seconds']=time.perf_counter()-start
            if status: row['status']='failed'; write(); raise RuntimeError(f'{model} failed; see suite training log')
            created=set(branch.glob('*/parameter_tree.pkl'))-before
            if len(created)!=1: raise RuntimeError('Expected one new checkpoint')
            row.update(model_dir=str(created.pop().parent),status='trained'); write()
            if args.analyze:
                subprocess.run([sys.executable,str((_CASE_DIR / "_impl" / 'analyze_suite.py')),'--model-dir',row['model_dir'],'--study',args.analysis_study],cwd=CASE,check=True)
                row['status']='analyzed'; write()
    if args.analyze:
        subprocess.run([sys.executable,str((_CASE_DIR / "_impl" / 'analyze_suite.py')),'--suite-manifest',str(manifest_path),'--study',args.analysis_study,'--compare-only'],cwd=CASE,check=True)
    print('SUITE MANIFEST',manifest_path,flush=True)

if __name__=='__main__': main()
