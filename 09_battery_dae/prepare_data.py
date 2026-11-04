"""Download and prepare the public CALB LFP measurements (Zenodo 6323623)."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parent
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from pathlib import Path
import urllib.request,json,hashlib
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
CASE=_CASE_DIR
RECORD='https://zenodo.org/api/records/6323623'
TRAIN=['CCCV_18A_chg.csv','CCCV_18A_dis.csv','CCCV_180A_chg.csv','CCCV_180A_dis.csv','pulsed_chg.csv','pulsed_dis.csv']
VALIDATION=['CCCV_50A_chg.csv','CCCV_50A_dis.csv']
TEST=['halfcycles.csv','loadprofile.csv']

def compress_curve(x,y,tolerance=5e-7):
    keep={0,len(x)-1}; pending=[(0,len(x)-1)]
    while pending:
        a,b=pending.pop()
        if b-a<2: continue
        error=abs(y[a:b+1]-np.interp(x[a:b+1],x[[a,b]],y[[a,b]])); c=a+int(np.argmax(error))
        if error.max()>tolerance: keep.add(c); pending.extend([(a,c),(c,b)])
    idx=np.array(sorted(keep)); return x[idx],y[idx]

def main():
    raw=CASE/'data'/'raw'; raw.mkdir(parents=True,exist_ok=True)
    info=json.load(urllib.request.urlopen(RECORD)); checks={}
    for f in info['files']:
        if not f['key'].endswith('.csv'): continue
        path=raw/f['key']
        if not path.exists(): urllib.request.urlretrieve(f['links']['self'],path)
        actual=hashlib.md5(path.read_bytes()).hexdigest()
        if actual!=f['checksum'].split(':')[-1]: raise ValueError('Source checksum mismatch: '+str(path))
        checks[f['key']]={'zenodo_md5':actual,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (raw/'zenodo_record.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    curve=pd.read_csv(raw/'OCV_SOC_data.csv',sep=';').to_numpy(dtype=float); curve=curve[np.argsort(curve[:,0])]
    ox,oy=compress_curve(curve[:,0],curve[:,1]); ocv_error=float(np.max(abs(np.interp(curve[:,0],ox,oy)-curve[:,1])))
    arrays={'ocv_soc':ox,'ocv_voltage':oy}; names=TRAIN+VALIDATION+TEST; metadata=[]
    for index,name in enumerate(names):
        frame=pd.read_csv(raw/name,sep=';'); original_count=len(frame)
        frame=frame.replace([np.inf,-np.inf],np.nan).dropna().sort_values(frame.columns[0]).drop_duplicates(frame.columns[0],keep='last')
        values=frame.to_numpy(dtype=float); t=values[:,0]-values[0,0]; current=values[:,1].copy(); current[abs(current)<.25]=0; voltage=values[:,2]
        if np.any(np.diff(t)<=0): raise ValueError('Times must strictly increase')
        initial_soc=float(np.clip(np.interp(voltage[0],oy,ox),0,1)); charge=cumulative_trapezoid(current,t,initial=0)
        arrays.update({f't_{index}':t,f'i_{index}':current,f'v_{index}':voltage,f'charge_{index}':charge,f'soc0_{index}':np.array(initial_soc)})
        split='train' if name in TRAIN else 'validation' if name in VALIDATION else 'test'
        metadata.append({'id':index,'file':name,'split':split,'original_measurements':original_count,'retained_measurements':len(t),'duration_seconds':float(t[-1]),'initial_soc_from_first_rest_voltage':initial_soc,'initial_current_A':float(current[0])})
    destination=CASE/'data'/'data.npz'; np.savez_compressed(destination,**arrays)
    meta={'source_doi':'10.5281/zenodo.6323623','source_checksums':checks,'data_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'protocols':metadata,'measurement_counts':{s:sum(m['retained_measurements'] for m in metadata if m['split']==s) for s in ['train','validation','test']},'ocv_original_points':len(curve),'ocv_compressed_points':len(ox),'ocv_compression_max_error_V':ocv_error,'current_sign':'positive discharge; negative charge','units':{'time':'s','current':'A','voltage':'V','capacity':'Ah'},'initialization':'Only first observed voltage, near zero current, used for OCV inverse SOC estimate. Initial polarization=0 is an assumption. SOC/polarization are unobserved latent states.','split_protocol':'Entire 50 A protocols reserved for validation; independent half-cycle and home-storage load profiles reserved for test. No row-random split.'}
    (CASE/'data'/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(json.dumps(meta['measurement_counts'],indent=2)); print('OCV compressed',len(curve),'to',len(ox),'max error',ocv_error)
if __name__=='__main__': main()
