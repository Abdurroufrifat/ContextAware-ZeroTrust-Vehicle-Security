"""Reused independently checked arithmetic; no previous monitor behavior."""
from pathlib import Path
import hashlib,io,json,math,os,tempfile
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def blocked(s):return any(t in str(s).lower() for t in ('urban02','rav4','vehicle_final_evaluation'))

def write(p,value):
 p=Path(p);raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();atomic(p,raw)

def atomic(p,raw):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=None
 try:
  with tempfile.NamedTemporaryFile(dir=p.parent,delete=False) as f:temp=Path(f.name);f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(temp,p)
 finally:
  if temp is not None:temp.unlink(missing_ok=True)

def write_npz(p,values):
 b=io.BytesIO();np.savez_compressed(b,**values);atomic(p,b.getvalue())

def summary(out,times,truth,masks):
 results={}
 for name,mask in masks.items():
  accepted=mask&(out['route']>0);valid=accepted&np.isfinite(truth);e=abs(out['prediction_mps'][valid]-truth[valid]);count=len(e)
  errors={'count':count,'missing_truth':int(np.sum(accepted&~np.isfinite(truth))),'mae_mps':float(e.mean()) if count else None,'rmse_mps':float(np.sqrt(np.mean(e**2))) if count else None,'p95_mps':float(np.quantile(e,.95)) if count else None,'maximum_mps':float(e.max()) if count else None,'above_2_mps':int(np.sum(e>2.))}
  availability=float(accepted.sum()/mask.sum());pairs=mask[1:]&mask[:-1]&(out['route'][1:]>0)&(out['route'][:-1]>0)&(np.diff(times)<=50_000_000);steps=abs(np.diff(out['prediction_mps']))[pairs]
  gates={'availability_ge_95pct':availability>=.95,'p95_le_0_5_mps':errors['p95_mps'] is not None and errors['p95_mps']<=.5,'no_missing_accepted_truth':errors['missing_truth']==0,'no_accepted_error_above_2_mps':errors['above_2_mps']==0}
  results[name]={'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':availability,'route_counts':{str(k):int(np.sum(mask&(out['route']==k))) for k in range(3)},'primary_status_counts':{str(k):int(np.sum(mask&(out['primary_status']==k))) for k in range(5)},'radar_status_counts':{str(k):int(np.sum(mask&(out['radar_status']==k))) for k in range(5)},'accepted_errors':errors,'gates':gates,'joint_gate_pass':bool(all(gates.values())),'adjacent_accepted_pairs':len(steps),'source_transitions':int(np.sum(pairs&(out['route'][1:]!=out['route'][:-1]))),'adjacent_prediction_step_p95_mps':float(np.quantile(steps,.95)) if len(steps) else None,'adjacent_prediction_step_maximum_mps':float(steps.max()) if len(steps) else None}
 return results
