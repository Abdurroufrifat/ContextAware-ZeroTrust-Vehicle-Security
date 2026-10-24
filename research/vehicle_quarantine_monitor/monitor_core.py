"""Offline latched jump heuristic. No physical or provenance guarantee."""
from pathlib import Path
import hashlib,io,json,os,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
STEP_MPS=1.;GAP_NS=100_000_000
CASES=('nominal','radar_plus3','peer_plus3','radar_peer_plus3','fusion_peer_plus3','all_peer_plus3','dropout','dropout_radar_plus3','dropout_radar_peer_plus3')
MODELS=('radar_imu_parent','nonlinear7','linear7','huber7','extra_trees7')
METHODS=('radar_only',)+tuple(m+'__deadband__'+s for m in MODELS for s in ('withhold','fallback'))
FIELDS=('route','status','prediction_mps','latest_endpoint','age_ns','primary_status','radar_status')
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

def monitor(times,prediction,base_accepted):
 """Input has already passed the existing selector. Never grants a denial.

 First eligible sample is an unverified anchor. A gap >100ms reanchors only
 before a latch. A jump >1m/s within100ms quarantines all later proposals.
 No automatic release, truth, case name or future sample enters this rule.
 """
 t=np.asarray(times);p=np.asarray(prediction);b=np.asarray(base_accepted)
 if t.ndim!=1 or t.dtype.kind not in 'iu' or p.shape!=t.shape or b.shape!=t.shape or b.dtype.kind!='b' or np.any(t<0) or np.any(t[1:]<=t[:-1]):raise ValueError('strict aligned integer clocks and boolean acceptance required')
 if np.any(b&(~np.isfinite(p)|(p<0))):raise ValueError('accepted estimates must be finite and nonnegative')
 allowed=np.zeros(len(t),bool);reason=np.ones(len(t),np.uint8);first=None;last=None
 for i in range(len(t)):
  if not b[i]:continue
  if first is not None:reason[i]=3;continue
  gap=last is not None and int(t[i])-int(t[last])>GAP_NS
  if last is not None and not gap and abs(float(p[i])-float(p[last]))>STEP_MPS:
   first=i;reason[i]=2;continue
  allowed[i]=True;reason[i]=4 if gap else 0;last=i
 return {'allowed':allowed,'reason':reason,'first_alarm':first}

def filtered(old,allowed):
 out={k:np.asarray(v).copy() for k,v in old.items()};deny=(out['route']>0)&~allowed
 for k in ('route',):out[k][deny]=0
 out['status'][deny]=4;out['prediction_mps'][deny]=np.nan;out['latest_endpoint'][deny]=-1;out['age_ns'][deny]=-1
 return out

def summary(out,times,truth,masks):
 results={}
 for name,mask in masks.items():
  accepted=mask&(out['route']>0);valid=accepted&np.isfinite(truth);e=abs(out['prediction_mps'][valid]-truth[valid]);count=len(e)
  errors={'count':count,'missing_truth':int(np.sum(accepted&~np.isfinite(truth))),'mae_mps':float(e.mean()) if count else None,'rmse_mps':float(np.sqrt(np.mean(e**2))) if count else None,'p95_mps':float(np.quantile(e,.95)) if count else None,'maximum_mps':float(e.max()) if count else None,'above_2_mps':int(np.sum(e>2.))}
  availability=float(accepted.sum()/mask.sum());pairs=mask[1:]&mask[:-1]&(out['route'][1:]>0)&(out['route'][:-1]>0)&(np.diff(times)<=50_000_000);steps=abs(np.diff(out['prediction_mps']))[pairs]
  gates={'availability_ge_95pct':availability>=.95,'p95_le_0_5_mps':errors['p95_mps'] is not None and errors['p95_mps']<=.5,'no_missing_accepted_truth':errors['missing_truth']==0,'no_accepted_error_above_2_mps':errors['above_2_mps']==0}
  results[name]={'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':availability,'route_counts':{str(k):int(np.sum(mask&(out['route']==k))) for k in range(3)},'primary_status_counts':{str(k):int(np.sum(mask&(out['primary_status']==k))) for k in range(5)},'radar_status_counts':{str(k):int(np.sum(mask&(out['radar_status']==k))) for k in range(5)},'accepted_errors':errors,'gates':gates,'joint_gate_pass':bool(all(gates.values())),'adjacent_accepted_pairs':len(steps),'source_transitions':int(np.sum(pairs&(out['route'][1:]!=out['route'][:-1]))),'adjacent_prediction_step_p95_mps':float(np.quantile(steps,.95)) if len(steps) else None,'adjacent_prediction_step_maximum_mps':float(steps.max()) if len(steps) else None}
 return results

def fixture_inputs():
 t=np.arange(61,dtype=np.int64)*50_000_000;i=np.arange(61);constant=np.full(61,10.)
 return {'abrupt_plus3':(t,constant+np.where(i>=10,3.,0.),constant),'slow_plus3':(t,constant+np.minimum(np.maximum(i-9,0)*.1,3.),constant),'startup_plus3':(t,constant+3.,constant),'gap_plus3':(np.array([0,100_000_001],np.int64),np.array([10.,13.]),np.array([10.,10.])),'benign_fast_change':(t,constant+np.where(i>=10,1.01,0.),constant+np.where(i>=10,1.01,0.)),'subthreshold_ramp':(t,constant+np.minimum(np.maximum(i-9,0)*.9,3.),constant)}
def fixture_results():
 rows={}
 for name,(t,p,y) in fixture_inputs().items():
  out=monitor(t,p,np.ones(len(t),bool));e=abs(p[out['allowed']]-y[out['allowed']]);rows[name]={'samples':len(t),'accepted':int(out['allowed'].sum()),'first_alarm':out['first_alarm'],'gap_reanchors':int(np.sum(out['reason']==4)),'accepted_error_above_2':int(np.sum(e>2.)),'maximum_accepted_error_mps':float(e.max()) if len(e) else None,'allowed':out['allowed'].tolist(),'reason':out['reason'].tolist()}
 return rows

def check_bundle():
 lock=read(ROOT/'analysis_lock.json')
 if digest(ROOT/'protocol.json')!=lock['protocol_sha256']:raise ValueError('monitor protocol changed')
 for name,h in lock['code_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('monitor code changed '+name)
 for name,h in read(ROOT/'sha256.json').items():
  if blocked(name) or digest(ROOT/name)!=h:raise ValueError('monitor artifact changed '+name)
 return lock

def prior_inputs():
 """Read only existing source-compromise and explicitly locked prior inputs."""
 old=RESEARCH/'vehicle_source_compromise';manifest=read(PROJECT/'PROJECT_MANIFEST.json');protocol=read(old/'protocol.json');sha=read(old/'sha256.json');paths={'research/vehicle_source_compromise/sha256.json':digest(old/'sha256.json')}
 for name,h in sha.items():
  if blocked(name):raise ValueError('protected prior artifact')
  path=old/name
  if digest(path)!=h:raise ValueError('prior artifact changed '+name)
  paths['research/vehicle_source_compromise/'+name]=h
 expected={'vehicle_radar_fallback/fallback_core.py','vehicle_radar_fallback/verify_fallback.py','vehicle_nonlinear_correction/nonlinear_core.py','vehicle_availability_audit/request_reference.json','vehicle_masked_history/frame_index.json','vehicle_multisource_imu/development_replay.npz','vehicle_night_training/predictions.npz','vehicle_night_training/decisions.npz','vehicle_matched_head_comparison/results.json',*[folder+'/'+case+suffix for case in ('nominal','dropout') for folder,suffix in [('vehicle_sensor_faults/replays','.npz'),('vehicle_deadband_guard/replays','__deadband.npz'),('vehicle_matched_head_comparison/replays','.npz')]]}
 if set(protocol['input_sha256'])!=expected:raise ValueError('prior input coverage changed')
 for name,h in protocol['input_sha256'].items():
  if blocked(name):raise ValueError('protected input')
  if digest(RESEARCH/name)!=h:raise ValueError('prior input changed '+name)
  paths['research/'+name]=h
 for name,h in paths.items():
  if manifest.get(name)!=h:raise ValueError('project manifest mismatch '+name)
 if protocol['cases']!=list(CASES) or protocol['methods']!=list(METHODS):raise ValueError('source matrix changed')
 frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');events=np.array([v['time_ns'] for v in frames],np.int64);refs=read(RESEARCH/'vehicle_availability_audit/request_reference.json');times=np.array([v['time_ns'] for v in refs],np.int64);truth=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in refs],float)
 if len(events)!=40690 or len(times)!=37340 or np.any(np.diff(events)<=0) or np.any(np.diff(times)<=0):raise ValueError('fixed schedule changed')
 duration=int(events[-1])-int(events[0]);start=int(events[0])+duration*2//5;end=int(events[0])+duration*3//5;segment=(times-times[0])//300_000_000_000
 masks={'all':np.ones(len(times),bool),'before':times<start,'during':(times>=start)&(times<end),'after':times>=end,**{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}}
 prior=read(old/'results.json')
 if set(prior['cases'])!=set(CASES) or prior['decision']!='SOURCE_COMPROMISE_BOUNDARY_COMPLETE_NO_PROMOTION':raise ValueError('prior case matrix changed')
 return old,paths,times,truth,events,masks,prior

def proposal(saved,method,times,events):
 out={k:saved[method+'__'+k] for k in FIELDS}
 if any(v.shape!=times.shape for v in out.values()):raise ValueError('request alignment')
 accepted=out['route']>0
 if not np.array_equal(accepted,out['status']==0):raise ValueError('route/status mismatch')
 if any(out[k].dtype.kind not in 'iu' for k in ('route','status','latest_endpoint','age_ns','primary_status','radar_status')):raise ValueError('integer metadata required')
 if np.any(out['route']>2) or np.any(out['status']>4):raise ValueError('invalid route/status')
 if np.any(accepted&(~np.isfinite(out['prediction_mps'])|(out['prediction_mps']<0))):raise ValueError('invalid accepted estimate')
 expected=np.searchsorted(events,times,side='right')-1
 if np.any(accepted&((out['latest_endpoint']!=expected)|(expected<0))):raise ValueError('causal endpoint mismatch')
 age=times-events[np.maximum(expected,0)]
 if np.any(accepted&((out['age_ns']!=age)|(age<0)|(age>GAP_NS))):raise ValueError('source age mismatch')
 return out
