"""Offline post-estimator compromise at the existing policy input boundary.

Does not simulate raw radar perturbation, forging a signature, or takeover.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
sys.path.insert(0,str(RESEARCH/'vehicle_radar_fallback'));import fallback_core as fb
sys.path.insert(0,str(RESEARCH/'vehicle_nonlinear_correction'));import nonlinear_core as n
read=fb.read;write=fb.write;load=fb.load;digest=fb.digest
MODELS=('radar_imu_parent','nonlinear7','linear7','huber7','extra_trees7')
METHODS=('radar_only',)+tuple(m+'__deadband__'+s for m in MODELS for s in ('withhold','fallback'))
CASES=('nominal','radar_plus3','peer_plus3','radar_peer_plus3','fusion_peer_plus3','all_peer_plus3','dropout','dropout_radar_plus3','dropout_radar_peer_plus3')
def permitted(name):return not any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation'))
def attack_vectors(events,start,end,radar,predictions,peer,case):
 if case not in CASES:raise ValueError('unknown compromise case')
 events=np.asarray(events);r=np.asarray(radar,float).copy();q=np.asarray(peer,float).copy();p={k:np.asarray(v,float).copy() for k,v in predictions.items()}
 if events.ndim!=1 or not np.issubdtype(events.dtype,np.integer) or start>=end or any(v.shape!=events.shape for v in (r,q,*p.values())):raise ValueError('aligned endpoint vectors and integer clocks required')
 active=(events>=start)&(events<end);tag=case.removeprefix('dropout_')
 if tag in ('radar_plus3','radar_peer_plus3','all_peer_plus3'):r[active]+=3.
 if tag in ('peer_plus3','radar_peer_plus3','fusion_peer_plus3','all_peer_plus3'):q[active]+=3.
 if tag in ('fusion_peer_plus3','all_peer_plus3'):
  for v in p.values():v[active]+=3.
 return r,p,q,active

def endpoint_quality(p,peer,source,imu_status):
 return np.asarray(source)&(np.asarray(imu_status)==0)&np.isfinite(p)&np.isfinite(peer)&(p>=0)&(peer>=0)&(abs(p-peer)<=1.)
def make_branch(times,events,ready,p,peer,source,imu_status,binding,content):
 keep=endpoint_quality(p,peer,source,imu_status);out=n.serve(times,events,ready,p,keep,binding)
 reject=(out['status']==0)&~content;out['status'][reject]=4;out['prediction_mps'][out['status']!=0]=np.nan
 return out

def freeze(create=False):
 protocol=read(ROOT/'protocol.json')
 if protocol['cases']!=list(CASES) or protocol['methods']!=list(METHODS) or protocol['offset_mps']!=3.:raise ValueError('comparison scope changed')
 for name,h in protocol['input_sha256'].items():
  if not permitted(name):raise ValueError('protected input')
  if digest(RESEARCH/name)!=h:raise ValueError('input changed '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ('compromise_core.py','run_compromise.py','verify_compromise.py','test_compromise.py')}}
 if (ROOT/'analysis_lock.json').exists():
  if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('analysis lock changed')
 elif create:write(ROOT/'analysis_lock.json',lock)
 else:raise ValueError('freeze before scoring')

def inputs():
 times,y,events,masks=fb.cohort();frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');ready=np.array([v['ready'] for v in frames],bool)
 source=load(RESEARCH/'vehicle_night_training/decisions.npz')['day_plus_night__quality'];peer=np.full(len(events),np.nan);peer[ready]=load(RESEARCH/'vehicle_night_training/predictions.npz')['day_plus_night__masked_cnn_4ch']
 return times,y,events,masks,ready,source,peer

def branches(case,data):
 times,y,events,masks,ready,source,peer=data;basecase='dropout' if case.startswith('dropout') else 'nominal'
 base=load(RESEARCH/'vehicle_sensor_faults/replays'/(basecase+'.npz'));guard=load(RESEARCH/'vehicle_deadband_guard/replays'/(basecase+'__deadband.npz'));heads=load(RESEARCH/'vehicle_matched_head_comparison/replays'/(basecase+'.npz'))
 def expand(values):
  out=np.full(len(events),np.nan);out[ready]=values;return out
 radar=expand(base['radar_only__ready_prediction_mps']);pred={m:expand((base if m in MODELS[:2] else heads)[m+'__ready_prediction_mps']) for m in MODELS}
 duration=int(events[-1])-int(events[0]);start=int(events[0])+duration*2//5;end=int(events[0])+duration*3//5
 r,p,q,active=attack_vectors(events,start,end,radar,pred,peer,case)
 radar_branch=make_branch(times,events,ready,r,q,source,np.zeros(len(events),int),np.ones(len(times),bool),np.ones(len(times),bool))
 primary={m:make_branch(times,events,ready,p[m],q,source,base['current__status'],base['binding'],guard['request_content_fresh']) for m in MODELS}
 if case in ('nominal','dropout'):
  for k in fb.FIELDS:np.testing.assert_array_equal(radar_branch[k],base['radar_only__request__'+k])
  for m in MODELS:
   for k in fb.FIELDS:
    key=m+'__request__'+k if m in MODELS[:2] else m+'__deadband__withhold__'+k
    compare=primary[m] if m in MODELS[:2] else fb.select(primary[m],radar_branch,False)
    np.testing.assert_array_equal(compare[k],(guard if m in MODELS[:2] else heads)[key])
 return radar_branch,primary,{'attack_active_endpoint':active,'radar_ready_prediction_mps':r[ready],'peer_ready_prediction_mps':q[ready],**{m+'__ready_prediction_mps':p[m][ready] for m in MODELS}}

def generate():
 freeze()
 if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('one-shot scoring already attempted')
 (ROOT/'replays').mkdir();data=inputs();times,y,events,masks,*_=data;results={};historical=read(RESEARCH/'vehicle_matched_head_comparison/results.json')
 for case in CASES:
  radar,primary,saved=branches(case,data);outputs={'radar_only':fb.select(radar,radar,False)}
  for m in MODELS:
   for strategy in ('withhold','fallback'):outputs[m+'__deadband__'+strategy]=fb.select(primary[m],radar,strategy=='fallback')
  results[case]={name:fb.summarize(out,times,y,masks) for name,out in outputs.items()}
  if case in ('nominal','dropout') and results[case]!=historical['cases'][case]:raise ValueError('historical control mismatch '+case)
  for name,out in outputs.items():
   for key,v in out.items():saved[name+'__'+key]=v
  fb.atomic_npz(ROOT/'replays'/(case+'.npz'),saved)
  cell=results[case]['extra_trees7__deadband__fallback']['during'];print(case,'ET fallback during:',cell['accepted_requests'],'/',cell['requests'],'P95',cell['accepted_errors']['p95_mps'],'>2m/s',cell['accepted_errors']['above_2_mps'],flush=True)
 flags={k:False for k in ('models_refitted','labels_changed','thresholds_changed','production_policy_changed','reserved_final_opened','promotion_allowed','raw_sensor_attack_simulated','authenticated_provenance_established')}
 write(ROOT/'results.json',{'decision':'SOURCE_COMPROMISE_BOUNDARY_COMPLETE_NO_PROMOTION','cases':results,**flags})
 print('SOURCE_COMPROMISE_BOUNDARY_COMPLETE_NO_PROMOTION',flush=True)
