"""Offline fixed-rate request accounting; no model, sensor or permission changes."""
from pathlib import Path
import collections,hashlib,json,math,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'vehicle_continuous_radar'
sys.path.insert(0,str(PREVIOUS))
import continuous_core as c
STATUS={0:'usable',1:'no_prior_endpoint',2:'latest_endpoint_stale',3:'latest_endpoint_unready',4:'latest_endpoint_quality_withheld'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def request_times(start,end,period=50000000):
 if any(type(v) is not int for v in (start,end,period)) or end<start or period<=0:raise ValueError('invalid request schedule')
 return start+np.arange((end-start)//period+1,dtype=np.int64)*period
def serve(times,events,ready,prediction,quality,ttl):
 times=np.asarray(times);events=np.asarray(events);ready=np.asarray(ready);prediction=np.asarray(prediction);quality=np.asarray(quality)
 if times.ndim!=1 or events.ndim!=1 or times.dtype.kind not in 'iu' or events.dtype.kind not in 'iu' or np.any(np.diff(times)<=0) or np.any(np.diff(events)<=0) or not len(events):raise ValueError('ambiguous request or event clock')
 if any(a.shape!=events.shape for a in (ready,prediction,quality)) or ready.dtype.kind!='b' or quality.dtype.kind!='b' or type(ttl) is not int or ttl<0:raise ValueError('unaligned scheduler inputs')
 if np.any(quality&~ready) or np.any(~np.isfinite(prediction[ready])) or np.any(prediction[ready]<0) or np.any(np.isfinite(prediction[~ready])):raise ValueError('invalid endpoint readiness or prediction')
 latest=np.searchsorted(events,times,side='right')-1;present=latest>=0;ids=np.maximum(latest,0);age=np.where(present,times-events[ids],-1);status=np.zeros(len(times),dtype=np.uint8)
 status[~present]=1;fresh=present&(age<=ttl);status[present&~fresh]=2;status[fresh&~ready[ids]]=3;status[fresh&ready[ids]&~quality[ids]]=4
 usable=status==0;timing=usable|(status==4)
 return {'request_times_ns':times,'latest_endpoint':latest,'age_ns':age,'status':status,'prediction_mps':np.where(usable,prediction[ids],np.nan),'timing_prediction_mps':np.where(timing,prediction[ids],np.nan)}
def endpoint_reasons(rows):
 grid=c.e.RadarGrid(rows)
 if not rows:raise ValueError('empty radar source')
 start=rows[0]['bag_time_ns'];out=[]
 for ordinal,row in enumerate(rows):
  t=row['bag_time_ns'];reason='ready';stale_ticks=0;maximum_age=None;stale_after=[]
  if t-start<6350000000:reason='startup'
  elif not len(grid.times):reason='no_valid_history'
  else:
   ticks=np.int64(t)-np.arange(127,-1,-1,dtype=np.int64)*50000000;j=np.searchsorted(grid.times,ticks,side='right')-1
   if np.any(j<0):reason='no_prior_history'
   elif grid.times[j[-1]]!=t:reason='current_sample_invalid'
   elif np.any(grid.times[j]<start):reason='history_before_start'
   else:
    age=ticks-grid.times[j];stale_ticks=int((age>300000000).sum());maximum_age=int(age.max())
    if stale_ticks:reason='history_stale';stale_after=[int(v) for v in np.unique(grid.times[j[age>300000000]])]
    elif np.searchsorted(grid.times,t,side='right')-np.searchsorted(grid.times,t-1000000000,side='left')<10:reason='anchor_count'
  out.append({'ordinal':ordinal,'time_ns':t,'reason':reason,'stale_ticks':stale_ticks,'maximum_history_age_ns':maximum_age,'stale_after_valid_ns':stale_after})
 return out
def scores(y,p,usable):
 y=np.asarray(y);p=np.asarray(p);usable=np.asarray(usable)
 if y.ndim!=1 or p.shape!=y.shape or usable.shape!=y.shape or usable.dtype.kind!='b' or np.any(~np.isfinite(p[usable])):raise ValueError('unaligned/nonfinite accepted request scores')
 known=usable&np.isfinite(y);out=c.q.error_metrics(y[known],p[known]);out['missing_truth']=int((usable&~np.isfinite(y)).sum());return out
def episodes(times,status,period):
 out=[];begin=0
 while begin<len(status):
  end=begin+1
  while end<len(status) and (status[end]!=0)==(status[begin]!=0):end+=1
  if status[begin]!=0:
   out.append({'first_request_ns':int(times[begin]),'last_request_ns':int(times[end-1]),'requests':end-begin,'scheduled_slot_equivalent_s':(end-begin)*period/1e9,'status_counts':{STATUS[int(k)]:int(v) for k,v in zip(*np.unique(status[begin:end],return_counts=True))}})
  begin=end
 return out
def freeze(create=False):
 protocol=read(ROOT/'protocol.json')
 for name,h in protocol['input_sha256'].items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT.parent) or digest(p)!=h:raise ValueError('availability input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ('availability_core.py','run_availability.py')},'models_refitted':False,'thresholds_tuned':False}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('availability analysis lock changed')
 if create and not path.exists():write(path,lock)
 return protocol
