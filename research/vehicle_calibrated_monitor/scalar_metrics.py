"""Reused independently checked arithmetic; no previous monitor behavior."""
from pathlib import Path
import hashlib,io,json,math,os,tempfile
import numpy as np
def close(a,b,path=''):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('keys '+path)
  for k in a:close(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list):
  if len(a)!=len(b):raise AssertionError('length '+path)
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+'/'+str(i))
 elif isinstance(a,float):
  if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=0,abs_tol=1e-9):raise AssertionError('numeric '+path)
 elif a!=b:raise AssertionError('value '+path)

def quantile(values):
 values=sorted(values);at=(len(values)-1)*.95;i=int(at);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(at-i)

def scalar_summary(out,allowed,times,truth,masks):
 cells={name:{'requests':0,'route':[0]*3,'primary':[0]*5,'radar':[0]*5,'errors':[],'missing':0,'steps':[],'switches':0} for name in masks};previous=None
 for i,t in enumerate(times):
  route=int(out['route'][i]) if allowed[i] else 0;p=float(out['prediction_mps'][i]) if route else math.nan;groups=[name for name,mask in masks.items() if mask[i]]
  for name in groups:
   c=cells[name];c['requests']+=1;c['route'][route]+=1;c['primary'][int(out['primary_status'][i])]+=1;c['radar'][int(out['radar_status'][i])]+=1
   if route:
    if math.isfinite(float(truth[i])):c['errors'].append(abs(p-float(truth[i])))
    else:c['missing']+=1
  if previous is not None:
   old_t,old_route,old_p,old_groups=previous
   if route and old_route and int(t)-old_t<=50_000_000:
    for name in groups:
     if name in old_groups:cells[name]['steps'].append(abs(p-old_p));cells[name]['switches']+=int(route!=old_route)
  previous=int(t),route,p,groups
 result={}
 for name,c in cells.items():
  e=c['errors'];n=len(e);a=c['route'][1]+c['route'][2];availability=a/c['requests'];p95=quantile(e) if n else None;above=sum(v>2. for v in e)
  errors={'count':n,'missing_truth':c['missing'],'mae_mps':math.fsum(e)/n if n else None,'rmse_mps':math.sqrt(math.fsum(v*v for v in e)/n) if n else None,'p95_mps':p95,'maximum_mps':max(e) if n else None,'above_2_mps':above}
  gates={'availability_ge_95pct':availability>=.95,'p95_le_0_5_mps':p95 is not None and p95<=.5,'no_missing_accepted_truth':c['missing']==0,'no_accepted_error_above_2_mps':above==0}
  result[name]={'requests':c['requests'],'accepted_requests':a,'availability':availability,'route_counts':{str(k):v for k,v in enumerate(c['route'])},'primary_status_counts':{str(k):v for k,v in enumerate(c['primary'])},'radar_status_counts':{str(k):v for k,v in enumerate(c['radar'])},'accepted_errors':errors,'gates':gates,'joint_gate_pass':all(gates.values()),'adjacent_accepted_pairs':len(c['steps']),'source_transitions':c['switches'],'adjacent_prediction_step_p95_mps':quantile(c['steps']) if c['steps'] else None,'adjacent_prediction_step_maximum_mps':max(c['steps']) if c['steps'] else None}
 return result
