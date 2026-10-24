"""Frozen-output, same-row diagnostics. No fitting or policy promotion."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
MODELS=('context_boosting','affine_median_matched','ridge_context','huber_context')
GROUPS=('Urban01','Urban03','Urban04')
SPEED_NAMES=('lt1','1to5','5to10','ge10')
GAP_NS=150_000_000
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def vector(x):
 x=np.asarray(x,dtype=float)
 if x.ndim!=1 or not np.isfinite(x).all() or np.any(x<0):raise ValueError('expected finite nonnegative vector')
 return x

def speed_bins(y):return np.searchsorted(np.array([1.,5.,10.]),vector(y),side='right').astype(np.int8)

def metrics(y,p):
 y=vector(y);p=vector(p)
 if y.shape!=p.shape:raise ValueError('unaligned target/prediction')
 if not len(y):return {'count':0,**{k:None for k in ['mae_mps','rmse_mps','p95_mps','maximum_mps','bias_prediction_minus_reference_mps']},'above_half_mps':0,'above_2_mps':0}
 residual=p-y;e=np.abs(residual)
 return {'count':len(y),'mae_mps':float(e.mean()),'rmse_mps':float(np.sqrt(np.mean(e**2))),'p95_mps':float(np.quantile(e,.95)),'maximum_mps':float(e.max()),'bias_prediction_minus_reference_mps':float(residual.mean()),'above_half_mps':int(np.sum(e>.5)),'above_2_mps':int(np.sum(e>2.))}

def timestamps(t):
 t=np.asarray(t)
 if t.ndim!=1 or not len(t) or t.dtype.kind!='i' or np.any(t<0) or np.any(t[1:]<=t[:-1]):raise ValueError('expected nonempty strictly increasing integer timestamps')
 return t

def time_thirds(t):
 t=timestamps(t);span=int(t[-1])-int(t[0])
 if span==0:return np.zeros(len(t),dtype=np.int8)
 return np.array([min(2,((int(v)-int(t[0]))*3)//span) for v in t],dtype=np.int8)

def error_runs(t,errors,gap_ns=GAP_NS):
 t=timestamps(t);errors=vector(errors)
 if errors.shape!=t.shape or type(gap_ns) is not int or gap_ns<=0:raise ValueError('invalid supported-row run input')
 result=[];active=[]
 def finish():
  if active:
   first,last=active[0],active[-1];result.append({'first_local_index':first,'last_local_index':last,'first_time_ns':int(t[first]),'last_time_ns':int(t[last]),'rows':len(active),'span_ns':int(t[last])-int(t[first]),'peak_absolute_error_mps':float(np.max(errors[active]))});active.clear()
 for i,e in enumerate(errors):
  if e<=.5:finish();continue
  if active and int(t[i])-int(t[active[-1]])>gap_ns:finish()
  active.append(i)
 finish();return result

def overlap(a,b):
 a=np.asarray(a);b=np.asarray(b)
 if a.ndim!=1 or a.dtype.kind!='b' or b.dtype.kind!='b' or b.shape!=a.shape:raise ValueError('unaligned exceedance flags')
 return {'both':int(np.sum(a&b)),'hgb_only':int(np.sum(a&~b)),'baseline_only':int(np.sum(~a&b)),'neither':int(np.sum(~a&~b))}

def build(members,y,predictions):
 y=vector(y)
 if len(y)!=len(members) or not len(y) or set(predictions)!=set(MODELS):raise ValueError('incomplete row/model population')
 p={name:vector(predictions[name]) for name in MODELS}
 if any(v.shape!=y.shape for v in p.values()):raise ValueError('prediction row count differs')
 if any(v['id']!=i or type(v['time_ns']) is not int or type(v['day']) is not bool or v['group'] not in GROUPS or v['route'].split('/')[0]!=v['group'] or v['route'].endswith('-Day')!=v['day'] for i,v in enumerate(members)):raise ValueError('row identity or route/day metadata differs')
 if len({(v['route'],v['time_ns']) for v in members})!=len(members):raise ValueError('duplicate recording timestamp')
 windows=np.array([v['window_id'] for v in members]);routes=np.array([v['route'] for v in members]);groups=np.array([v['group'] for v in members]);days=np.array([v['day'] for v in members]);time=np.array([v['time_ns'] for v in members],dtype=np.int64);bins=speed_bins(y);thirds=np.full(len(y),-1,dtype=np.int8);run_records=[];window_spans={}
 def cell(mask):
  return {'count':int(mask.sum()),'models':{name:metrics(y[mask],a[mask]) for name,a in p.items()}}
 allmask=np.ones(len(y),dtype=bool);bywindow={}
 for name in sorted(set(windows.tolist())):
  ids=np.flatnonzero(windows==name)
  if len(set(routes[ids].tolist()))!=1:raise ValueError('one window spans multiple recordings')
  ids=ids[np.argsort(time[ids],kind='stable')];timestamps(time[ids]);thirds[ids]=time_thirds(time[ids]);window_spans[name]={'first_supported_time_ns':int(time[ids[0]]),'last_supported_time_ns':int(time[ids[-1]]),'supported_span_ns':int(time[ids[-1]])-int(time[ids[0]])};bywindow[name]=cell(windows==name)
  for model in MODELS:
   for run in error_runs(time[ids],np.abs(p[model][ids]-y[ids])):
    first,last=run['first_local_index'],run['last_local_index'];run_records.append({'model':model,'window_id':name,'route':routes[ids[0]].item(),'first_row_id':int(ids[first]),'last_row_id':int(ids[last]),**run})
 if np.any(thirds<0):raise ValueError('unassigned supported-time stratum')
 byspeed={label:cell(bins==i) for i,label in enumerate(SPEED_NAMES)};bythird={name:{str(k):cell((windows==name)&(thirds==k)) for k in range(3)} for name in bywindow};window_speed={name:{label:cell((windows==name)&(bins==i)) for i,label in enumerate(SPEED_NAMES)} for name in bywindow}
 failed=[name for name,s in bywindow.items() if s['models']['context_boosting']['p95_mps']>.5];failedmask=np.isin(windows,failed);strata={'failed_hgb_windows':{label:cell(failedmask&(bins==i)) for i,label in enumerate(SPEED_NAMES)},'other_hgb_windows':{label:cell(~failedmask&(bins==i)) for i,label in enumerate(SPEED_NAMES)}}
 hgbflag=np.abs(p['context_boosting']-y)>.5;overlaps={name:overlap(hgbflag,np.abs(p[name]-y)>.5) for name in MODELS[1:]};summaries={}
 for name in MODELS:
  selected=[v for v in run_records if v['model']==name];summaries[name]={'run_count':len(selected),'exceedance_rows':sum(v['rows'] for v in selected),'longest_sample_adjacent_span_ns':max((v['span_ns'] for v in selected),default=0),'longest_row_count':max((v['rows'] for v in selected),default=0)}
 total=int(hgbflag.sum());concentration=[{'window_id':name,'count':v['count'],'hgb_exceedance_rows':v['models']['context_boosting']['above_half_mps'],'fraction_of_all_hgb_exceedances':v['models']['context_boosting']['above_half_mps']/total if total else 0.,'p95_mps':v['models']['context_boosting']['p95_mps']} for name,v in bywindow.items()];concentration.sort(key=lambda v:(-v['hgb_exceedance_rows'],v['window_id']))
 result={'scope':'Retrospective supported-row error localization; associations only, no established root cause','supported_rows':len(y),'models':list(MODELS),'pooled':cell(allmask),'by_group':{name:cell(groups==name) for name in sorted(set(groups.tolist()))},'by_recording':{name:cell(routes==name) for name in sorted(set(routes.tolist()))},'by_window':bywindow,'by_speed':byspeed,'by_day_night':{'day':cell(days),'night':cell(~days)},'by_window_speed':window_speed,'by_window_third':bythird,'window_supported_spans':window_spans,'failed_hgb_windows':failed,'failed_vs_other_by_speed':strata,'hgb_exceedance_concentration':concentration,'baseline_exceedance_overlap':overlaps,'sample_adjacent_run_summary':summaries,'all_rows_retained':True,'all_models_share_each_stratum':True,'p95_diagnostic_limit_mps':.5,'root_cause_established':False,'reference_independence_established':False,'new_model_fitted':False,'parent_development_gate_still_failed':True,'promotion_allowed':False,'reserved_final_opened':False,'witness_enabled':False,'decision':'FAILURE_DIAGNOSIS_COMPLETE_PARENT_GATE_STILL_FAILED'}
 arrays={'row_id':np.arange(len(y),dtype=np.int64),'reference_speed_bin':bins,'within_window_supported_third':thirds,'in_failed_hgb_window':failedmask,'hgb_error_above_half':hgbflag}
 return result,arrays,run_records

def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['models']!=list(MODELS) or p['speed_boundaries_mps']!=[1.,5.,10.] or p['max_run_gap_ns']!=GAP_NS or p['diagnostic_limit_mps']!=.5:raise ValueError('diagnostic constants changed')
 for name,h in p['input_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT.parent) or digest(path)!=h:raise ValueError('frozen diagnostic input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ['diagnosis_core.py','generate_diagnosis.py']}}
 target=ROOT/'analysis_lock.json'
 if target.exists() and read(target)!=lock:raise ValueError('diagnostic code changed after freeze')
 if create and not target.exists():write(target,lock)
 return p

def load_inputs():
 freeze();route=ROOT.parent/'vehicle_route_validation';matched=ROOT.parent/'vehicle_matched_baselines';members=read(route/'membership.json');support=read(route/'support_costs.json');allowed=read(route/'protocol.json')['routes']
 with np.load(matched/'oof_predictions.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 if set(arrays)!={n+'__prediction_mps' for n in MODELS}|{'target_mps','fold_id'}:raise ValueError('frozen prediction schema differs')
 with np.load(route/'oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(z['target_mps'],arrays['target_mps']) or not np.array_equal(z['fold_id'],arrays['fold_id']) or not np.array_equal(z['prediction_mps'],arrays['context_boosting__prediction_mps']):raise ValueError('parent frozen rows/targets/predictions differ')
 if len(members)!=27178 or set(v['route'] for v in members)!=set(allowed) or set(v['group'] for v in members)!=set(GROUPS) or not np.array_equal(arrays['fold_id'],np.array([GROUPS.index(v['group']) for v in members])):raise ValueError('fixed route cohort differs')
 if set(support)!=set(allowed):raise ValueError('support recording population differs')
 for route_name,cost in support.items():
  n=sum(v['route']==route_name for v in members)
  if n!=cost['supported_rows_used'] or n+cost['upstream_prefix_not_indexed']+cost['unsupported_or_unmatched_indexed']!=cost['raw_endpoints_in_fixed_windows']:raise ValueError('support denominator differs')
 return members,arrays['target_mps'],{n:arrays[n+'__prediction_mps'] for n in MODELS},support
