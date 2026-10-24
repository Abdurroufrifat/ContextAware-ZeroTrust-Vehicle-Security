"""Fixed-output paired robustness; no fitting, inference, or row selection."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
GROUPS=('Urban01','Urban03','Urban04')
MODELS=('context_boosting','affine_median_matched','ridge_context','huber_context','magnitude_context','magnitude_context_anchor')
CONTROL=MODELS[0];COMPARATORS=MODELS[1:];KEYS=('mae_mps','rmse_mps','p95_mps');TOL=1e-12
FAMILIES={'window':'window_id','recording':'route','route_group':'group'}
def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))
def write(f,v):Path(f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def vector(x):
 x=np.asarray(x,dtype=float)
 if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('expected nonempty finite nonnegative vector')
 return x

def metrics(y,p):
 y=vector(y);p=vector(p)
 if p.shape!=y.shape:raise ValueError('target/prediction alignment differs')
 residual=p-y;e=abs(residual)
 return {'count':len(y),'mae_mps':float(e.mean()),'rmse_mps':float(np.sqrt(np.mean(e**2))),'p95_mps':float(np.quantile(e,.95)),'maximum_mps':float(e.max()),'bias_prediction_minus_reference_mps':float(residual.mean()),'above_half_mps':int(np.sum(e>.5)),'above_2_mps':int(np.sum(e>2.))}

def labels_for(labels,n):
 a=np.asarray(labels)
 if a.shape!=(n,) or a.dtype.kind not in 'US' or any(not v for v in a.tolist()):raise ValueError('invalid aligned unit labels')
 return a

def balanced(y,p,labels):
 y=vector(y);p=vector(p)
 if p.shape!=y.shape:raise ValueError('balanced prediction alignment differs')
 labels=labels_for(labels,len(y));units=sorted(set(labels.tolist()));stats=[metrics(y[labels==u],p[labels==u]) for u in units]
 return {'units':len(units),'equal_unit_mae_mps':float(np.mean([v['mae_mps'] for v in stats])),'equal_unit_rmse_mps':float(np.sqrt(np.mean([v['rmse_mps']**2 for v in stats]))),'mean_unit_p95_mps':float(np.mean([v['p95_mps'] for v in stats]))}

def pair_counts(y,control,comparator):
 y=vector(y);a=vector(control);b=vector(comparator)
 if a.shape!=y.shape or b.shape!=y.shape:raise ValueError('pair alignment differs')
 ea=abs(a-y);eb=abs(b-y);delta=ea-eb;higha=ea>.5;highb=eb>.5
 return {'count':len(y),'control_lower_abs_error':int(np.sum(delta < -TOL)),'tied_abs_error':int(np.sum(abs(delta)<=TOL)),'control_higher_abs_error':int(np.sum(delta>TOL)),'above_half_pairs':{'both':int(np.sum(higha&highb)),'control_only':int(np.sum(higha&~highb)),'comparator_only':int(np.sum(~higha&highb)),'neither':int(np.sum(~higha&~highb))}}

def win_counts(deltas):
 x=np.asarray(deltas,dtype=float)
 if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid metric deltas')
 return {'control_lower':int(np.sum(x < -TOL)),'tied':int(np.sum(abs(x)<=TOL)),'control_higher':int(np.sum(x>TOL))}

def validate_members(members,n):
 if len(members)!=n or not n:raise ValueError('incomplete membership')
 seen=set();windows={}
 for i,m in enumerate(members):
  if type(m['id']) is not int or m['id']!=i or type(m['time_ns']) is not int or m['time_ns']<0 or type(m['day']) is not bool or m['group'] not in GROUPS or m['route'].split('/')[0]!=m['group'] or m['route'].endswith('-Day')!=m['day'] or not m['window_id']:raise ValueError('invalid row identity/route/day/time')
  identity=(m['route'],m['time_ns'])
  if identity in seen:raise ValueError('duplicate recording timestamp')
  seen.add(identity)
  if m['window_id'] in windows and windows[m['window_id']]!=m['route']:raise ValueError('window spans multiple recordings')
  windows[m['window_id']]=m['route']

def omission_masks(labels):
 a=labels_for(labels,len(labels));units=sorted(set(a.tolist()))
 if len(units)<2:raise ValueError('cannot omit only unit')
 return {name:a!=name for name in units}

def validate_fold_ids(members,fold):
 fold=np.asarray(fold)
 if fold.shape!=(len(members),) or fold.dtype.kind not in 'iu' or not np.array_equal(fold,np.array([GROUPS.index(v['group']) for v in members])):raise ValueError('fold IDs do not match excluded named routes')

def load_inputs():
 research=ROOT.parent;members=read(research/'vehicle_route_validation/membership.json');support=read(research/'vehicle_route_validation/support_costs.json')
 with np.load(research/'vehicle_matched_baselines/oof_predictions.npz',allow_pickle=False) as z:
  expected={n+'__prediction_mps' for n in MODELS[:4]}|{'target_mps','fold_id'}
  if set(z.files)!=expected:raise ValueError('matched prediction schema differs')
  y=z['target_mps'].copy();fold=z['fold_id'].copy();pred={n:z[n+'__prediction_mps'].copy() for n in MODELS[:4]}
 with np.load(research/'vehicle_magnitude_ablation/oof_predictions.npz',allow_pickle=False) as z:
  if set(z.files)!={n+'__prediction_mps' for n in ('original_context',)+MODELS[4:]}|{'target_mps','fold_id'} or not np.array_equal(y,z['target_mps']) or not np.array_equal(fold,z['fold_id']) or not np.array_equal(pred[CONTROL],z['original_context__prediction_mps']):raise ValueError('magnitude row/control identity differs')
  pred.update({n:z[n+'__prediction_mps'].copy() for n in MODELS[4:]})
 with np.load(research/'vehicle_route_validation/oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(y,z['target_mps']) or not np.array_equal(fold,z['fold_id']) or not np.array_equal(pred[CONTROL],z['prediction_mps']):raise ValueError('original route control differs')
 validate_members(members,len(y))
 validate_fold_ids(members,fold)
 if len(y)!=27178 or any(vector(v).shape!=y.shape for v in pred.values()) or len(set(v['route'] for v in members))!=5 or len(set(v['window_id'] for v in members))!=15:raise ValueError('fixed row/fold/unit population differs')
 return members,vector(y),pred,support

def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['models']!=list(MODELS) or p['comparison_metrics']!=list(KEYS) or p['families']!=FAMILIES or p['tie_tolerance']!=TOL or p['p95_limit_mps']!=.5 or p['model_refitting'] or p['new_inference'] or p['promotion_allowed']:raise ValueError('paired protocol changed')
 for name,h in p['input_sha256'].items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(ROOT.parent) or digest(f)!=h:raise ValueError('paired source changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['paired_core.py','build_paired.py']}}
 f=ROOT/'analysis_lock.json'
 if f.exists() and read(f)!=lock:raise ValueError('paired analysis changed after lock')
 if create and not f.exists():write(f,lock)
 return p

def build(members,y,pred,support):
 validate_members(members,len(y));y=vector(y)
 if set(pred)!=set(MODELS) or any(vector(v).shape!=y.shape for v in pred.values()):raise ValueError('model/row population differs')
 def cell(mask):
  stats={n:metrics(y[mask],v[mask]) for n,v in pred.items()}
  return {'count':int(mask.sum()),'models':stats,'control_minus_comparator':{n:{k:stats[CONTROL][k]-stats[n][k] for k in KEYS} for n in COMPARATORS}}
 pooled=cell(np.ones(len(y),bool));units={};balance={};wins={};omissions={};sensitivity={}
 for family,key in FAMILIES.items():
  labels=np.array([m[key] for m in members]);names=sorted(set(labels.tolist()));units[family]={name:cell(labels==name) for name in names};balance[family]={n:balanced(y,v,labels) for n,v in pred.items()};wins[family]={n:{k:win_counts([units[family][name]['control_minus_comparator'][n][k] for name in names]) for k in KEYS} for n in COMPARATORS};omissions[family]={name:{'omitted_rows':int(np.sum(~mask)),'retained':cell(mask)} for name,mask in omission_masks(labels).items()};sensitivity[family]={}
  for n in COMPARATORS:
   sensitivity[family][n]={}
   for k in KEYS:
    deltas=[v['retained']['control_minus_comparator'][n][k] for v in omissions[family].values()];sensitivity[family][n][k]={**win_counts(deltas),'min_control_minus_comparator_mps':float(min(deltas)),'max_control_minus_comparator_mps':float(max(deltas))}
 pairs={n:pair_counts(y,pred[CONTROL],pred[n]) for n in COMPARATORS}
 failures={n:{'pooled':pooled['models'][n]['p95_mps']>.5,'route_groups':[u for u,v in units['route_group'].items() if v['models'][n]['p95_mps']>.5],'recordings':[u for u,v in units['recording'].items() if v['models'][n]['p95_mps']>.5],'windows':[u for u,v in units['window'].items() if v['models'][n]['p95_mps']>.5]} for n in MODELS}
 return {'scope':'Descriptive fixed-output paired robustness; previously inspected outcomes, no inferential significance or new model selection','supported_rows':len(y),'models':list(MODELS),'pooled':pooled,'by_unit':units,'equal_unit_summaries':balance,'unit_metric_wins':wins,'paired_row_counts':pairs,'diagnostic_omissions':omissions,'omission_sensitivity':sensitivity,'diagnostic_failures':failures,'raw_support_costs':support,'omission_changes_main_cohort':False,'independent_replications_claimed':False,'p_values_computed':False,'population_confidence_intervals_computed':False,'model_refitting':False,'new_inference':False,'promotion_allowed':False,'parent_full_recording_development_gate_still_failed':True,'reserved_final_opened':False,'witness_enabled':False,'huber_optimizer_caveat':'All3 parent Huber fits reached max_iter500; retained final iterates, no convergence claim','decision':'PAIRED_ROBUSTNESS_COMPLETE_NO_PROMOTION'}
