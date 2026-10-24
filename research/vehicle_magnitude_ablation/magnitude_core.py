"""Two fixed magnitude ablations; no held-label fitting or model selection."""
from pathlib import Path
import hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vehicle_route_validation'))
import route_core as r
c=r.c
sys.path.insert(0,str(ROOT.parent/'vehicle_failure_diagnosis'))
import diagnosis_core as d
from verify_diagnosis import close
read=r.read;write=r.write;digest=r.digest
VARIANTS=('magnitude_context','magnitude_context_anchor')
MODELS=('original_context',)+VARIANTS

def magnitude_grid(x):
 x=np.asarray(x)
 if x.ndim!=3 or x.shape[1:]!=(4,128) or not np.isfinite(x).all() or np.any((x[:,3]!=0)&(x[:,3]!=1)):raise ValueError('invalid aligned radar grid')
 out=x.copy();out[:,0]=np.abs(out[:,0]);return out

def assemble(base,anchor,median):
 base=np.asarray(base,dtype=float);anchor=np.asarray(anchor,dtype=float);median=np.asarray(median,dtype=float)
 if base.ndim!=2 or base.shape[1]!=75 or anchor.shape!=(len(base),) or median.shape!=anchor.shape or not np.isfinite(base).all() or not np.isfinite(anchor).all() or not np.isfinite(median).all() or np.any(anchor<0) or np.any(median<0):raise ValueError('invalid magnitude feature assembly')
 out=base.copy();out[:,1]=anchor;out[:,2]=median;return out

def feature_names(variant):
 if variant not in VARIANTS:raise ValueError('unknown fixed magnitude variant')
 return ['current_magnitude_div10','signed_median_anchor_mps' if variant==VARIANTS[0] else 'magnitude_median_anchor_mps','abs_signed_median_raw' if variant==VARIANTS[0] else 'median_abs_raw']+['magnitude_'+v for v in c.FEATURE_NAMES[3:]]

def new_anchor_calibration(data,groups,day,held,median_abs):
 median_abs=np.asarray(median_abs,dtype=float)
 if median_abs.shape!=np.asarray(data['y']).shape or not np.isfinite(median_abs).all() or np.any(median_abs<0):raise ValueError('magnitude anchor population differs')
 return r.calibrate({**data,'median_feature':median_abs},groups,day,held)

def validated_calibration(rebuilt,saved):
 close(rebuilt,saved);return saved

def array_digest(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()

def load_data():
 data,members,groups,day,support=r.load_data()
 with np.load(ROOT.parent/'vehicle_trace_audit/row_traces.npz',allow_pickle=False) as z:
  if not np.array_equal(z['row_id'],np.arange(len(members))) or not np.array_equal(z['abs_signed_median_raw'],data['median_feature']):raise ValueError('source-audited median row identity differs')
  median_abs=z['median_abs_raw'].copy()
 if median_abs.shape!=data['y'].shape or not np.isfinite(median_abs).all() or np.any(median_abs<0):raise ValueError('invalid source-audited magnitude median')
 return data,members,groups,day,support,median_abs

def original_base(data):
 x=magnitude_grid(data['X']);return c.features(x,np.zeros(len(x)),np.zeros(len(x)))

def fold_bases(data,cal,base):
 ids=np.array(cal['train_ids']);x=magnitude_grid(data['X'][ids]);aug=c.m.augment(x,np.random.default_rng(20261019));augbase=c.features(aug,np.zeros(len(aug)),np.zeros(len(aug)));return np.concatenate([base[ids],augbase]),aug

def training_inputs(data,median_abs,base,cal,variant):
 if variant not in VARIANTS:raise ValueError('unknown training variant')
 ids=np.array(cal['train_ids']);median=(data['median_feature'] if variant==VARIANTS[0] else median_abs)[ids];anchor=c.e.affine(cal['anchor'],median);a=np.tile(anchor,2);x=assemble(base,a,np.tile(median,2));y=np.tile(data['y'][ids]-anchor,2);w=np.tile(r.weights(data['y'][ids],cal)*.5,2)
 return x,y,w,a

def freeze(create=False):
 from score_routes import check_models
 r.freeze();check_models();p=read(ROOT/'protocol.json')
 if p['variants']!=list(VARIANTS) or p['model']!=read(r.ROOT/'protocol.json')['model'] or p['masked_seed']!=20261019 or not p['all_six_models_locked_before_new_held_scoring']:raise ValueError('fixed ablation design changed')
 for name,h in p['input_sha256'].items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(ROOT.parent) or digest(f)!=h:raise ValueError('magnitude input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ['magnitude_core.py','train_magnitude.py','score_magnitude.py']}}
 target=ROOT/'analysis_lock.json'
 if target.exists() and read(target)!=lock:raise ValueError('magnitude implementation changed after lock')
 if create and not target.exists():write(target,lock)
 return p

def check_models():
 lock=read(ROOT/'model_lock.json')
 if not lock['all_six_models_locked_before_new_held_scoring'] or lock['new_held_scores_created'] or len(lock['files_sha256'])!=12:raise ValueError('partial or selected magnitude models')
 for key,name in [('analysis_lock_sha256','analysis_lock.json'),('training_trace_sha256','training_trace.json')]:
  if lock[key]!=digest(ROOT/name):raise ValueError('magnitude model ancestry differs')
 for name,h in lock['files_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('magnitude model changed: '+name)
 return lock

def summarize(members,y,prediction,assignment,support):
 y=d.vector(y)
 if len(y)!=len(members) or set(prediction)!=set(MODELS):raise ValueError('ablation population/model list differs')
 group=np.array([m['group'] for m in members]);route=np.array([m['route'] for m in members]);window=np.array([m['window_id'] for m in members]);day=np.array([m['day'] for m in members]);speed=d.speed_bins(y)
 if not np.array_equal(assignment,np.array([r.GROUPS.index(v) for v in group])):raise ValueError('ablation held assignment differs')
 p={name:d.vector(prediction[name]) for name in MODELS}
 if any(v.shape!=y.shape for v in p.values()):raise ValueError('ablation prediction alignment differs')
 def cell(mask):
  metric={name:d.metrics(y[mask],v[mask]) for name,v in p.items()}
  return {'count':int(mask.sum()),'models':metric,'delta_metric_from_original':{name:{k:metric[name][k]-metric['original_context'][k] for k in ['mae_mps','rmse_mps','p95_mps','maximum_mps']} for name in VARIANTS}}
 pooled=cell(np.ones(len(y),dtype=bool));bygroup={name:cell(group==name) for name in r.GROUPS};byrecord={name:cell(route==name) for name in sorted(set(route.tolist()))};bywindow={name:cell(window==name) for name in sorted(set(window.tolist()))};byspeed={name:cell(speed==i) for i,name in enumerate(d.SPEED_NAMES)};byday={'day':cell(day),'night':cell(~day)}
 failures={name:{'pooled':pooled['models'][name]['p95_mps']>.5,'groups':[k for k,v in bygroup.items() if v['models'][name]['p95_mps']>.5],'recordings':[k for k,v in byrecord.items() if v['models'][name]['p95_mps']>.5],'windows':[k for k,v in bywindow.items() if v['models'][name]['p95_mps']>.5]} for name in MODELS}
 return {'scope':'Retrospective two-variant magnitude pipeline ablation; prior held outcomes already inspected; no selection/promotion','supported_rows':len(y),'models':list(MODELS),'folds':3,'pooled':pooled,'by_group':bygroup,'by_recording':byrecord,'by_window':bywindow,'by_reference_speed':byspeed,'by_day_night':byday,'diagnostic_failures':failures,'raw_support_costs':support,'all_models_share_each_row':True,'no_quality_filter':True,'new_variant_or_hyperparameter_selection':False,'physical_units_verified':False,'causal_improvement_established':False,'parent_full_recording_development_gate_still_failed':True,'promotion_allowed':False,'reserved_final_opened':False,'witness_enabled':False,'decision':'MAGNITUDE_ABLATION_COMPLETE_NO_PROMOTION'}
