"""Frozen-output retrospective diagnostics. Never fits or changes a policy."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'vehicle_signed_axis_correction'));import axis_core as a
read=a.read;write=a.write;digest=a.digest;METHODS=a.METHODS;PRIMARY='signed_axis_corrected';BASE='uncorrected_radar_imu'
def metrics(y,p):
 r=a.m.d.metrics(y,p);e=np.asarray(p)-np.asarray(y);tail=abs(e)>.5;r['tail_overprediction_fraction']=float(np.mean(e[tail]>0)) if tail.any() else None
 return r
def stats(x):
 x=np.asarray(x);v=x[np.isfinite(x)]
 return {'total':len(x),'finite':len(v),'min':float(v.min()) if len(v) else None,'median':float(np.median(v)) if len(v) else None,'p95':float(np.quantile(v,.95)) if len(v) else None,'max':float(v.max()) if len(v) else None}
def bucket_ids(t,origin):
 t=a.m.d.timestamps(t)
 if type(origin) is not int or origin>int(t[0]):raise ValueError('invalid integer time origin')
 return (t-origin)//20_000_000_000
def runs(t,e):return a.m.d.error_runs(t,e,150_000_000)
def support_flags(training,query):
 x=np.asarray(training,dtype=float);q=np.asarray(query,dtype=float)
 if x.ndim!=2 or not len(x) or q.ndim!=2 or q.shape[1]!=x.shape[1] or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('invalid training support arrays')
 return (q<x.min(0))|(q>x.max(0))
def head_terms(h,x):
 x=np.asarray(x,dtype=float);terms=((x-np.array(h['mean']))/np.array(h['scale']))*np.array(h['coefficient']);return terms,float(h['intercept'])+terms.sum(1)
def imu_ages(source,times):
 q=a.m.im.lookup(source,times);return {'imu_status':q['status'],'imu_header_age_ns':q['header_age_ns'],'imu_arrival_age_ns':q['arrival_age_ns']}
def freeze(create=False):
 if not create and not (ROOT/'analysis_lock.json').is_file():raise ValueError('diagnosis must be frozen')
 a.check_heads();p=read(ROOT/'protocol.json')
 if p['speed_boundaries_mps']!=[1.,5.,10.] or p['time_bucket_ns']!=20_000_000_000 or p['run_gap_ns']!=150_000_000 or p['diagnostic_limit_mps']!=.5 or p['models']!=list(METHODS) or p['new_model_fitted'] or p['thresholds_changed'] or p['promotion_allowed']:raise ValueError('diagnostic prespecification differs')
 for name,h in p['input_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(RESEARCH) or digest(path)!=h:raise ValueError('diagnostic input changed '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['remaining_core.py','generate_remaining.py']}};file=ROOT/'analysis_lock.json'
 if file.exists() and read(file)!=lock:raise ValueError('frozen diagnostic code changed')
 if create and not file.exists():write(file,lock)
def load_rows():
 freeze();data,members,groups,day,support=a.m.r.load_data();index,ready,x,median,anchor,peer,source,old=a.m.f.load_full();axis=a.cache();imu=a.m.cache();fitn=len(members)
 with np.load(a.ROOT/'oof_predictions.npz',allow_pickle=False) as z:fit={'row_id':np.arange(fitn,dtype=np.int64),'time_ns':np.array([v['time_ns'] for v in members],dtype=np.int64),'window_id':np.array([v['window_id'] for v in members]),'target_mps':z['target_mps'].copy(),**{n+'__prediction_mps':z[n+'__prediction_mps'].copy() for n in METHODS},**{n+'__accepted':np.ones(fitn,bool) for n in METHODS}}
 with np.load(a.ROOT/'development_replay.npz',allow_pickle=False) as z:full={'row_id':np.flatnonzero(ready),'time_ns':np.array([v['time_ns'] for v in index],dtype=np.int64)[ready],**{n+'__prediction_mps':z[n+'__ready_prediction_mps'].copy() for n in METHODS},**{n+'__accepted':z[n+'__quality'][ready].copy() for n in METHODS}}
 reference=read(RESEARCH/'vehicle_continuous_radar/reference_targets.json');full['target_mps']=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in reference])[ready];full['reference_age_ns']=full['time_ns']-np.array([-1 if v['reference_header_ns'] is None else v['reference_header_ns'] for v in reference],dtype=np.int64)[ready]
 with np.load(RESEARCH/'vehicle_trace_audit/row_traces.npz',allow_pickle=False) as z:fit['reference_age_ns']=z['reference_age_ns'].copy();np.testing.assert_array_equal(z['row_id'],fit['row_id'])
 np.testing.assert_array_equal(fit['target_mps'],data['y'])
 if not np.isfinite(full['target_mps']).all():raise ValueError('fixed ready cohort reference missing')
 labels=np.full(len(index),'outside_supported_blocks',dtype='<U32');lookup={v['time_ns']:i for i,v in enumerate(index)}
 for v in read(a.m.c.OLD/'prepared/calibration/index.json'):
  if v['ready']:
   j=lookup[v['time_ns']]
   if labels[j]!='outside_supported_blocks':raise ValueError('overlapping old blocks')
   labels[j]=v['window_id']
 full['window_id']=labels[ready]
 for rows,tensors,axes,cov in [(fit,data['X'],axis['fit_axes'],axis['fit_coverage']),(full,x,axis['full_axes'][ready],axis['full_coverage'][ready])]:
  rows['signed_axes']=axes;rows['imu_valid_tick_fraction']=cov;valid=tensors[:,3,:]==1;count=valid.sum(1);rows['radar_valid_fraction']=count/128;rows['radar_mean_recorded_age_ms']=np.where(valid,tensors[:,2,:],0.).sum(1)/np.maximum(count,1)*300.;rows['radar_mixed_sign_grid']=np.any(valid&(tensors[:,0,:]>0),axis=1)&np.any(valid&(tensors[:,0,:]<0),axis=1)
  rows['outside_training_axis']=np.zeros((len(axes),6),bool);rows['axis_standardized_abs_max']=np.zeros(len(axes));rows['head_terms']=np.zeros_like(axes);rows['raw_head_delta_mps']=np.zeros(len(axes));covered=np.zeros(len(axes),int)
  for scope in (a.m.r.GROUPS if rows is fit else ['all_fit']):
   cal=read(a.m.ROOT/'models'/scope/'calibration.json');train=np.array(cal['train_ids']);ids=np.array(cal['held_ids']) if rows is fit else np.arange(len(axes));head=read(a.ROOT/'heads'/scope/'head.json');rows['outside_training_axis'][ids]=support_flags(axis['fit_axes'][train],axes[ids]);rows['axis_standardized_abs_max'][ids]=np.max(abs((axes[ids]-np.array(head['mean']))/np.array(head['scale'])),axis=1);terms,delta=head_terms(head,axes[ids]);rows['head_terms'][ids]=terms;rows['raw_head_delta_mps'][ids]=delta;covered[ids]+=1
  if not np.all(covered==1):raise ValueError('diagnostic scope assignment differs')
  rows['applied_delta_mps']=rows[PRIMARY+'__prediction_mps']-rows[BASE+'__prediction_mps'];np.testing.assert_allclose(np.maximum(0.,rows[BASE+'__prediction_mps']+rows['raw_head_delta_mps']),rows[PRIMARY+'__prediction_mps'],rtol=0,atol=1e-9)
 for key in ['imu_status','imu_header_age_ns','imu_arrival_age_ns']:fit[key]=np.zeros(fitn,dtype=np.int64)
 for key,route in a.m.sa.l.ROUTES.items():
  if key=='Urban01_night':continue
  _,s=a.m.sa.load(key);ids=np.array([i for i,v in enumerate(members) if v['route']==route]);q=imu_ages(s,fit['time_ns'][ids])
  for name,value in q.items():fit[name][ids]=value
 full['imu_status']=imu['full_current__status'][ready];full['imu_header_age_ns']=full['time_ns']-imu['full_current__header_ns'][ready];full['imu_arrival_age_ns']=full['time_ns']-imu['full_current__bag_ns'][ready]
 return {'fit':fit,'full':full}
COVARIATES=['reference_age_ns','imu_header_age_ns','imu_arrival_age_ns','imu_valid_tick_fraction','radar_valid_fraction','radar_mean_recorded_age_ms','axis_standardized_abs_max','raw_head_delta_mps','applied_delta_mps']
def cell(rows,ids,details=False):
 y=rows['target_mps'][ids];p={n:rows[n+'__prediction_mps'][ids] for n in METHODS};bad=abs(p[PRIMARY]-y)>.5;basebad=abs(p[BASE]-y)>.5
 r={'count':len(ids),'models':{n:metrics(y,v) for n,v in p.items()},'accepted_counts_on_same_rows':{n:int(rows[n+'__accepted'][ids].sum()) for n in METHODS},'base_corrected_tail_overlap':{'both':int((bad&basebad).sum()),'base_only':int((basebad&~bad).sum()),'corrected_only':int((bad&~basebad).sum()),'neither':int((~basebad&~bad).sum())}}
 if details:
  r['tail_vs_other_covariates']={name:{'tail':stats(rows[name][ids][bad]),'other':stats(rows[name][ids][~bad])} for name in COVARIATES}
  for label,flag in [('tail',bad),('other',~bad)]:
   which=ids[flag];r[label+'_support']={'rows':len(which),'outside_any_training_axis':int(rows['outside_training_axis'][which].any(1).sum()),'outside_training_axis_counts':rows['outside_training_axis'][which].sum(0).tolist(),'mixed_sign_radar_grid':int(rows['radar_mixed_sign_grid'][which].sum()),'invalid_latest_imu':int((rows['imu_status'][which]!=0).sum()),'reference_age_above100ms':int((rows['reference_age_ns'][which]>100_000_000).sum()),'negative_reference_age':int((rows['reference_age_ns'][which]<0).sum()),'head_term_stats':{name:stats(rows['head_terms'][which,j]) for j,name in enumerate(a.FEATURE_NAMES)}}
 return r
def build(rows):
 result={'scope':'Exploratory localization of frozen errors; all fixed windows/blocks retained; no new predictions','populations':{},'associations_are_not_causes':True,'root_cause_established':False,'reference_independence_established':False,'new_model_fitted':False,'thresholds_changed':False,'prediction_or_timestamp_changed':False,'reserved_final_opened':False,'promotion_allowed':False,'witness_enabled':False,'decision':'REMAINING_FAILURE_DIAGNOSIS_COMPLETE_NO_PROMOTION'};records=[]
 for scope,q in rows.items():
  labels=q['window_id'];names=sorted(set(labels.tolist()));windows={}
  for name in names:
   ids=np.flatnonzero(labels==name);ids=ids[np.argsort(q['time_ns'][ids],kind='stable')];selected=ids[q[PRIMARY+'__accepted'][ids]];common=ids[np.logical_and.reduce([q[n+'__accepted'][ids] for n in METHODS])];buckets=bucket_ids(q['time_ns'][ids],int(q['time_ns'][ids[0]]));sb=np.searchsorted([1.,5.,10.],q['target_mps'][selected],side='right');ss=bucket_ids(q['time_ns'][selected],int(q['time_ns'][ids[0]])) if len(selected) else np.array([],dtype=np.int64)
   windows[name]={'all_supported_or_ready_rows':cell(q,ids),'corrected_accepted_rows':cell(q,selected,True),'all_three_common_accepted_rows':cell(q,common),'by_reference_speed':{key:cell(q,selected[sb==i]) for i,key in enumerate(a.m.d.SPEED_NAMES)},'by_elapsed_20s':{str(int(k)):cell(q,selected[ss==k]) for k in np.unique(buckets)}}
   if len(selected):
    for run in runs(q['time_ns'][selected],abs(q[PRIMARY+'__prediction_mps'][selected]-q['target_mps'][selected])):
     records.append({'population':scope,'window_id':name,'first_row_id':int(q['row_id'][selected[run['first_local_index']]]),'last_row_id':int(q['row_id'][selected[run['last_local_index']]]),**run})
  selected=np.flatnonzero(q[PRIMARY+'__accepted']);result['populations'][scope]={'rows':len(labels),'corrected_accepted':len(selected),'all_rows':cell(q,np.arange(len(labels))),'windows':windows,'reference_age_max_ns':int(q['reference_age_ns'].max()),'imu_header_age_max_ns':int(q['imu_header_age_ns'].max()),'imu_arrival_age_max_ns':int(q['imu_arrival_age_ns'].max())}
 result['remaining_failed_fit_windows']=[n for n,v in result['populations']['fit']['windows'].items() if v['corrected_accepted_rows']['models'][PRIMARY]['p95_mps']>.5];result['remaining_failed_old_development_blocks']=[n for n,v in result['populations']['full']['windows'].items() if n!='outside_supported_blocks' and v['corrected_accepted_rows']['models'][PRIMARY]['p95_mps']>.5]
 return result,records
