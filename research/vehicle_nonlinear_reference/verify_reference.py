"""Separate scalar reductions and exact-record lookup; no model fitting."""
import io,math,unittest
import numpy as np
from reference_core import ROOT,RESEARCH,PROJECT,METHODS,read,load,digest,freeze
def close(a,b):
 if isinstance(b,dict):
  if set(a)!=set(b):raise AssertionError('dictionary membership differs')
  for k in b:close(a[k],b[k])
 elif isinstance(b,list):
  if len(a)!=len(b):raise AssertionError('list differs')
  for x,y in zip(a,b):close(x,y)
 elif isinstance(b,float):
  if not math.isclose(float(a),b,rel_tol=0,abs_tol=1e-10):raise AssertionError((a,b))
 elif a!=b:raise AssertionError((a,b))
def scalar_metrics(y,p):
 errors=[float(b)-float(a) for a,b in zip(y,p)];v=sorted(map(abs,errors));n=len(v)
 if n:
  at=(n-1)*.95;lo=int(at);p95=v[lo]+(v[min(lo+1,n-1)]-v[lo])*(at-lo)
 else:p95=None
 return {'count':n,'mae_mps':math.fsum(v)/n if n else None,'rmse_mps':math.sqrt(math.fsum(x*x for x in v)/n) if n else None,'p95_mps':p95,'maximum_mps':v[-1] if n else None,'bias_mps':math.fsum(errors)/n if n else None,'above_half_mps':sum(x>.5 for x in v),'above_2_mps':sum(x>2 for x in v)}
def scalar_cell(y,pred,obd,ids):
 models={};yy=[y[i] for i in ids];oo=[obd[i] for i in ids]
 for name,p in pred.items():
  pp=[p[i] for i in ids];a=[abs(float(b)-float(v))>.5 for b,v in zip(pp,yy)];b=[abs(float(v)-float(o))>.5 for v,o in zip(pp,oo)]
  models[name]={'primary_reference':scalar_metrics(yy,pp),'obd':scalar_metrics(oo,pp),'above_half_against_both':sum(x and z for x,z in zip(a,b)),'primary_tail_but_obd_le_half':sum(x and not z for x,z in zip(a,b)),'obd_tail_but_primary_le_half':sum(not x and z for x,z in zip(a,b))}
 return {'rows':len(ids),'models':models,'obd_minus_primary':scalar_metrics(yy,oo)}
def verify():
 freeze();result=read(ROOT/'results.json');records=read(RESEARCH/'vehicle_reference_check/matched_endpoints.json');diag=load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz');pairs=load(RESEARCH/'vehicle_frame_reference_audit/reference_pairs.npz');identity=load(ROOT/'row_identity.npz');members=read(RESEARCH/'vehicle_route_validation/membership.json');oof=load(RESEARCH/'vehicle_nonlinear_correction/oof_predictions.npz');new=load(RESEARCH/'vehicle_nonlinear_correction/development_replay.npz');old=load(RESEARCH/'vehicle_signed_axis_correction/development_replay.npz');parent=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');lookup={}
 for j,row in enumerate(records):
  key=(row['window_id'],row['time_ns'])
  if key in lookup or type(row['time_ns']) is not int:raise AssertionError('duplicate/noninteger source key')
  lookup[key]=j
 total_matches=metric_cells=0
 for scope in ['fit','full']:
  q={k.split('__',1)[1]:v for k,v in diag.items() if k.startswith(scope+'__')};n=len(q['row_id']);ids=[];obd=[];present=[]
  for i,(window,t) in enumerate(zip(q['window_id'],q['time_ns'])):
   if q['time_ns'].dtype.kind not in 'iu':raise AssertionError('integer endpoint clock required')
   j=lookup.get((str(window),int(t)),-1);ids.append(j);present.append(j>=0)
   if j<0:obd.append(math.nan);continue
   r=records[j]
   if r['reference_horizontal_mps']!=float(q['target_mps'][i]) or int(t)-r['reference_header_time_ns']!=int(q['reference_age_ns'][i]):raise AssertionError('reference value/age differs')
   obd.append(float(r['label_mps']))
   if not math.isfinite(obd[-1]) or obd[-1]<0:raise AssertionError('invalid OBD')
  np.testing.assert_array_equal(ids,pairs[scope+'__reference_match_id']);np.testing.assert_array_equal(present,pairs[scope+'__comparison_present']);np.testing.assert_allclose(obd,pairs[scope+'__obd_mps'],rtol=0,atol=0,equal_nan=True);np.testing.assert_array_equal(q['row_id'],pairs[scope+'__row_id'])
  if scope=='fit':
   np.testing.assert_array_equal(q['row_id'],np.arange(n));np.testing.assert_array_equal(q['target_mps'],oof['target_mps']);np.testing.assert_array_equal(q['time_ns'],[v['time_ns'] for v in members]);np.testing.assert_array_equal(q['window_id'],[v['window_id'] for v in members]);pred={m:oof[m+'__prediction_mps'] for m in METHODS};accepted=[True]*n
  else:
   pred={m:new[m+'__ready_prediction_mps'] for m in ['nonlinear7','linear7']};pred.update(original_context=old['original_context__ready_prediction_mps'],parent_radar_imu=parent['radar_imu__ready_prediction_mps'],signed6_previous=old['signed_axis_corrected__ready_prediction_mps']);accepted=new['nonlinear7__quality'][q['row_id']].tolist()
  if n!=(27178 if scope=='fit' else 40678) or any(v.shape!=(n,) or not np.isfinite(v).all() for v in pred.values()):raise AssertionError('prediction cohort differs')
  for key,expected in [('row_id',q['row_id']),('time_ns',q['time_ns']),('match_id',ids),('present',present),('accepted',accepted)]:
   if identity[scope+'__'+key].dtype.kind not in ('b' if key in ['present','accepted'] else 'iu'):raise AssertionError('saved identity dtype differs')
   np.testing.assert_array_equal(identity[scope+'__'+key],expected)
  total_matches+=sum(present);categories={'pooled':{'all':list(range(n))},'by_window':{name:[i for i,v in enumerate(q['window_id']) if v==name] for name in sorted(set(q['window_id'].tolist()))}}
  if scope=='fit':
   for cat,key in [('by_group','group'),('by_recording','route')]:categories[cat]={name:[i for i,v in enumerate(members) if v[key]==name] for name in sorted({v[key] for v in members})}
  if set(categories)!=set(result['populations'][scope]):raise AssertionError('result category differs')
  for cat,cells in categories.items():
   if set(cells)!=set(result['populations'][scope][cat]):raise AssertionError('population omitted')
   for name,rows in cells.items():
    matched=[i for i in rows if present[i]];v={'parent_rows':len(rows),'matched_rows':len(matched),'missing_rows':len(rows)-len(matched),'all_matched':scalar_cell(q['target_mps'],pred,obd,matched)};metric_cells+=len(METHODS)*2+1
    if scope=='full':v['nonlinear_accepted_matched']=scalar_cell(q['target_mps'],pred,obd,[i for i in matched if accepted[i]]);metric_cells+=len(METHODS)*2+1
    close(v,result['populations'][scope][cat][name])
 for flag in ['primary_model_changed','new_model_fitted','labels_changed','thresholds_changed','reserved_final_opened','promotion_allowed','reference_threat_independence_established']:
  if result[flag] is not False:raise AssertionError('unsupported claim')
 if result['decision']!='FROZEN_NONLINEAR_ALTERNATE_REFERENCE_CHECK_COMPLETE_NO_PROMOTION' or result['primary_method']!='nonlinear7':raise AssertionError('unexpected decision')
 for n,h in read(ROOT/'sha256.json').items():
  if digest(ROOT/n)!=h:raise AssertionError('delivery artifact changed: '+n)
 prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for n,h in prior.items():
  if n!='FINAL_PROJECT_README.md' and manifest.get(n)!=h:raise AssertionError('prior manifest changed: '+n)
 for n,h in manifest.items():
  if 'urban02' in n.lower() or 'rav4' in n.lower() or n.startswith('research/vehicle_final_evaluation/'):protected+=1;continue
  path=PROJECT/n
  if not path.is_file():missing+=1;continue
  if digest(path)!=h:raise AssertionError('project artifact changed: '+n)
  verified+=1
 import test_reference
 stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_reference))
 if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
 print('ALTERNATE REFERENCE VERIFICATION PASS:',total_matches,'exact matched rows;',metric_cells,'scalar metric cells;',tests.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print(result['decision'])
if __name__=='__main__':verify()
