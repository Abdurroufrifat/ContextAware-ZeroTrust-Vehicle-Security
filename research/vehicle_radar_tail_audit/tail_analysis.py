"""Post-hoc diagnostics only: no model, gate, permit or data exclusion changes."""
from pathlib import Path
import base64,gzip,hashlib,importlib.util,json,math
import numpy as np
import core_io as c
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'vehicle_radar_multidrive'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def joined_rows(index,labels,y,prediction):
 if len(y)!=len(prediction) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(prediction)):raise ValueError('invalid prediction/target arrays')
 keys=[(q['window_id'],q['time_ns']) for q in labels]
 if len(set(keys))!=len(keys):raise ValueError('duplicate source endpoint')
 lookup=dict(zip(keys,labels));rows=[];seen=set()
 for q in index:
  key=q['window_id'],q['time_ns']
  if key in seen or key not in lookup:raise ValueError('duplicate or orphan support endpoint')
  seen.add(key)
  if not q['ready']:continue
  j=q['array_index']
  if j!=len(rows) or j>=len(y) or lookup[key]['reference_horizontal_mps']!=float(y[j]):raise ValueError('misaligned reference/prediction join')
  rows.append({**lookup[key],'array_index':j})
 if len(rows)!=len(y) or len(seen)!=len(labels):raise ValueError('support join loses endpoints')
 return rows
def summary(y,p):
 if not len(y):raise ValueError('empty summary')
 error=np.abs(np.asarray(p)-np.asarray(y));squared=error**2;total=float(squared.sum())
 return {'count':len(error),'mae_mps':float(error.mean()),'rmse_mps':float(np.sqrt(squared.mean())),'p95_mps':float(np.quantile(error,.95)),'maximum_mps':float(error.max()),'tails':{'above_'+str(t)+'_mps':{'count':int((error>t).sum()),'endpoint_fraction':float((error>t).mean()),'squared_error_share':float(squared[error>t].sum()/total) if total else 0.} for t in (.5,1,2,5)}}
def clusters(rows,errors,threshold,gap_ns):
 groups=[]
 for i,row in enumerate(rows):
  if errors[i]<=threshold:continue
  if not groups or row['window_id']!=rows[groups[-1][-1]]['window_id'] or row['time_ns']-rows[groups[-1][-1]]['time_ns']>gap_ns:groups.append([])
  groups[-1].append(i)
 return groups
def radar_window(rows,t):
 good=[r for r in rows if t-1000000000<=r['bag_time_ns']<=t and math.isfinite(r['speed_raw']) and 0<=r['bag_time_ns']-r['header_time_ns']<=250000000]
 if not good:raise ValueError('no valid causal radar sample')
 x=np.asarray([r['speed_raw'] for r in good]);return {'samples':len(good),'first_ns':good[0]['bag_time_ns'],'last_ns':good[-1]['bag_time_ns'],'signed_median_native':float(np.median(x)),'absolute_of_signed_median_native':float(abs(np.median(x))),'low_absolute_return_fraction':float(np.mean(abs(x)<=1.)),'maximum_delivery_ns':max(r['bag_time_ns']-r['header_time_ns'] for r in good)}
def radar_rows(role):
 rows=[]
 with gzip.open(MODEL/'inputs'/role/'packets.jsonl.gz','rt',encoding='utf-8') as file:
  for line in file:
   p=json.loads(line)
   if p['file']=='radars.bag':rows.append(c.decode_message('radar',base64.b64decode(p['data'],validate=True),p['time_ns']))
 rows.sort(key=lambda r:r['bag_time_ns'])
 if any(a['bag_time_ns']>=b['bag_time_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('ambiguous radar clock')
 return rows
def source_reference():
 path=ROOT.parent/'vehicle_reference_check/run_reference.py';spec=importlib.util.spec_from_file_location('_tail_reference_source',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.decode_bag('calibration')[0]['INSPVA']
def compute():
 protocol=read(ROOT/'protocol.json')
 for name,sha in protocol['source_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT.parent) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('frozen diagnostic source changed: '+name)
 result={'posthoc':True,'all_observations_retained':True,'model_refitted':False,'production_policy_changed':False,'witness_enabled':False,'fresh_test_opened':False,'roles':{}}
 cal=None
 for role in ('fit','calibration'):
  with np.load(MODEL/'prepared'/role/'data.npz',allow_pickle=False) as z:data=dict(z)
  with np.load(MODEL/(role+'_predictions.npz'),allow_pickle=False) as z:pred=dict(z)
  rows=joined_rows(read(MODEL/'prepared'/role/'index.json'),read(MODEL/'inputs'/role/'labels.json'),data['y'],pred['causal_residual_tcn']);error=abs(pred['causal_residual_tcn']-data['y']);events=clusters(rows,error,2.,300000000)
  methods={k:summary(data['y'],p) for k,p in pred.items() if '_seed_' not in k};blocks={}
  for name in dict.fromkeys(r['window_id'] for r in rows):
   ids=np.asarray([i for i,r in enumerate(rows) if r['window_id']==name]);blocks[name]=summary(data['y'][ids],pred['causal_residual_tcn'][ids])
  result['roles'][role]={'supported_endpoints':len(rows),'methods':methods,'primary_blocks':blocks,'tail_events_above_2_mps':[{'window_id':rows[g[0]]['window_id'],'start_ns':rows[g[0]]['time_ns'],'end_ns':rows[g[-1]]['time_ns'],'samples':len(g),'max_error_mps':float(error[g].max())} for g in events]}
  if role=='calibration':cal=data,pred,rows,error
 data,pred,rows,error=cal;j=int(np.argmax(error));worst=rows[j];t=worst['time_ns'];raw=radar_rows('calibration');statistics=radar_window(raw,t)
 if not math.isclose(statistics['absolute_of_signed_median_native'],float(data['median_feature'][j]),abs_tol=1e-10):raise ValueError('raw median is not the saved anchor feature')
 reference=source_reference();import reference_core as ref
 reference_index=ref.ReferenceIndex(reference);matched=reference_index.latest(t,100000000)
 if matched is None or not math.isclose(matched['horizontal_mps'],float(data['y'][j]),abs_tol=1e-10):raise ValueError('publisher-verified reference does not match worst endpoint')
 anchor=float(pred['reference_median_affine'][j]);result['worst_calibration_endpoint']={'window_id':worst['window_id'],'time_ns':t,'array_index':j,'reference_mps':float(data['y'][j]),'obd_mps':worst['label_mps'],'gnss_associated_mps':worst['gnss_mps'],'reference_age_ns':t-matched['header_time_ns'],'reference_status':matched['status'],'reference_time_status':matched['time_status'],'raw_causal_history':statistics,'grid_last_1s_abs_signed_median_native':float(abs(np.median(data['X'][j,0,-20:].astype(float)*10))),'model_predictions_mps':{k:float(p[j]) for k,p in pred.items() if '_seed_' not in k},'tcn_anchor_mps':anchor,'tcn_learned_residual_mps':float(pred['causal_residual_tcn'][j]-anchor),'paper_cnn_is_exact_reproduction':False}
 radius=protocol['context_radius_ns'];ids=[i for i,r in enumerate(rows) if r['window_id']==worst['window_id'] and abs(r['time_ns']-t)<=radius]
 context={'center_ns':t,'retrospective_context_contains_future_samples':True,'not_used_as_online_model_input':True,'radar_records':[r for r in raw if abs(r['bag_time_ns']-t)<=radius],'reference_records':[{'time_ns':r['header_time_ns'],'horizontal_mps':r['horizontal_mps'],'status':r['status']} for r in reference if abs(r['header_time_ns']-t)<=radius],'endpoints':[{'time_ns':rows[i]['time_ns'],'reference_mps':float(data['y'][i]),'obd_mps':rows[i]['label_mps'],'gnss_mps':rows[i]['gnss_mps'],'predictions_mps':{k:float(p[i]) for k,p in pred.items() if '_seed_' not in k}} for i in ids]}
 return result,context
if __name__=='__main__':
 result,context=compute();write(ROOT/'results.json',result);write(ROOT/'worst_event_context.json',context)
 q=result['roles']['calibration']['methods']['causal_residual_tcn'];print('TAIL AUDIT:',q['tails']['above_2_mps'],'maximum',q['maximum_mps'],flush=True)
