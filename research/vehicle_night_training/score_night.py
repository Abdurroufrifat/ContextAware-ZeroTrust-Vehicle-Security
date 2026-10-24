"""One unchanged development replay after both model conditions are frozen."""
from night_core import *
from score_masked import summarize as prior_summarize
from availability_core import scores
MAP={'day_only':'clean_control','day_plus_night':'masked_training'}
def check_models():
 lock=read(ROOT/'model_lock.json')
 for key,file in [('analysis_lock_sha256','analysis_lock.json'),('data_lock_sha256','data_lock.json'),('training_trace_sha256','training_trace.json')]:
  if lock[key]!=digest(ROOT/file):raise ValueError('training lock changed')
 if len(lock['models_sha256'])!=12:raise ValueError('incomplete matched conditions')
 for name,h in lock['models_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('model changed')
 return lock
def predict():
 check_models()
 with np.load(MASK/'prepared.npz',allow_pickle=False) as z:x=z['X'].copy();median=z['median_feature'].copy()
 anchor=e.affine(read(OLD/'classical_models.json')['reference_median_affine'],median);out={'anchor':anchor}
 for condition in CONDITIONS:
  for kind in KINDS:
   models=[read(ROOT/f'models/{condition}/{kind}_{seed}.json') for seed in SEEDS];mean,vectors=m.ensemble(models,x,anchor);key=condition+'__'+kind;out[key]=mean
   for seed,v in zip(SEEDS,vectors):out[key+'__'+str(seed)]=v
   print('NIGHT REPLAY',condition,kind,len(mean),'endpoints',flush=True)
 return out
def summarize(pred):
 remap={'anchor':pred['anchor']}
 for condition in CONDITIONS:
  for kind in KINDS:remap[MAP[condition]+'__'+kind]=pred[condition+'__'+kind]
 result,arrays=prior_summarize(read(MASK/'frame_index.json'),remap)
 result['scope']='Night training coverage comparison on the same inspected Urban01 night development recording; not final validation'
 result['conditions']={condition:result['conditions'][MAP[condition]] for condition in CONDITIONS}
 arrays={next((k.replace(old,new,1) for new,old in MAP.items() if k.startswith(old+'__')),k):v for k,v in arrays.items()}
 y=np.array([np.nan if p['reference_mps'] is None else p['reference_mps'] for p in read(ROOT.parent/'vehicle_availability_audit/request_reference.json')]);both=(arrays['day_only__request__status']==0)&(arrays['day_plus_night__request__status']==0);segment=(arrays['day_only__request__request_times_ns']-arrays['day_only__request__request_times_ns'][0])//300000000000
 def compare(ids):return {'shared_accepted_requests':int(ids.sum()),**{condition:scores(y,arrays[condition+'__request__prediction_mps'],ids) for condition in CONDITIONS}}
 result['same_accepted_cohort']={'overall':compare(both),'segments_300s':{str(int(k)):compare(both&(segment==k)) for k in np.unique(segment)}}
 result['primary']='day_plus_night';result['decision']='NIGHT_DEVELOPMENT_GATES_PASS_NO_PROMOTION' if result['all_gates_pass'] else 'STOP_AFTER_NIGHT_DEVELOPMENT_FAILURE';result['training_counts']=read(ROOT/'data_report.json')['samples'];result['reserved_final_opened']=False;result['witness_enabled']=False;result['journal_readiness_established']=False
 return result,arrays
def score():
 freeze();check_models()
 if (ROOT/'results.json').exists():raise ValueError('development result already exists; no retries')
 pred=predict();np.savez_compressed(ROOT/'predictions.npz',**pred);result,arrays=summarize(pred);np.savez_compressed(ROOT/'decisions.npz',**arrays);write(ROOT/'results.json',result);print(result['decision'],result['gates'],flush=True)
if __name__=='__main__':score()
