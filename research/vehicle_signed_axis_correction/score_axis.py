from axis_core import *
def compute():
 check_heads();axes=cache();data,members,groups,day,support=m.r.load_data()
 with np.load(m.ROOT/'oof_predictions.npz',allow_pickle=False) as z:y=z['target_mps'].copy();old=z['original_context__prediction_mps'].copy();base=z['radar_imu__prediction_mps'].copy();fold=z['fold_id'].copy()
 np.testing.assert_array_equal(y,data['y']);corrected=np.full(len(y),np.nan);covered=np.zeros(len(y),int)
 for scope in m.r.GROUPS:
  cal=read(m.ROOT/'models'/scope/'calibration.json');ids=np.array(cal['held_ids']);head=read(ROOT/'heads'/scope/'head.json');corrected[ids]=np.maximum(0.,base[ids]+head_predict(head,axes['fit_axes'][ids]));covered[ids]+=1
 if not np.all(covered==1):raise ValueError('correction held membership differs')
 # Parent summaries expose three numerical slots; only their output names are remapped.
 route=rename(m.route_summary(members,y,{'original_context':old,'radar_imu':base,'imu_only':corrected},support));index,ready,*_=m.f.load_full()
 with np.load(m.ROOT/'development_replay.npz',allow_pickle=False) as z:fullbase=z['radar_imu__ready_prediction_mps'].copy()
 fullcorrected=np.maximum(0.,fullbase+head_predict(read(ROOT/'heads/all_fit/head.json'),axes['full_axes'][ready]));slots,rawarrays=m.full_summary({'radar_imu':fullbase,'imu_only':fullcorrected},m.cache());full=rename(slots);output={}
 for name,value in rawarrays.items():
  prefix,suffix=name.split('__',1);output[SLOT_NAMES[prefix]+'__'+suffix]=value
 # Direct paired cohort for the new correction, alongside original-control pairs.
 times=output['signed_axis_corrected__request__request_times_ns'];segment=(times-times[0])//300_000_000_000
 truth=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in read(RESEARCH/'vehicle_availability_audit/request_reference.json')]);pair=['uncorrected_radar_imu','signed_axis_corrected'];common=(output[pair[0]+'__request__status']==0)&(output[pair[1]+'__request__status']==0)
 def paired(mask):return {'shared_requests':int(mask.sum()),'models':{n:m.av.scores(truth,output[n+'__request__prediction_mps'],mask) for n in pair}}
 full['correction_common_accepted_requests']={'overall':paired(common),'segments_300s':{str(int(k)):paired(common&(segment==k)) for k in np.unique(segment)}}
 result={'scope':'One prespecified six-axis training-residual ridge correction; retrospective routes and inspected development','route':route,'full_development':full,'parent_models_unchanged':True,'all_four_heads_locked_before_new_scoring':True,'full_development_reused':True,'thresholds_changed':False,'promotion_allowed':False,'production_policy_changed':False,'reserved_final_opened':False,'witness_enabled':False,'journal_readiness_established':False,'decision':'SIGNED_AXIS_CORRECTION_COMPLETE_NO_PROMOTION'}
 return result,{'target_mps':y,'fold_id':fold,**{name+'__prediction_mps':v for name,v in zip(METHODS,[old,base,corrected])}},output
def score():
 if any((ROOT/n).exists() for n in ['results.json','oof_predictions.npz','development_replay.npz']):raise ValueError('correction outcomes already scored; no retry')
 result,route,full=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'oof_predictions.npz',**route);np.savez_compressed(ROOT/'development_replay.npz',**full)
 for n in METHODS:print('ROUTE',n,result['route']['pooled']['models'][n]['p95_mps'],'failed windows',len(result['route']['diagnostic_failures'][n]['windows']),flush=True)
 for n,v in result['full_development']['methods'].items():print('FULL',n,'P95',v['requests']['overall']['accepted']['p95_mps'],'gates',result['full_development']['all_gates_pass'][n],flush=True)
if __name__=='__main__':score()
