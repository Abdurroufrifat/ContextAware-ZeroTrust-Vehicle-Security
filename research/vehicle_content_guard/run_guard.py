from guard_core import *
def run():
 freeze()
 if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('guard experiment already attempted')
 data=f.inputs();baseline=read(RESEARCH/'vehicle_sensor_faults/results.json');(ROOT/'replays').mkdir();guarded={};gates={'no_prior_denial_revived':True}
 for case in f.CASES:
  old=load(RESEARCH/'vehicle_sensor_faults/replays'/(case+'.npz'));values,arrays=summarize(data,old,case);guarded[case]=values;f.write_replay(ROOT/'replays'/(case+'.npz'),arrays);v=values['nonlinear7']['requests']['during'];print(case,'candidate accepted',v['accepted_requests'],'of',v['requests'],'P95',v['accepted_errors']['p95_mps'],'additional total withheld',values['nonlinear7']['additional_withheld_requests'],flush=True)
  if case=='nominal':gates['no_additional_nominal_request_denials']=all(v['additional_withheld_requests']==0 for v in values.values())
  if case=='frozen_values':
   mask=(data['times']>=data['start']+BUDGET)&(data['times']<data['end']);gates['zero_multisource_grants_in_sustained_frozen_interval']=all(not np.any(mask&(arrays[m+'__request__status']==0)) for m in ['radar_imu_parent','nonlinear7'])
 sources=source_checks();gates['no_extra_content_denials_in_six_captured_sources']=all(v['unchanged_content_withheld']==0 for v in sources.values());result={'decision':'OFFLINE_CONTENT_FRESHNESS_GUARD_COMPLETE_NO_PROMOTION','scope':'Post-failure fixed heuristic serving overlay on existing development replays; no physical attestation claim','baseline':baseline['cases'],'guarded':guarded,'sources':sources,'controls':controls(),'observed_contracts':gates,'models_retrained':False,'prediction_values_changed':False,'thresholds_tuned':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False,'physical_security_validated':False};write(ROOT/'results.json',result);print('GUARD CONTRACTS',gates,flush=True)
if __name__=='__main__':run()
