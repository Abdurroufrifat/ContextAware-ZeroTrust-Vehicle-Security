from deadband_core import *
def run():
 parameters=check_parameters();scale=np.array(parameters['scale']);data=f.inputs()
 if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('already scored; no retry')
 (ROOT/'replays').mkdir();cases={}
 for case in CASES:
  source=attack_source(data,case,scale)
  if case in f.CASES:base=load(RESEARCH/'vehicle_sensor_faults/replays'/(case+'.npz'))
  else:base=predict_source(data,source);f.write_replay(ROOT/'replays'/(case+'__base.npz'),base)
  cases[case]={}
  for policy in POLICIES:
   result,arrays=overlay(data,base,source,policy,scale);cases[case][policy]=result
   if policy=='deadband' or (policy=='exact' and case not in f.CASES):f.write_replay(ROOT/'replays'/(case+'__'+policy+'.npz'),arrays)
   if policy=='exact' and case in f.CASES:
    previous=load(RESEARCH/'vehicle_content_guard/replays'/(case+'.npz'))
    for key,v in arrays.items():np.testing.assert_allclose(v,previous[key],rtol=0,atol=0,equal_nan=True)
   v=result['nonlinear7']['requests']['during'];print(case,policy,'candidate',v['accepted_requests'],'of',v['requests'],'P95',v['accepted_errors']['p95_mps'],'extra withheld',result['nonlinear7']['additional_withheld_requests'],flush=True)
 result={'decision':'FIXED_DEADBAND_CONTENT_GUARD_COMPARISON_COMPLETE_NO_PROMOTION','scope':'Post-failure fitting-scaled heuristic; reused development, no physical attestation or novel-algorithm claim','cases':cases,'source_checks':source_checks(scale),'controls':controls(scale),'models_retrained':False,'labels_changed':False,'threshold_search':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False};write(ROOT/'results.json',result);print(result['decision'],flush=True)
if __name__=='__main__':run()
