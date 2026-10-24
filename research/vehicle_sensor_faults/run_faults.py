from fault_core import *
def run():
 freeze()
 if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('already attempted; no result-driven retry')
 data=inputs();(ROOT/'replays').mkdir();results={}
 for case in CASES:
  result,arrays=simulate(data,case);results[case]=result;write_replay(ROOT/'replays'/(case+'.npz'),arrays)
  for method,v in result.items():
   d=v['requests']['during'];print(case,method,d['accepted_requests'],d['availability'],d['accepted_errors']['p95_mps'],flush=True)
 write(ROOT/'results.json',{'decision':'FROZEN_MODEL_SENSOR_FAULT_REPLAY_COMPLETE_NO_PROMOTION','scope':'Recovered synthetic decoded-source faults on reused Urban01 NIGHT development; prior outcomes were known during reconstruction','fault_interval_ns':[data['start'],data['end']],'cases':results,'thresholds_changed':False,'model_retrained':False,'reserved_final_opened':False,'production_policy_changed':False,'promotion_allowed':False,'physical_security_validated':False})
if __name__=='__main__':run()
