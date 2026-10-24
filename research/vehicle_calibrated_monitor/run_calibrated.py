"""One locked fitting-only calibration followed by one retrospective replay."""
import csv,io
import numpy as np
from calibrated_core import *
from scalar_metrics import close

def fixtures():
 i=np.arange(61);t=i.astype(np.int64)*50_000_000;base=np.full(61,10.)
 rows={'abrupt_return':(t,base+np.where((i>=10)&(i<30),3.,0.),base),'persistent_step':(t,base+np.where(i>=10,3.,0.),base),'slow_bias':(t,base+np.minimum(np.maximum(i-9,0)*.1,3.),base),'startup_bias':(t,base+3.,base),'gap_hidden_bias':(np.array([0,100_000_001],np.int64),np.array([10.,13.]),np.array([10.,10.])),'benign_fast_change':(t,base+np.where((i>=10)&(i<30),3.,0.),base+np.where((i>=10)&(i<30),3.,0.)),'anchor_collision':(t,base+np.where((i>=10)&(i<20),3.,0.),base-np.where(i>=20,3.,0.))}
 results={}
 for name,(t,p,y) in rows.items():
  results[name]={}
  for kind in ('latch','return3'):
   v=bank(t,p,np.ones(len(t),bool),t.copy(),1.,kind);error=abs(p[v['allowed']]-y[v['allowed']]);results[name][kind]={'samples':len(t),'allowed':v['allowed'].tolist(),'reason':v['reason'].tolist(),'alarm_rows':np.flatnonzero(v['alarm']).tolist(),'recovery_rows':np.flatnonzero(v['recovery']).tolist(),'accepted':int(v['allowed'].sum()),'accepted_error_above_2':int(np.sum(error>2.)),'maximum_accepted_error_mps':float(error.max()) if len(error) else None}
 return results

def diagnostics(bank,source,t,masks):
 alarms=np.flatnonzero(bank['alarm']);recoveries=np.flatnonzero(bank['recovery'])
 return {'alarm_rows':alarms.tolist(),'recovery_rows':recoveries.tolist(),'first_alarm_time_ns':int(t[alarms[0]]) if len(alarms) else None,'by_population':{name:{'eligible':int(np.sum(mask&source['eligible'])),'allowed':int(np.sum(mask&bank['allowed'])),'alarms':int(np.sum(mask&bank['alarm'])),'recoveries':int(np.sum(mask&bank['recovery'])),'gap_reanchors':int(np.sum(mask&(bank['reason']==6)))} for name,mask in masks.items()}}

def generate():
 freeze();locked=read(ROOT/'calibration_lock.json')
 if locked!={'parameters_sha256':digest(ROOT/'parameters.json'),'pairs_sha256':digest(ROOT/'calibration_pairs.npz'),'analysis_lock_sha256':digest(ROOT/'analysis_lock.json')}:raise ValueError('calibration changed')
 if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('replay already attempted; no outcome search')
 t,y,events,masks=data();parameters=read(ROOT/'parameters.json');previous=read(RESEARCH/'vehicle_source_compromise/results.json');results={};diags={};csv_rows=[];(ROOT/'replays').mkdir()
 for case in CASES:
  old=load(RESEARCH/'vehicle_source_compromise/replays'/(case+'.npz'));sources=branch_inputs(old,t,events);banks={kind:{source:bank(t,v['prediction'],v['eligible'],v['endpoint_ns'],parameters['source_limits_mps'][source],kind) for source,v in sources.items()} for kind in ('latch','return3')};saved={};results[case]={};diags[case]={kind:{s:diagnostics(v,sources[s],t,masks) for s,v in group.items()} for kind,group in banks.items()}
  for kind,group in banks.items():
   for s,values in group.items():
    for key,value in values.items():saved[kind+'__'+s+'__'+key]=value
  for method in METHODS:
   baseline=select_output(sources,method)
   for key in FIELDS:np.testing.assert_allclose(baseline[key],old[method+'__'+key],rtol=0,atol=0,equal_nan=True)
   cell={'baseline':summary(baseline,t,y,masks)};close(cell['baseline'],previous['cases'][case][method])
   for kind,group in banks.items():
    out=select_output(sources,method,group);cell[kind]=summary(out,t,y,masks);saved[kind+'__'+method+'__route']=out['route']
   results[case][method]=cell
   for variant,summary_rows in cell.items():
    for pop,v in summary_rows.items():csv_rows.append({'case':case,'method':method,'variant':variant,'population':pop,'requests':v['requests'],'accepted_requests':v['accepted_requests'],'availability':v['availability'],'p95_mps':v['accepted_errors']['p95_mps'],'above_2_mps':v['accepted_errors']['above_2_mps'],'joint_gate_pass':v['joint_gate_pass']})
  write_npz(ROOT/'replays'/(case+'.npz'),saved)
  print(case,{kind:{'accepted':results[case]['extra_trees7__deadband__fallback'][kind]['during']['accepted_requests'],'p95':results[case]['extra_trees7__deadband__fallback'][kind]['during']['accepted_errors']['p95_mps'],'above2':results[case]['extra_trees7__deadband__fallback'][kind]['during']['accepted_errors']['above_2_mps']} for kind in VARIANTS},flush=True)
 buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
 from replay_metrics import atomic
 atomic(ROOT/'comparison.csv',buf.getvalue().encode())
 flags={k:False for k in ('models_refitted','labels_changed','accuracy_or_availability_thresholds_changed','production_policy_changed','reserved_final_opened','promotion_allowed','independent_source_trust_established','recovery_certified_safe')}
 write(ROOT/'results.json',{'decision':'SOURCE_CALIBRATED_MONITOR_COMPARISON_COMPLETE_NO_PROMOTION','cases':results,'source_diagnostics':diags,'synthetic_counterexamples':fixtures(),**flags})
 print('SOURCE_CALIBRATED_MONITOR_COMPARISON_COMPLETE_NO_PROMOTION',flush=True)
if __name__=='__main__':generate()
