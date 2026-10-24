"""Describe frozen failures without changing or selecting policy settings."""
import collections,json
import final_core as f
import run_temporal as run
from temporal_core import TemporalPolicy
def diagnose():
 result={}
 for role in run.read('protocol.json')['recordings']:
  rows=run.load(role);blocks=collections.defaultdict(list)
  for r in rows:blocks[r['window_id']].append(r)
  events=collections.Counter();errors=[]
  for stream in blocks.values():
   p=TemporalPolicy(run.read('policy_lock.json')['parameters']);last=None
   for r in stream:
    t=r['time_ns'];v=r['valid'];gap=last is not None and t-last>100000000;last=t
    grant=p.step(t,r['gnss_mps'],r['label_mps'],r['prediction_mps'],v)
    events['eligible']+=1;events['denied']+=int(not grant);events['clock_gaps_over_100ms']+=int(gap);events['confirmed_alarm_endpoints']+=int(p.confirmed_now)
    if v:
     errors.append(abs(r['label_mps']-r['prediction_mps']));soft=max(abs(r['gnss_mps']-r['prediction_mps']),abs(r['label_mps']-r['prediction_mps']))>1
     events['physical_discrepancy_over_1mps']+=int(soft)
     if not grant and not soft and not p.confirmed_now:events['denied_while_matching_during_admission_or_recovery']+=1
  result[role]={'events':dict(events),'radar_obd_p95_disagreement_mps':f.percentile(errors),'radar_obd_max_disagreement_mps':max(errors)}
 return result
if __name__=='__main__':
 value=diagnose();(run.ROOT/'diagnostics.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
