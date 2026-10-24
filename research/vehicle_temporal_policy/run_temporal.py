"""One fixed calibration, prospective replay, and explicit failed-gate reporting."""
from pathlib import Path
import collections,hashlib,json,sys
import final_core as f
from temporal_core import TemporalPolicy,calibrate
ROOT=Path(__file__).resolve().parent
CODE=('core_io.py','final_core.py','temporal_core.py','data_stream.py','acquire_continuous.py','run_temporal.py')
def read(path):return json.loads((ROOT/path).read_text())
def load(role):return read('roles/'+role+'/stream.json')
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def freeze(write=True):
 lock={'parameters':calibrate(load('fit'),load('calibration')),'protocol_sha256':digest('protocol.json'),'radar_model_sha256':digest('radar_model_lock.json'),'code_sha256':{n:digest(n) for n in CODE},'development_data_sha256':{role:digest('roles/'+role+'/stream.json') for role in ('fit','calibration')}}
 path=ROOT/'policy_lock.json'
 if path.exists() and read('policy_lock.json')!=lock:raise ValueError('frozen policy changed')
 if write and not path.exists():path.write_text(json.dumps(lock,indent=2)+'\n')
 return lock
def replay(rows,params):
 protocol=read('protocol.json');trials=[];aggregate=collections.defaultdict(collections.Counter);blocks=collections.defaultdict(list)
 for row in rows:blocks[row['window_id']].append(row)
 for window,stream in blocks.items():
  first=stream[0]['time_ns'];span=stream[-1]['time_ns']-first;onset=first+int(.4*span);offset=first+int(.8*span)
  for attack in ['normal']+protocol['attacks']:
   states={n:TemporalPolicy(params) if n=='confirmed_cusum_recovery' else f.ReplayPolicy(n) for n in protocol['policies']};counts={n:collections.Counter() for n in states};first_grants={n:None for n in states}
   for row in stream:
    t=row['time_ns'];valid=row['valid'];g=row['gnss_mps'];w=row['label_mps'];p=row['prediction_mps'];active=False;shift=0.
    if attack=='startup_common_bias_7':active=t<=offset
    elif attack!='normal':active=onset<=t<=offset
    if active:
     if attack in ('startup_common_bias_7','delayed_common_step_7','wheel_only_step_7'):shift=7.
     elif attack=='common_ramp_075':shift=.75*(t-onset)/1e9
     elif attack=='slow_common_ramp_005':shift=.05*(t-onset)/1e9
     elif attack=='radar_aware_offset_095' and valid:shift=max(0.,p+.95)-(g+w)/2
     if valid:
      if attack=='radar_aware_offset_095':g=w=max(0.,p+.95)
      elif attack=='wheel_only_step_7':w+=shift
      else:g+=shift;w+=shift
    base=valid and abs(g-w)<=1;witness=valid and max(abs(g-p),abs(w-p))<=1
    for name,state in states.items():
     grant=state.step(t,g,w,p,valid) if name=='confirmed_cusum_recovery' else state.step(t,base,witness)
     c=counts[name];c['eligible']+=1;c['granted']+=int(grant);c['missing_or_invalid']+=int(not valid)
     if attack=='normal':c['normal_eligible']+=1;c['normal_denied']+=int(not grant)
     if active:
      c['attack_eligible']+=1;c['attack_granted']+=int(grant)
      if valid and abs(shift)>=1:c['attack_shift_ge_1mps_eligible']+=1;c['attack_shift_ge_1mps_granted']+=int(grant)
     elif attack!='normal' and t>offset:
      c['post_attack_eligible']+=1;c['post_attack_denied']+=int(not grant)
      if grant and first_grants[name] is None:first_grants[name]=(t-offset)/1e9
   for name,c in counts.items():
    aggregate[attack+'::'+name].update(c);trials.append({'window_id':window,'role':stream[0]['role'],'attack':attack,'policy':name,**dict(c),'post_attack_first_grant_delay_s':first_grants[name]})
 return {'interpretation':'Abstract attacks under an assumed honest physical radar channel; no authenticated field attacks or independent true-speed labels. All missing evidence is denied and counted. No whole-trip availability claim.','counts':{k:dict(v) for k,v in aggregate.items()},'trials':trials}
def score():
 lock=freeze(False);roles=[r for r in read('protocol.json')['recordings'] if r.startswith('evaluation')];rows=[v for role in roles for v in load(role)];security=replay(rows,lock['parameters']);candidate='confirmed_cusum_recovery';reference='radar_recovery_after_1s_fresh_contiguous_clean';normal=[];per_role={};prefix=0
 for role in roles:
  windows=read('roles/'+role+'/windows.json');prefix+=sum(w['warmup_denied_endpoints'] for w in windows)
  ts=[t for t in security['trials'] if t['role']==role and t['attack']=='normal' and t['policy']==candidate];denied=sum(t['normal_denied'] for t in ts);eligible=sum(t['normal_eligible'] for t in ts);warm=sum(w['warmup_denied_endpoints'] for w in windows)
  per_role[role]={'normal_eligible':eligible,'normal_denied':denied,'normal_denial_fraction':denied/eligible,'warmup_denied':warm,'prefix_inclusive_denial_fraction':(denied+warm)/(eligible+warm),'missing_or_invalid':sum(w['missing_or_invalid_endpoints'] for w in windows)}
  normal.extend({'window_id':t['window_id'],'denied':t['normal_denied'],'eligible':t['normal_eligible'],'fraction':t['normal_denied']/t['normal_eligible']} for t in ts)
 counts=security['counts'];ramps=('common_ramp_075','slow_common_ramp_005');harm=lambda policy:sum(counts[a+'::'+policy].get('attack_shift_ge_1mps_granted',0) for a in ramps);baseline=harm(reference);new=harm(candidate)
 gates={'normal_recording_and_block_denial_le_5pct':all(t['fraction']<=.05 for t in normal) and all(t['normal_denial_fraction']<=.05 for t in per_role.values()),'ramp_harmful_grants_reduced_by_at_least_half':new<=.5*baseline,'startup_and_step_zero_attack_grants':all(counts[a+'::'+candidate].get('attack_granted',0)==0 for a in ('startup_common_bias_7','delayed_common_step_7'))}
 result={'scope':'Single frozen prospective temporal-policy study; unchanged radar predictor; two evaluation drives; no post-evaluation tuning','parameters':lock['parameters'],'gates':gates,'all_predeclared_gates_pass':all(gates.values()),'normal_by_recording':per_role,'normal_by_block':normal,'harmful_ramp_grants':{'original_recovery':baseline,'temporal_candidate':new},'scored_endpoints':len(rows),'warmup_denied':prefix,'trial_count':len(security['trials']),'production_policy_changed':False,'witness_enabled':False,'journal_readiness_established':False}
 return result,security
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='freeze':print(json.dumps(freeze(),indent=2))
 else:
  result,security=score()
  for name,value in [('results.json',result),('security_replay.json',security)]: (ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
  print(json.dumps(result,indent=2))
