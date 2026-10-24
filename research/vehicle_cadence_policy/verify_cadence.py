"""Reconstruct every scored endpoint, parameter and trial from compact source packets."""
from pathlib import Path
import base64,collections,contextlib,gzip,hashlib,io,json,unittest
import core_io as c
import final_core as f
from data_stream import build_stream
import run_cadence as run
import test_cadence
import test_previous_and_stream
import test_observability
from observability import check
ROOT=Path(__file__).resolve().parent
def audit():
 hashes=run.read('sha256.json')
 for name,sha in hashes.items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('payload changed: '+name)
 protocol=run.read('protocol.json');packets=0;endpoints=0;model=run.read('radar_model_lock.json')['models']['signed_median_balanced']
 result,security=run.score()
 roles=['fit','calibration']+([r for r in protocol['recordings'] if r.startswith('evaluation')] if result['evaluation'] is not None else [])
 for role in roles:
  cfg=protocol['recordings'][role]
  folder=ROOT/'roles'/role;plan=run.read('roles/'+role+'/plan.json');meta={n:run.read('metadata/'+role+'/'+n+'.json') for n in ('radars.bag','imus.bag','gnss.bag')};indices={n:c.chunk_infos((ROOT/'metadata'/role/(n+'.index')).read_bytes()) for n in meta};streams=collections.defaultdict(list)
  expected_protocol=run.digest('development_source_protocol.json') if role in ('fit','calibration') else run.digest('protocol.json')
  if plan['protocol_sha256']!=expected_protocol:raise ValueError('protocol mismatch')
  ranges={(q['file'],q['chunk']):q for q in plan['ranges']}
  with gzip.open(folder/'packets.jsonl.gz','rt',encoding='utf8') as file:
   for line in file:
    p=json.loads(line);n=p['file'];i=p['chunk']
    if (n,i) not in ranges:raise ValueError('unselected source packet')
    data=base64.b64decode(p['data'],validate=True);t=p['time_ns'];row=f.decode_gnss(data,t) if n=='gnss.bag' else c.decode_message('radar' if n=='radars.bag' else 'obd',data,t);streams[n,i].append(row);packets+=1
  for (n,i),q in ranges.items():
   topic='/omnipresense/radar/front_bumper' if n=='radars.bag' else '/obd2/speed' if n=='imus.bag' else '/novatel/gps';conn=next(v['conn'] for v in meta[n]['connections'] if v['topic']==topic);end=(indices[n][i+1]['position'] if i+1<len(indices[n]) else meta[n]['index_pos'])-1
   if len(streams[n,i])!=indices[n][i]['counts'].get(conn,0) or q['start']!=indices[n][i]['position'] or q['end']!=end:raise ValueError('source index mismatch')
  first=min(q['start_ns'] for q in indices['radars.bag']);last=max(q['end_ns'] for q in indices['radars.bag']);replayed=[];windows=[];union=set()
  if len(plan['windows'])!=len(cfg['centers']):raise ValueError('block count changed')
  for wi,(w,fraction) in enumerate(zip(plan['windows'],cfg['centers'])):
   start=first+int((last-first)*fraction)-protocol['block_duration_ns']//2;end=start+protocol['block_duration_ns'];chunks={n:[i for i,q in enumerate(index) if q['start_ns']<=end and q['end_ns']>=start-1000000000] for n,index in indices.items()}
   expected={'id':f'{role}_{wi}','role':role,'fraction':fraction,'start_ns':start,'end_ns':end,'chunks':chunks}
   if w!=expected:raise ValueError('fixed block selection changed')
   union.update((n,i) for n,ids in chunks.items() for i in ids)
   selected={n:sorted([r for i in ids for r in streams[n,i]],key=lambda r:r['bag_time_ns']) for n,ids in chunks.items()};rows,stats=build_stream(selected['radars.bag'],selected['imus.bag'],selected['gnss.bag'],start,end,model)
   replayed.extend({'window_id':w['id'],'role':role,**r} for r in rows);windows.append({**w,**stats})
  if union!=set(ranges) or replayed!=run.load(role) or windows!=run.read('roles/'+role+'/windows.json'):raise ValueError('source timeline reconstruction mismatch: '+role)
  endpoints+=len(replayed)
 if result!=run.read('results.json'):raise ValueError('frozen result reconstruction mismatch')
 if check(result['parameters'])!=run.read('observability_result.json'):raise ValueError('observability counterexample reconstruction mismatch')
 if security is not None and security!=run.read('security_replay.json'):raise ValueError('evaluation replay mismatch')
 prior=run.read('previous_policy_lock.json')
 if prior['protocol_sha256']!=run.digest('development_source_protocol.json') or prior['radar_model_sha256']!=run.digest('radar_model_lock.json') or prior['code_sha256']['temporal_core.py']!=run.digest('previous_temporal_core.py'):raise ValueError('previous policy provenance mismatch')
 output=io.StringIO();suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_cadence,test_previous_and_stream,test_observability)]);test=unittest.TextTestRunner(stream=output).run(suite)
 if not test.wasSuccessful():raise ValueError(output.getvalue())
 if result['witness_enabled'] or result['production_policy_changed']:raise ValueError('unsupported policy promotion')
 if result['evaluation'] is None:
  if any((ROOT/'roles'/r).exists() for r in protocol['recordings'] if r.startswith('evaluation')):raise ValueError('new evaluation was opened despite failed gate')
  import subprocess,sys
  guard=subprocess.run([sys.executable,str(ROOT/'acquire_continuous.py'),'evaluation_urban02_night'],capture_output=True,text=True)
  if guard.returncode==0 or 'Development viability gate failed' not in guard.stdout+guard.stderr:raise ValueError('evaluation acquisition guard not enforced')
 print(f'CADENCE EVIDENCE INTEGRITY: PASS ({packets} raw packets; {endpoints} development endpoints; {test.testsRun} policy/freshness tests)')
 for name,value in result['development']['gates'].items():print('DEVELOPMENT GATE '+name+': '+('PASS' if value else 'NOT MET'))
 print('CADENCE DECISION: '+result['decision'])
 print('Fresh evaluation not opened after failed development gates. WITNESS ENABLED: NO.')
 print('BOUNDED-ERROR AMBIGUITY EXAMPLE: REPRODUCED; hypothetical bound, not dataset ground truth.')
 return result
if __name__=='__main__':audit()
