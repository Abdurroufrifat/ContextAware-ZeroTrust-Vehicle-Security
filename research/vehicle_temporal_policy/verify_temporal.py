"""Reconstruct every scored endpoint, parameter and trial from compact source packets."""
from pathlib import Path
import base64,collections,contextlib,gzip,hashlib,io,json,unittest
import core_io as c
import final_core as f
from data_stream import build_stream
import run_temporal as run
import test_temporal
from diagnose_results import diagnose
ROOT=Path(__file__).resolve().parent
def audit():
 hashes=run.read('sha256.json')
 for name,sha in hashes.items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('payload changed: '+name)
 protocol=run.read('protocol.json');packets=0;endpoints=0;model=run.read('radar_model_lock.json')['models']['signed_median_balanced']
 for role,cfg in protocol['recordings'].items():
  folder=ROOT/'roles'/role;plan=run.read('roles/'+role+'/plan.json');meta={n:run.read('metadata/'+role+'/'+n+'.json') for n in ('radars.bag','imus.bag','gnss.bag')};indices={n:c.chunk_infos((ROOT/'metadata'/role/(n+'.index')).read_bytes()) for n in meta};streams=collections.defaultdict(list)
  if plan['protocol_sha256']!=run.digest('protocol.json'):raise ValueError('protocol mismatch')
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
 result,security=run.score()
 if result!=run.read('results.json') or security!=run.read('security_replay.json'):raise ValueError('frozen result reconstruction mismatch')
 if diagnose()!=run.read('diagnostics.json'):raise ValueError('diagnostic reconstruction mismatch')
 output=io.StringIO();test=unittest.TextTestRunner(stream=output).run(unittest.defaultTestLoader.loadTestsFromModule(test_temporal))
 if not test.wasSuccessful():raise ValueError(output.getvalue())
 if result['witness_enabled'] or result['production_policy_changed']:raise ValueError('unsupported policy promotion')
 print(f'TEMPORAL EVIDENCE INTEGRITY: PASS ({packets} raw packets; {endpoints} scored development/evaluation endpoints; {test.testsRun} policy tests)')
 for name,value in result['gates'].items():print('RESEARCH GATE '+name+': '+('PASS' if value else 'NOT MET'))
 print('Frozen experiment reproduced. A passed audit is not a passed research gate. WITNESS ENABLED: NO.')
 return result
if __name__=='__main__':audit()
