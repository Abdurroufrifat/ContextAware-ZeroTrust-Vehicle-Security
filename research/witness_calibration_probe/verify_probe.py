"""Offline integrity and replay audit. Scientific eligibility is reported separately."""
from pathlib import Path
import csv,hashlib,io,json,math,struct
import probe_core as c
import analyze_probe as analysis
HERE=Path(__file__).resolve().parent

def csv_bytes(rows):
    f=io.StringIO(newline='')
    if rows:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return f.getvalue().encode()

def main():
    hashes=json.loads((HERE/'sha256.json').read_text())
    for name,digest in hashes.items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE.resolve()):raise ValueError('unsafe manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f'digest mismatch: {name}')
    protocol=json.loads((HERE/'protocol.json').read_text());plan=json.loads((HERE/'acquisition_plan.json').read_text())
    if plan['protocol_sha256']!=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest():raise ValueError('protocol changed after selection')
    for role,fractions in protocol['window_fractions'].items():
        if [w['fraction'] for w in plan['windows'] if w['role']==role]!=fractions:raise ValueError('window selection changed')
    downloads=json.loads((HERE/'evidence/downloads.json').read_text())
    metadata={n:json.loads((HERE/'source_index'/(n+'.json')).read_text()) for n in ('radars.bag','imus.bag')}
    chunks={n:c.chunk_infos((HERE/'source_index'/(n+'.index')).read_bytes()) for n in metadata}
    streams={};checked_chunks=0
    for q in plan['ranges']:
        n=q['file'];i=q['chunk_index'];key=q['relative_cache'];data=(HERE/'evidence/chunks'/key).read_bytes();download=downloads[key]
        if len(data)!=q['end']-q['start']+1 or hashlib.sha256(data).hexdigest()!=download['sha256']:raise ValueError('download digest/length mismatch')
        c.validate_range_response(206,download['content_range'],str(len(data)),q['start'],q['end'],q['source_size'])
        outer=list(c.records(data));h,body=outer[0]
        if h.get('op')!=b'\x05' or h.get('compression')!=b'none' or c.u32(h['size'])!=len(body):raise ValueError('invalid chunk')
        kind='radar' if n=='radars.bag' else 'obd';topic='/omnipresense/radar/front_bumper' if kind=='radar' else '/obd2/speed'
        conn=next(v['conn'] for v in metadata[n]['connections'] if v['topic']==topic);counts={};rows=[]
        for head,payload in c.records(body):
            if head['op']!=b'\x02':continue
            cid=c.u32(head['conn']);counts[cid]=counts.get(cid,0)+1
            if cid==conn:rows.append(c.decode_message(kind,payload,c.ros_time(head['time'])))
        if counts!=chunks[n][i]['counts']:raise ValueError('source index/message count mismatch')
        streams[n,i]=rows;checked_chunks+=1
    radar_csv=[];obd_csv=[];pairs_csv=[]
    tmin=min(v['start_ns'] for v in chunks['radars.bag']);tmax=max(v['end_ns'] for v in chunks['radars.bag'])
    for w in plan['windows']:
        target=tmin+int((tmax-tmin)*w['fraction']);idx=w['radar_chunk'];chunk=chunks['radars.bag'][idx]
        if not chunk['start_ns']<=target<=chunk['end_ns']:raise ValueError('window target mismatch')
        if chunk['start_ns']!=w['start_ns'] or chunk['end_ns']!=w['end_ns']:raise ValueError('window times mismatch')
        radar=streams['radars.bag',idx]
        obd=sorted([r for i in w['imu_chunks'] for r in streams['imus.bag',i]],key=lambda r:r['bag_time_ns'])
        pairs,stats=c.causal_pairs(radar,obd,w['start_ns'])
        for rows,dest in [(radar,radar_csv),(obd,obd_csv),(pairs,pairs_csv)]:
            for r in rows:dest.append({'window_id':w['id'],'role':w['role'],**r})
    for name,rows in [('radar_raw',radar_csv),('obd_raw',obd_csv),('causal_pairs',pairs_csv)]:
        if csv_bytes(rows)!=(HERE/(name+'.csv')).read_bytes():raise ValueError(f'replay mismatch: {name}')
    expected=json.loads((HERE/'result.json').read_text());actual=analysis.compute()
    if actual!=expected:raise ValueError('calibration/metric replay mismatch')
    if actual['witness_enabled'] or actual['attack_evaluation_performed']:raise ValueError('probe cannot authorize witness or attack claims')
    print(f'INTEGRITY AND REPLAY PASS: {len(hashes)} payloads, {checked_chunks} source chunks, {len(pairs_csv)} matched records.')
    print('CALIBRATION TARGET:', 'MET (PROVISIONAL ONLY)' if actual['overall_provisional_quality_met'] else 'NOT MET')
    print('WITNESS ENABLED: NO. This is a development probe, not an independent security validation.')

if __name__=='__main__':main()
