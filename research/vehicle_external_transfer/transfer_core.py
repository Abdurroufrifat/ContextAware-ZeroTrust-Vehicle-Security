"""Civic-only public speed-log adapter. Log age and GNSS agreement are proxies."""
import bisect, hashlib, io, json, math, struct, urllib.request, zlib
from pathlib import Path
import numpy as np

SUFFIXES=('CAN/wheel_speed/t','CAN/wheel_speed/value','GNSS/live_gnss_ublox/t','GNSS/live_gnss_ublox/value')
RESERVED=('rav4','b0c9d2329ad1606b','urban02_night','vehicle_final_evaluation')

def safe(path):
    p=Path(path).resolve()
    text=str(p).lower().replace('\\','/').replace(' ','_')
    if any(x in text for x in RESERVED):raise ValueError('Reserved path blocked before access')
    return p

def validate_member(name):
    parts=name.split('/')
    if (len(parts)!=7 or parts[0]!='Chunk_3' or not parts[1].startswith('99c94dc769b5d96e|')
        or not parts[2].isdigit() or parts[3]!='processed_log' or '/'.join(parts[4:]) not in SUFFIXES
        or any(x in name.lower() for x in RESERVED)):
        raise ValueError('Member is outside the Civic speed-log allowlist')

def digest(path):return hashlib.sha256(safe(path).read_bytes()).hexdigest()
def write(path,value):
    p=safe(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def range_get(url,start,end,total):
    if not 0<=start<=end<total or end-start+1>4_000_000:raise ValueError('Invalid bounded byte range')
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=45) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
            raise ValueError('Server did not honor the exact requested range')
        b=r.read(end-start+2)
    if len(b)!=end-start+1:raise ValueError('Incorrect range length')
    return b

def decode(entry,b):
    size=entry['size']
    if not 0<size<=2_000_000:raise ValueError('Member size cap exceeded')
    if entry['method']==8:
        d=zlib.decompressobj(-15);out=d.decompress(b,size+1)
        if not d.eof or d.unused_data or d.unconsumed_tail:raise ValueError('Invalid or excessive compressed data')
    elif entry['method']==0:out=b
    else:raise ValueError('Unsupported ZIP compression')
    if len(out)!=size or zlib.crc32(out)!=entry['crc']:raise ValueError('ZIP size/CRC mismatch')
    return out

def download(entry,selection,destination):
    name=entry['path'];validate_member(name);p=safe(destination)
    if p.exists():
        b=p.read_bytes()
        if len(b)!=entry['size'] or zlib.crc32(b)!=entry['crc']:raise ValueError('Changed cached member')
        return
    off=entry['offset'];total=selection['archive_size'];url=selection['url']
    head=range_get(url,off,off+29,total);h=struct.unpack('<4s5H3L2H',head)
    if h[0]!=b'PK\x03\x04' or h[2]&1 or h[3]!=entry['method']:raise ValueError('Unexpected ZIP local header')
    n,e=h[-2:];start=off+30
    b=range_get(url,start,start+n+e+entry['compressed_size']-1,total)
    if b[:n].decode('utf-8')!=name:raise ValueError('Local/central member names differ')
    out=decode(entry,b[n+e:]);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(out)

def load_array(b):
    a=np.load(io.BytesIO(b),allow_pickle=False)
    if not isinstance(a,np.ndarray) or a.dtype.kind not in 'fiu':raise ValueError('Expected numeric NPY array')
    return a

def timestamps(a):
    if a.ndim!=1 or not np.all(np.isfinite(a)) or np.any(a<0):raise ValueError('Invalid boot timestamps')
    t=[round(float(x)*1_000_000) for x in a]
    if any(y<=x for x,y in zip(t,t[1:])):raise ValueError('Non-increasing timestamps; no sorting/repair allowed')
    return t

def align(gt,gv,wt,wv):
    if gv.shape!=(len(gt),6) or wv.shape!=(len(wt),4):raise ValueError('Array shape differs from documented contract')
    g=timestamps(gt);w=timestamps(wt);rows=[];invalid_g=0;missing=0
    for t,value in zip(g,gv):
        ref=float(value[2]);ref=ref if math.isfinite(ref) and ref>=0 else None
        invalid_g+=ref is None;j=bisect.bisect_right(w,t)-1;stamp=w[j] if j>=0 else None;candidate=None
        if j>=0 and t-w[j]<=100000:
            wheel=[float(x) for x in wv[j]]
            if all(math.isfinite(x) and x>=0 for x in wheel):
                wheel.sort();candidate=(wheel[1]+wheel[2])/2
        missing+=candidate is None
        rows.append(dict(pose_utime=t,pose_speed_mps=ref,wheel_utime=stamp,wheel_speed_mps=candidate))
    return rows,dict(reference_rows=len(g),invalid_reference_rows=invalid_g,stale_or_missing_wheel_rows=missing,
                     wheel_rows=len(w),reference_time_span_us=g[-1]-g[0] if g else 0)
