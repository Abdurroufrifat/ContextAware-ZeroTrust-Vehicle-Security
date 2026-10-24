"""Offline development probe primitives; never issues witness evidence or permission."""
import bisect
import math
import statistics
import struct


def u32(b,p=0):
    if p<0 or p+4>len(b):raise ValueError('truncated uint32')
    return struct.unpack_from('<I',b,p)[0]


def fields(b):
    result={};p=0
    while p<len(b):
        n=u32(b,p);p+=4
        if n>len(b)-p:raise ValueError('truncated header field')
        q=b[p:p+n];p+=n
        if b'=' not in q:raise ValueError('missing field separator')
        k,v=q.split(b'=',1);k=k.decode('ascii')
        if k in result:raise ValueError('duplicate field')
        result[k]=v
    return result


def records(b):
    p=0
    while p<len(b):
        n=u32(b,p);p+=4
        if n>len(b)-p:raise ValueError('truncated record header')
        h=fields(b[p:p+n]);p+=n;l=u32(b,p);p+=4
        if l>len(b)-p:raise ValueError('truncated record data')
        d=b[p:p+l];p+=l;yield h,d


def ros_time(b):
    if len(b)!=8:raise ValueError('invalid ROS time')
    s,ns=struct.unpack('<II',b)
    if ns>=1000000000:raise ValueError('invalid nanoseconds')
    return s*1000000000+ns


def chunk_infos(index_bytes):
    result=[]
    for h,d in records(index_bytes):
        if h['op']!=b'\x06':continue
        if len(d)%8:raise ValueError('invalid chunk count data')
        counts=dict(struct.iter_unpack('<II',d))
        if len(counts)!=u32(h['count']):raise ValueError('chunk connection count mismatch')
        result.append({'position':struct.unpack('<Q',h['chunk_pos'])[0],
                       'start_ns':ros_time(h['start_time']),'end_ns':ros_time(h['end_time']),
                       'counts':counts})
    if not result:raise ValueError('no chunks')
    return result


class Cursor:
    def __init__(self,b):self.b=b;self.p=0
    def take(self,n):
        if n<0 or n>len(self.b)-self.p:raise ValueError('truncated message')
        v=self.b[self.p:self.p+n];self.p+=n;return v
    def unpack(self,fmt):return struct.unpack(fmt,self.take(struct.calcsize(fmt)))
    def string(self):return self.take(self.unpack('<I')[0]).decode('utf8')
    def header(self):
        seq,s,ns=self.unpack('<III')
        if ns>=1000000000:raise ValueError('invalid message nanoseconds')
        return seq,s*1000000000+ns,self.string()
    def array(self):self.take(self.unpack('<I')[0]*4)


def decode_message(kind,b,bag_time_ns):
    c=Cursor(b)
    if kind=='radar':
        speed=c.unpack('<f')[0];direction=c.string();sensor_time=c.unpack('<f')[0]
        sensorid=c.string();objnum=c.unpack('<f')[0];seq,stamp,frame=c.header()
        for _ in range(4):c.array()
        distance=c.unpack('<f')[0]
        extra={'direction':direction,'sensor_id':sensorid,'device_time_raw':sensor_time,
               'objects_raw':objnum,'range_raw':distance,'unit':'UNVERIFIED'}
    elif kind=='obd':
        seq,stamp,frame=c.header();speed=c.unpack('<d')[0];extra={'unit':c.string()}
    else:raise ValueError('unknown message kind')
    if c.p!=len(b):raise ValueError('unexpected trailing bytes')
    return {'bag_time_ns':bag_time_ns,'header_time_ns':stamp,'sequence':seq,
            'frame_id':frame,'speed_raw':speed,**extra}


def valid_sample(r,max_delivery):
    return (math.isfinite(r['speed_raw']) and
            0<=r['bag_time_ns']-r['header_time_ns']<=max_delivery)


def causal_pairs(radars,obds,window_start_ns):
    """Radar-only causal feature, with benign OBD matched for offline comparison."""
    for rows in (radars,obds):
        if any(a['bag_time_ns']>b['bag_time_ns'] for a,b in zip(rows,rows[1:])):
            raise ValueError('unordered stream')
    valid_obd=[r for r in obds if valid_sample(r,100000000) and r['unit']=='kph' and r['speed_raw']>=0]
    otimes=[r['bag_time_ns'] for r in valid_obd];history=[];hp=0;pairs=[];eligible=0
    for r in radars:
        t=r['bag_time_ns'];good=valid_sample(r,250000000)
        if good:history.append(r)
        while hp<len(history) and history[hp]['bag_time_ns']<t-1000000000:hp+=1
        if t<window_start_ns+1000000000:continue
        eligible+=1
        if not good or len(history)-hp<10:continue
        j=bisect.bisect_right(otimes,t)-1
        if j<0:continue
        o=valid_obd[j]
        if t-o['header_time_ns']>100000000:continue
        pairs.append({'time_ns':t,'feature_raw':statistics.median(abs(q['speed_raw']) for q in history[hp:]),
                      'label_mps':o['speed_raw']/3.6,'obd_header_age_ns':t-o['header_time_ns'],
                      'radar_delivery_ns':t-r['header_time_ns'],'radar_history_samples':len(history)-hp})
    return pairs,{'eligible_radar':eligible,'pairs':len(pairs),'coverage':len(pairs)/eligible if eligible else 0.0}


def fit_affine(rows):
    if len(rows)<2:return None
    xs=[r['feature_raw'] for r in rows];ys=[r['label_mps'] for r in rows]
    if not all(math.isfinite(v) for v in xs+ys) or max(ys)-min(ys)<3.0:return None
    x=statistics.mean(xs);y=statistics.mean(ys);den=sum((v-x)**2 for v in xs)
    if den<=1e-12:return None
    slope=sum((a-x)*(b-y) for a,b in zip(xs,ys))/den
    if slope<=0 or not math.isfinite(slope):return None
    return {'slope':slope,'intercept':y-slope*x}


def predict(model,feature_raw):
    return max(0.0,model['slope']*feature_raw+model['intercept'])

def validate_range_response(status,content_range,content_length,start,end,total):
    """RFC 7233 permits an unknown complete length ('*') for a valid range."""
    import re
    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',content_range or '')
    if status!=206 or match is None:raise ValueError('unexpected range status/header')
    first,last,complete=match.groups()
    if int(first)!=start or int(last)!=end:raise ValueError('range offsets mismatch')
    if complete!='*' and int(complete)!=total:raise ValueError('source size mismatch')
    if content_length is not None and int(content_length)!=end-start+1:raise ValueError('range byte count mismatch')
