"""Metadata-only additional dates; exact local-header payload bounds."""
import struct
import numpy as np
from coact_cps.vehicle.data24 import check_channel

CHANNELS={'GNSS/live_gnss_ublox':6,'CAN/wheel_speed':4,'IMU/accelerometer':3,'IMU/gyro':3}


def select_trips(records,excluded_routes):
    names=[r['name'] for r in records]
    if len(set(names))!=len(names):raise ValueError('duplicate archive entries')
    excluded=set(excluded_routes);days={r.split('|')[1][:10] for r in excluded}
    available={}
    for name in names:
        parts=name.split('/')
        if (len(parts)!=7 or parts[0]!='Chunk_2' or parts[3]!='processed_log'
            or not parts[1].startswith('b0c9d2329ad1606b|') or not parts[2].isdigit()):continue
        route=parts[1];day=route.split('|')[1][:10]
        if route in excluded or day in days:continue
        available.setdefault(route,{}).setdefault(int(parts[2]),set()).add('/'.join(parts[4:]))
    required={f'{channel}/{kind}' for channel in CHANNELS for kind in ('t','value')}
    result=[];selected_days=set()
    for route,segments in sorted(available.items()):
        complete={s for s,files in segments.items() if required<=files}
        pairs=sorted(s for s in complete if s+1 in complete);day=route.split('|')[1][:10]
        if not pairs or day in selected_days:continue
        selected_days.add(day);i=len(result)
        result.append(dict(id=f'rav31_{i+1:02d}',route_id=route,day=day,segments=[pairs[0],pairs[0]+1],
            role='development_pool' if i<6 else 'reserved_validation'))
        if len(result)==8:return result
    raise ValueError('eight eligible distinct complete dates required')


def payload_end(header,start,compressed_size):
    if (len(header)!=30 or type(start) is not int or start<0 or type(compressed_size) is not int
        or not 0<compressed_size<=8*1024*1024):raise ValueError('bounded exact local header required')
    fields=struct.unpack('<4s5H3L2H',header)
    if (fields[0]!=b'PK\x03\x04' or fields[2]&(1|32|64|8192) or fields[3] not in (0,8)
        or not 0<fields[-2]<=2048 or fields[-1]>4096):raise ValueError('invalid local ZIP header')
    return start+30+fields[-2]+fields[-1]+compressed_size-1


def channel_report(time,values,columns,allow_ties):
    try:return dict(status='PASS',**check_channel(time,values,columns,allow_ties))
    except ValueError as exc:
        # Report failed strict ordering; allowing ties here supplies diagnostic
        # counts only and never changes the quarantined eligibility status.
        try:details=check_channel(time,values,columns,True)
        except ValueError:details=dict(samples=int(len(np.asarray(time))))
        return dict(status='QUARANTINED',issue=str(exc),**details)
