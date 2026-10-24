"""Research point-time association from preserved scan Header plus encoded firing offsets."""
import numpy as np
MODEL="unibwtas_extended_seconds_from_scan_start"
def firing_times(msg,time_model):
    if time_model!=MODEL: raise ValueError("Unknown point-time semantics")
    h=msg.header.stamp
    if not 0<=int(h.nanosec)<1000000000: raise ValueError("Invalid header nanoseconds")
    header=int(h.sec)*1000000000+int(h.nanosec)
    if not 0<=header<2**63-1000000000: raise ValueError("Unsupported header epoch")
    if msg.width<1 or msg.height<1 or msg.point_step<1 or msg.row_step<msg.width*msg.point_step:
        raise ValueError("Invalid cloud layout")
    if len(msg.data)<msg.row_step*msg.height: raise ValueError("Truncated point payload")
    fields={}
    for f in msg.fields:
        if f.name in fields: raise ValueError("Duplicate field")
        fields[f.name]=f
    values={}
    for name in ("time","distance"):
        if name not in fields: raise ValueError("Required numeric field missing")
        f=fields[name]
        if f.datatype not in (7,8) or f.count!=1: raise ValueError("Unsupported field encoding")
        dtype=np.dtype((">" if msg.is_bigendian else "<")+("f4" if f.datatype==7 else "f8"))
        if f.offset<0 or f.offset+dtype.itemsize>msg.point_step: raise ValueError("Field exceeds point layout")
        values[name]=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,offset=f.offset,
            strides=(msg.row_step,msg.point_step)).reshape(-1).astype(np.float64)
    times=values["time"];distance=values["distance"]
    good=np.isfinite(times)&np.isfinite(distance)&(distance>0)
    offsets=times[good]
    if np.any(offsets<-.001) or np.any(offsets>.2): raise ValueError("Offsets outside declared 10Hz scan model")
    # Never convert the large epoch to float; encoded float seconds are relative.
    return np.rint(offsets*1000000000).astype(np.int64)+np.int64(header)
def fresh_mask(stamps,received_ns,now_ns):
    if received_ns>now_ns: return np.zeros(len(stamps),dtype=bool)
    return (stamps<=now_ns)&(stamps>=now_ns-100000000)
