"""Exact one-dimensional point-mass feedback model; no vehicle authority."""
from fractions import Fraction as F
DT_US=50000

def advance(speed,position,acceleration,dt_us):
    if any(type(x) not in (int,F) for x in (speed,position,acceleration)) or type(dt_us) is not int:
        raise ValueError("typed model quantities required")
    v,x,a=map(F,(speed,position,acceleration))
    if v<0 or x<0 or not -4000<=a<=2000 or dt_us<=0:
        raise ValueError("invalid model state or command")
    dt=F(dt_us,1000000)
    moving=min(dt,-v/a) if a<0 and v+a*dt<0 else dt
    return max(F(0),v+a*dt),x+v*moving+a*moving*moving/2

def speed_at(history,sample_us,now_us,current_speed):
    if type(sample_us) is not int or type(now_us) is not int or sample_us>now_us:
        raise ValueError("noncausal source query")
    if sample_us<0:return None
    if sample_us==now_us:return F(current_speed)
    index=sample_us//DT_US
    if index>=len(history):return None
    start,v,a=history[index]
    if start!=index*DT_US or not start<=sample_us<start+DT_US:
        raise ValueError("history interval mismatch")
    return max(F(0),v+a*F(sample_us-start,1000000))

def command(target_mmps,estimate_mmps):
    if type(target_mmps) is not int or target_mmps<0:
        raise ValueError("invalid target")
    if estimate_mmps is None:return F(-2000)
    if type(estimate_mmps) is not int or estimate_mmps<0:
        raise ValueError("invalid selected estimate")
    return max(F(-4000),min(F(2000),F(4,5)*(target_mmps-estimate_mmps)))

def target(time_us):
    if type(time_us) is not int or time_us<0:raise ValueError("invalid command time")
    return 15000 if time_us<8000000 else 22000 if time_us<18000000 else 10000
