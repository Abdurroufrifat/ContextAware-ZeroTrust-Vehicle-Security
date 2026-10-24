"""Exact scalar adaptation of S3E agreement sets, not a new estimation principle."""
from fractions import Fraction as F

def merge(intervals):
 out=[]
 for lo,hi in sorted((F(a),F(b)) for a,b in intervals):
  if lo>hi:raise ValueError("reversed interval")
  if out and lo<=out[-1][1]:out[-1]=(out[-1][0],max(hi,out[-1][1]))
  else:out.append((lo,hi))
 return out

def agree(prior,measurements):
 prior=merge(prior)
 if len(measurements)!=2:raise ValueError("p=2,q=1 specialization")
 if any(x is None for x in measurements):return prior
 pieces=[]
 for a,b in prior:
  for c,d in measurements:
   c,d=F(c),F(d)
   if c>d:raise ValueError("reversed measurement")
   lo,hi=max(a,c),min(b,d)
   if lo<=hi:pieces.append((lo,hi))
 return merge(pieces)

def predict(prior,dt_us,command):
 if type(dt_us) is not int or dt_us<=0:raise ValueError("invalid time")
 if command is not None and (type(command) is not int or not -4000<=command<=2000):raise ValueError("invalid command")
 lo,hi=(-4000,2000) if command is None else (max(-4000,command-1000),min(2000,command+1000))
 dt=F(dt_us,1000000)
 return merge([(max(F(0),a+lo*dt),max(F(0),b+hi*dt)) for a,b in merge(prior)])

def measurement(speed,age_us,clock_us):
 if any(type(x) is not int or x<0 for x in (speed,age_us,clock_us)):raise ValueError("invalid measurement")
 if age_us-clock_us<0 or age_us+clock_us>100000:return None
 dt=F(age_us+clock_us,1000000)
 return max(F(0),F(speed-250)-4000*dt),F(speed+250)+2000*dt

def output(intervals,lockdown=False):
 intervals=merge(intervals)
 if not intervals or lockdown:return None
 lo,hi=intervals[0][0],intervals[-1][1]
 value=round((lo+hi)/2)
 return value if max(abs(value-lo),abs(value-hi))<=500 else None
