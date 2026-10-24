"""Exact right-endpoint discrete kinematic interval; all units mm/s, us and um."""
from fractions import Fraction as F
import math,json
def refine(gap0,gap1,lead0,lead1,start,now,commands,broad):
 if type(start) is not int or type(now) is not int or not 0<now-start<=100000 or (now-start)%10000:
  return dict(status="unavailable",reason="unsupported_grid")
 h=F(now-start,1000000);low0,high0=map(F,broad);N=(now-start)//10000
 if low0<=2000*h:return dict(status="unavailable",reason="possible_zero_contact")
 if not commands or len(commands)>16:return dict(status="unavailable",reason="command_coverage")
 low_integral=F(0);high_integral=F(0);command_center=F(0)
 lower_lead=[];upper_lead=[]
 for j in range(1,N+1):
  left=start+(j-1)*10000;right=left+10000
  rows=[r for r in commands if len(r)==4 and r[0]<=left and r[1]>=right]
  if len(rows)!=1:return dict(status="unavailable",reason="command_coverage")
  a,b,n,d=rows[0]
  if any(type(x) is not int for x in (a,b,n,d)) or d<=0 or not -4000*d<=n<=2000*d:
   return dict(status="unavailable",reason="invalid_command")
  u=F(n,d);weight=F(j-1,N)*F(1,100)
  low_integral+=weight*max(F(-4000),u-1000)
  high_integral+=weight*min(F(2000),u+1000)
  command_center+=weight*u
  t=F(j,100)
  lower=max(F(0),F(lead0)-75-4000*t,F(lead1)-75-2000*(h-t))
  upper=min(F(lead0)+75+2000*t,F(lead1)+75+4000*(h-t))
  if lower>upper:return dict(status="conflict",reason="lead_endpoint_conflict")
  lower_lead.append(lower);upper_lead.append(upper)
 gaprate=(F(gap1)-F(gap0))*F(1000,now-start);gaperror=F(3000000,now-start)
 low=max(low0,sum(lower_lead)/N-gaprate-gaperror+low_integral)
 high=min(high0,sum(upper_lead)/N-gaprate+gaperror+high_integral)
 if low>high:return dict(status="conflict",reason="command_scene_conflict")
 return dict(status="available",reason="command_scene_interval",low=low,high=high,center=(low+high)/2,
  command_correction=command_center,lead_delta=F(lead1)-F(lead0),duration_us=now-start)
def relative_features(base,packets,scene,now,refined):
 center=refined["center"];has_c=base.get("candidate_status")=="authenticated" and packets.get("candidate") is not None and base.get("risk_context")=="normal"
 has_a=base.get("anchor_status")=="authenticated" and packets.get("anchor") is not None
 cb=json.loads(packets["candidate"]["wire"])["body"] if has_c else None
 ab=json.loads(packets["anchor"]["wire"])["body"] if has_a else None
 commands=json.loads(packets["command"]["wire"])["body"]["commands"]
 last=F(commands[-1][2],commands[-1][3]) if commands else F(0)
 cd=F(cb["speed_mmps"])-center if has_c else F(0)
 ad=F(ab["speed_mmps"])+last*F(now-ab["measured_us"],1000000)-center if has_a else F(0)
 # A far-away point is unavailable to this residual model, irrespective of attack sign.
 if abs(cd)>500:has_c=False;cd=F(0)
 if abs(ad)>500:has_a=False;ad=F(0)
 return [float(cd/1000),float(ad/1000),float(refined["command_correction"]/1000),
  float(refined["lead_delta"]/1000),float((refined["high"]-refined["low"])/2000),
  float(F(now-ab["measured_us"],100000)) if ab else 0.,float(has_c),float(has_a)]
