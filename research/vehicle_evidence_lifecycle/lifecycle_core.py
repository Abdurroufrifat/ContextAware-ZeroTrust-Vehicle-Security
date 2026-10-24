"""10ms right-endpoint certificates; no continuous-time or unbounded-attack claim."""
import math
DT=.01
def clip(x):return max(-4.,min(2.,x))
def stopping(v,b):
    if v<0 or b<=0 or not math.isfinite(v+b):raise ValueError("stopping state")
    n=max(0,math.ceil(v/(b*DT))-1)
    return DT*(n*v-b*DT*n*(n+1)/2)
def margin(command,ego_high,lead_low,gap_low,error,steps=10):
    if type(command) is not int or not -4000<=command<=2000:raise ValueError("command")
    if not 0<=error<4 or type(steps) is not int or not 0<=steps<=100:raise ValueError("authority/hold")
    if not all(math.isfinite(x) for x in (ego_high,lead_low,gap_low,error)) or min(ego_high,lead_low)<0:raise ValueError("state")
    ve,vl,g=ego_high,lead_low,gap_low;lowest=g-2
    for _ in range(steps):
        ve=max(0.,ve+clip(command/1000+error)*DT);vl=max(0.,vl-4*DT)
        g+=(vl-ve)*DT;lowest=min(lowest,g-2)
    terminal=g-2+stopping(vl,4)-stopping(ve,4-error)
    return min(lowest,terminal)-1e-8*(1+abs(gap_low)+ego_high+lead_low)
def choose(desired,ego,lead,gap,error):
    if type(desired) is not int or not -4000<=desired<=2000:raise ValueError("desired")
    args=(ego,lead,gap,error)
    if margin(-4000,*args)<0:return -4000,margin(-4000,*args)
    lo,hi=-4000,desired
    while lo<hi:
        mid=(lo+hi+1)//2
        if margin(mid,*args)>=0:lo=mid
        else:hi=mid-1
    return lo,margin(lo,*args)
def lease_steps(command,ego,lead,gap,error):
    if margin(command,ego,lead,gap,error,0)<0:return 0
    lo,hi=0,100
    while lo<hi:
        mid=(lo+hi+1)//2
        if margin(command,ego,lead,gap,error,mid)>=0:lo=mid
        else:hi=mid-1
    return lo
def interval(x):
    if len(x)!=2 or not all(math.isfinite(v) for v in x) or x[0]>x[1]:raise ValueError("interval")
    return list(map(float,x))
class ContextTube:
    def __init__(self,ego,lead,gap):
        self.ego=interval(ego);self.lead=interval(lead);self.gap=interval(gap)
        if min(*self.ego,*self.lead)<0:raise ValueError("negative speed")
    def advance(self,command,dt,error):
        n=round(dt/DT)
        if dt<0 or abs(n*DT-dt)>1e-10 or not 0<=error<4:raise ValueError("time/authority")
        for _ in range(n):
            self.ego=[max(0,self.ego[0]+clip(command/1000-error)*DT),max(0,self.ego[1]+clip(command/1000+error)*DT)]
            self.lead=[max(0,self.lead[0]-4*DT),self.lead[1]+2*DT]
            self.gap=[math.nextafter(self.gap[0]+(self.lead[0]-self.ego[1])*DT,-math.inf),
                      math.nextafter(self.gap[1]+(self.lead[1]-self.ego[0])*DT,math.inf)]
    def tighten_ego(self,ego):
        e=interval(ego);lo=max(e[0],self.ego[0]);hi=min(e[1],self.ego[1])
        if lo>hi:return False
        self.ego=[lo,hi];return True
    def observe(self,gap,lead):
        if gap is None or lead is None:return "missing_context"
        g=interval(gap);l=interval(lead)
        if g[1]<self.gap[0] or self.gap[1]<g[0]:return "range_conflict"
        if l[1]<self.lead[0] or self.lead[1]<l[0]:return "lead_conflict"
        return "plausible_unqualified_refresh"
    def bounds(self):
        return {"ego_low":self.ego[0],"ego_high":self.ego[1],"lead_low":self.lead[0],"lead_high":self.lead[1],
                "gap_low":self.gap[0],"gap_high":self.gap[1],"lead_mps":sum(self.lead)/2,"gap_m":sum(self.gap)/2}
