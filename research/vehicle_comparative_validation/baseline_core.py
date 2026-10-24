"""Explicit scalar adaptations: S3E interval unions and relaxed-reconstruction ZO-CBF."""
import itertools,math
def merge_sets(intervals):
    out=[]
    for lo,hi in sorted((float(a),float(b)) for a,b in intervals if a<=b):
        if not (math.isfinite(lo) and math.isfinite(hi)):raise ValueError("finite sets required")
        if out and lo<=out[-1][1]+1e-12:out[-1]=(out[-1][0],max(hi,out[-1][1]))
        else:out.append((lo,hi))
    return out
def intersection(left,right):
    return merge_sets([(max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<=min(b,d)])
def agreement(observations,q=1):
    if type(q) is not int or q<0:raise ValueError("attack budget")
    observations=[x for x in observations if x is not None]
    if len(observations)<=q:return []
    result=[]
    for group in itertools.combinations(observations,len(observations)-q):
        lo=max(x[0] for x in group);hi=min(x[1] for x in group)
        if lo<=hi:result.append((lo,hi))
    return merge_sets(result)
def predict_sets(previous,u,dt,bound=1):
    if dt<0 or bound<0:raise ValueError("time/bound")
    a=max(-4,min(2,u-bound));b=max(-4,min(2,u+bound))
    return merge_sets([(max(0,lo+dt*a),max(0,hi+dt*b)) for lo,hi in previous])
def s3e_update(previous,observations,u,dt,bound=1):
    prediction=predict_sets(previous,u,dt,bound)
    return intersection(prediction,agreement(observations,1))
def closest_point(intervals):
    return None if not intervals else (intervals[0][0]+intervals[-1][1])/2
def cbf_slack(command,ego_high,lead_low,gap_low):
    ego=ego_high;lead=lead_low;gap=gap_low
    initial=gap-2-ego
    minimum=min(gap-2,initial)
    for _ in range(10):
        ego=max(0,ego+max(-4,min(2,command/1000+1))*.01)
        lead=max(0,lead-.04)
        gap+=(lead-ego)*.01;minimum=min(minimum,gap-2,gap-2-ego)
    return min(minimum,gap-2-ego-.8*initial)
def zocbf_command(desired,ego_high,lead_low,gap_low):
    desired=max(-4000,min(2000,int(desired)))
    if cbf_slack(-4000,ego_high,lead_low,gap_low)<0:return -4000,cbf_slack(-4000,ego_high,lead_low,gap_low)
    lo=-4000;hi=desired
    while lo<hi:
        mid=(lo+hi+1)//2
        if cbf_slack(mid,ego_high,lead_low,gap_low)>=0:lo=mid
        else:hi=mid-1
    return lo,cbf_slack(lo,ego_high,lead_low,gap_low)
