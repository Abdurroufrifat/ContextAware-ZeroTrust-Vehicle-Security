"""Segment eligibility and explicit normal-transfer aggregation; no policy."""
from collections import defaultdict
from numbers import Integral
import numpy as np


def require_eligible(trip,segment,eligible):
    if (trip['role'] not in ('development','development_pool') or segment not in trip['segments']
        or (trip['id'],segment) not in eligible):raise ValueError('reserved, quarantined, or unlisted segment forbidden')


def alarm_summary(alarm,positions,time,start):
    a,p,t,s=map(np.asarray,(alarm,positions,time,start));n=len(t)
    if (t.ndim!=1 or s.shape!=t.shape or t.dtype.kind not in 'iu' or s.dtype.kind not in 'iu'
        or p.ndim!=1 or not len(p) or p.dtype.kind not in 'iu' or a.shape!=p.shape or a.dtype.kind!='b'
        or np.any(p<0) or np.any(p>=n) or np.any(np.diff(p.astype(np.int64))<=0)
        or np.any(t<=s) or np.any(np.diff(t.astype(np.int64))<=0) or np.any(s[1:]<t[:-1])):
        raise ValueError('ordered integer intervals/positions and aligned boolean alarms required')
    longest=current=0
    for j,i in enumerate(p):
        contiguous=j>0 and i==p[j-1]+1 and s[i]==t[p[j-1]]
        current=(current+1 if contiguous else 1) if a[j] else 0;longest=max(longest,current)
    return dict(epochs=n,scored_endpoints=len(p),unscored_epochs=n-len(p),alarm_endpoints=int(a.sum()),
        presumed_normal_alarm_fraction=float(a.mean()),longest_alarm_run_epochs=longest,
        alarm_observed_span_ms=int((t-s)[p][a].sum()))


def aggregate(rows):
    if not rows:raise ValueError('nonempty normal-transfer rows required')
    grouped=defaultdict(dict);seen=set();model_alarms=model_endpoints=0
    for r in rows:
        key=(r['trip'],r['seed'],r['segment']);n,a=r['scored_endpoints'],r['alarm_endpoints']
        if (key in seen or not isinstance(n,Integral) or isinstance(n,bool) or n<=0
            or not isinstance(a,Integral) or isinstance(a,bool) or not 0<=a<=n):
            raise ValueError('unique segment/seed rows and valid endpoint counts required')
        seen.add(key);grouped[(r['trip'],r['seed'])][r['segment']]=(int(n),int(a))
        model_alarms+=int(a);model_endpoints+=int(n)
    trips=defaultdict(list)
    for (trip,seed),segments in grouped.items():trips[trip].append((seed,segments))
    means={};unique=0;replicates={}
    for trip,blocks in trips.items():
        expected={segment:v[0] for segment,v in blocks[0][1].items()}
        if any({segment:v[0] for segment,v in b.items()}!=expected for _,b in blocks):
            raise ValueError('all seed replicas must cover identical segment endpoints')
        fractions=[sum(a for n,a in b.values())/sum(n for n,a in b.values()) for _,b in blocks]
        means[trip]=float(np.mean(fractions));unique+=sum(expected.values());replicates[trip]=len(blocks)
    return dict(trip_alarm_fractions=means,macro_alarm_fraction=float(np.mean(list(means.values()))),
        worst_trip_alarm_fraction=max(means.values()),pooled_model_endpoint_alarm_fraction=model_alarms/model_endpoints,
        pooled_model_endpoints=model_endpoints,unique_sensor_endpoints=unique,seed_replicates_per_trip=replicates)
