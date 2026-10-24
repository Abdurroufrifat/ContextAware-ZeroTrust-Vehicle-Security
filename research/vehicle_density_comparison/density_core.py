"""Frozen-prefix scalar density adaptation; no native AV framework reproduction."""
from bisect import bisect_left,bisect_right
from functools import lru_cache
import math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_monitor_comparison'))
import monitor_core as prior
source=prior.source
CASES=source.CASES
GROUPS=source.GROUPS
POLICIES=('fixed_density_unprotected','adaptive_density_unprotected','fixed_density_protected','adaptive_density_protected')
EPS=(1.2,1.,.8,.5,.4,.2)
MINPTS=(15,12,10,8,6,4)
ARMS=tuple((e,m) for e in EPS for m in MINPTS)


def normal(model,z):
    if not source.finite(z):raise ArithmeticError('Nonfinite density query')
    core=model['core'];at=bisect_left(core,z)
    return any(0<=j<len(core) and abs(z-core[j])<=model['eps'] for j in (at-1,at))


def model(values,eps,minpts):
    if not source.finite(eps) or eps<=0 or type(minpts) is not int or minpts<1:raise ValueError('Invalid density parameters')
    if not all(source.finite(x) for x in values):raise ArithmeticError('Nonfinite density training value')
    x=sorted(float(v) for v in values);left=0;right=0;core=[]
    for i,value in enumerate(x):
        while left<len(x) and value-x[left]>eps:left+=1
        while right<len(x) and x[right]-value<=eps:right+=1
        if right-left>=minpts:core.append(value)
    groups=[]
    for value in core:
        if not groups or value-groups[-1][-1]>eps:groups.append([])
        groups[-1].append(value)
    clusters=[[] for _ in groups];core_labels=[i for i,g in enumerate(groups) for _ in g]
    for value in x:
        at=bisect_left(core,value)
        labels=[core_labels[j] for j in (at-1,at) if 0<=j<len(core) and abs(value-core[j])<=eps]
        if labels:clusters[min(labels)].append(value)
    if len(clusters)==0:reward=-1.
    elif len(clusters)==1:reward=minpts/eps
    else:
        # Sorted prefix sums give absolute-distance sums without a pairwise matrix.
        tables=[]
        for group in clusters:
            sums=[0.]
            for value in group:sums.append(sums[-1]+value)
            tables.append((group,sums))
        def distance(value,table):
            group,sums=table;k=bisect_right(group,value)
            return max(0.,value*k-sums[k]+sums[-1]-sums[k]-value*(len(group)-k))
        scores=[]
        for label,group in enumerate(clusters):
            silhouettes=[]
            for value in group:
                if len(group)==1:silhouettes.append(0.);continue
                a=distance(value,tables[label])/(len(group)-1)
                b=min(distance(value,t)/len(t[0]) for j,t in enumerate(tables) if j!=label)
                silhouettes.append((b-a)/max(a,b) if max(a,b)>0 else 0.)
            scores.append(math.fsum(silhouettes)/len(group))
        reward=max(scores)
    if not math.isfinite(reward):raise ArithmeticError('Nonfinite density reward')
    return dict(eps=eps,minpts=minpts,core=core,clusters=len(clusters),reward=round(reward,12))


def select(rewards):
    if len(rewards)!=36 or not all(source.finite(x) for x in rewards):raise ValueError('Invalid arm rewards')
    counts=[1]*36
    for t in range(37,73):
        scores=[round(r+math.sqrt(2*math.log(t)/n),12) for r,n in zip(rewards,counts)]
        counts[max(range(36),key=lambda i:(scores[i],-i))]+=1
    scores=[round(r+math.sqrt(2*math.log(73)/n),12) for r,n in zip(rewards,counts)]
    return max(range(36),key=lambda i:(scores[i],-i)),counts


@lru_cache(maxsize=16)
def _fit(values):
    try:fit=prior.fit(values)
    except OverflowError:
        fit=dict(n=len(values),mean=statistics.mean(values) if values else None,sd=None,valid=False)
    for key in ('mean','sd'):
        if fit[key] is not None and not source.finite(fit[key]):fit[key]=None;fit['valid']=False
    models=None
    if fit['valid']:
        z=[(v-fit['mean'])/fit['sd'] for v in values]
        if not all(source.finite(x) for x in z):fit['valid']=False
        else:
            all_models=[model(z,e,m) for e,m in ARMS]
            choice,counts=select([m['reward'] for m in all_models])
            models=[all_models[ARMS.index((.5,8))],all_models[choice]]
            def describe(m):return {k:v for k,v in m.items() if k!='core'}|{'core_count':len(m['core'])}
            fit=fit|dict(fixed=describe(models[0]),adaptive=describe(models[1])|dict(arm_index=choice,pull_counts=counts))
    return fit,models


def fit_models(values):return _fit(tuple(values[:1000]))


def events(rows,case):
    times=[r.get('pose_utime') for r in rows if type(r.get('pose_utime')) is int]
    boundary=times[0]+5000000 if times else None
    previous=[None,None];history=[[],[]];cal=None;models=None;failed=[False,False];out=[]
    for e in source.events(rows,case):
        old,a,u,ar,assume=e[:5];i=old[0];t=rows[i].get('pose_utime');c=old[4];peer=old[5]
        current=source.finite(c) and c>=0 and source.fresh(old[3],t)
        anchor_ok=source.eligible(a,u,ar,t,assume)
        residual=[c-peer if current and source.finite(peer) and peer>=0 else None,c-a if current and anchor_ok else None]
        differences=[]
        for j,r in enumerate(residual):
            old_value=previous[j]
            value=r-old_value[0] if r is not None and old_value is not None and type(t) is int and 0<=t-old_value[1]<=100000 else None
            differences.append(value)
            previous[j]=(r,t) if r is not None and type(t) is int else None
        warm=type(t) is int and boundary is not None and t<boundary
        if type(t) is int and not warm and cal is None:
            fits=[fit_models(x) for x in history];cal=[f[0] for f in fits];models=[f[1] for f in fits]
        statuses=[];states=[None]*4;alarms=[None]*4;values=[None]*4;codes=[0]*4
        for j in range(2):
            delta=differences[j]
            if delta is not None and not source.finite(delta):failed[j]=True;differences[j]=None
            status=('unavailable' if type(t) is not int else 'warmup' if warm else 'invalid_calibration' if not cal[j]['valid'] else
                    'numerical_failure' if failed[j] else 'missing' if delta is None else 'monitored')
            if warm and delta is not None and source.finite(delta) and len(history[j])<1000:history[j].append(delta)
            if status=='monitored':
                z=(delta-cal[j]['mean'])/cal[j]['sd']
                if not source.finite(z):failed[j]=True;status='numerical_failure'
                else:
                    for k in (0,1):
                        states[2*j+k]=z;alarms[2*j+k]=not normal(models[j][k],z)
            statuses.append(status)
            for k in (2*j,2*j+1):
                if j==0:
                    if warm and e[5][0] is not None:values[k]=e[5][0];codes[k]=1
                    elif status=='monitored' and alarms[k] is False:values[k]=c;codes[k]=1
                elif anchor_ok:
                    values[k]=c if status=='monitored' and alarms[k] is False else a
                    codes[k]=1 if status=='monitored' and alarms[k] is False else 2
        out.append([i,differences,statuses,states,alarms,values,codes])
    if cal is None:cal=[fit_models(x)[0] for x in history]
    return out,dict(zip(('unprotected','protected'),cal))


class Metrics:
    def __init__(self):self.inner=prior.Metrics()
    def add(self,*args):self.inner.add(*args)
    def report(self):return dict(zip(POLICIES,self.inner.report().values()))
