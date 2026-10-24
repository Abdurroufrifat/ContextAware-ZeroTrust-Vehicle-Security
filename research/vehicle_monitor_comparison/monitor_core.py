"""Classical statistical baseline adaptations, not full AV defense reproductions."""
import math, statistics, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_conditional_source'))
import conditional_core as source

CASES=source.CASES
POLICIES=('cusum_unprotected','ewma_unprotected','cusum_protected','ewma_protected')
GROUPS=source.GROUPS
TRACKS=('unprotected','protected')


class Cusum:
    def __init__(self,k=.5,h=5.):self.k=k;self.h=h;self.hi=0.;self.lo=0.
    def state(self):return [self.hi,self.lo]
    def step(self,z):
        if not source.finite(z):raise ArithmeticError('Nonfinite standardized residual')
        hi=max(0.,self.hi+z-self.k);lo=max(0.,self.lo-z-self.k)
        if not source.finite(hi) or not source.finite(lo):raise ArithmeticError('CUSUM state overflow')
        self.hi=hi;self.lo=lo
        return self.state(),max(self.hi,self.lo)>self.h


class Ewma:
    def __init__(self):self.value=0.;self.limit=3*math.sqrt(.3/1.7)
    def state(self):return self.value
    def step(self,z):
        if not source.finite(z):raise ArithmeticError('Nonfinite standardized residual')
        value=.3*z+.7*self.value
        if not source.finite(value):raise ArithmeticError('EWMA state overflow')
        self.value=value
        return self.value,abs(self.value)>self.limit


def fit(values):
    n=len(values);mean=statistics.mean(values) if n else None
    sd=statistics.stdev(values) if n>=2 else None
    valid=n>=2 and source.finite(mean) and source.finite(sd) and sd>0
    return dict(n=n,mean=mean,sd=sd,valid=bool(valid))


def events(rows,case):
    times=[r.get('pose_utime') for r in rows if type(r.get('pose_utime')) is int]
    first=times[0] if times else None;history=[[],[]];cal=None;detectors=[Cusum(),Ewma(),Cusum(),Ewma()];out=[];failed=[False,False]
    for e in source.events(rows,case):
        old,a,u,ar,assume=e[:5];i=old[0];t=rows[i].get('pose_utime');c=old[4];peer=old[5]
        current=source.finite(c) and c>=0 and source.fresh(old[3],t)
        protected=source.eligible(a,u,ar,t,assume)
        residual=[c-peer if current and source.finite(peer) and peer>=0 else None,
                  c-a if current and protected else None]
        warm=type(t) is int and first is not None and t<first+5000000
        if type(t) is int and not warm and cal is None:cal=[fit(x) for x in history]
        statuses=[];states=[d.state() for d in detectors];alarms=[None]*4;values=[None]*4;codes=[0]*4
        for j in range(2):
            if type(t) is not int:status='unavailable'
            elif warm:status='warmup'
            elif not cal[j]['valid']:status='invalid_calibration'
            elif failed[j]:status='numerical_failure'
            elif residual[j] is None:status='missing'
            else:status='monitored'
            if status=='monitored':
                before_c=detectors[2*j].state();before_e=detectors[2*j+1].state()
                try:
                    z=(residual[j]-cal[j]['mean'])/cal[j]['sd']
                    states[2*j],alarms[2*j]=detectors[2*j].step(z)
                    states[2*j+1],alarms[2*j+1]=detectors[2*j+1].step(z)
                except ArithmeticError:
                    failed[j]=True;status='numerical_failure'
                    detectors[2*j].hi,detectors[2*j].lo=before_c
                    detectors[2*j+1].value=before_e
                    states[2*j]=before_c;states[2*j+1]=before_e
                    alarms[2*j]=None;alarms[2*j+1]=None
            statuses.append(status)
            if warm and residual[j] is not None:history[j].append(residual[j])
            for k in (2*j,2*j+1):
                if j==0:
                    if warm and e[5][0] is not None:values[k]=e[5][0];codes[k]=1
                    elif status=='monitored' and not alarms[k]:values[k]=c;codes[k]=1
                elif protected:
                    if status=='monitored' and not alarms[k]:values[k]=c;codes[k]=1
                    else:values[k]=a;codes[k]=2
        out.append([i,residual,statuses,states,alarms,values,codes])
    if cal is None:cal=[fit(x) for x in history]
    return out,dict(zip(TRACKS,cal))


class Metrics:
    """Keep each error once by phase; synthesize all-request metrics at reporting."""
    def __init__(self):
        self.cells={p:{g:source.Cell() for g in GROUPS[1:]} for p in POLICIES}
        self.monitor={p:{g:dict(requests=0,monitored=0,alarms=0,warmup=0,invalid_calibration=0,numerical_failure=0,missing=0,unavailable=0) for g in GROUPS} for p in POLICIES}
    def add(self,e,reference,clean,group):
        for j,p in enumerate(POLICIES):
            self.cells[p][group].add(e[5][j],e[6][j],reference,clean)
            for g in ('all',group):
                c=self.monitor[p][g];c['requests']+=1
                status=e[2][j//2];c[status]+=1;c['alarms']+=e[4][j] is True
    def report(self):
        out={}
        for p,groups in self.cells.items():
            total=source.Cell()
            for c in groups.values():
                total.n+=c.n;total.k+=c.k;total.unscorable+=c.unscorable;total.errors.extend(c.errors)
                total.trace_bad+=c.trace_bad
                if c.trace_max is not None:total.trace_max=c.trace_max if total.trace_max is None else max(total.trace_max,c.trace_max)
                total.counts=[a+b for a,b in zip(total.counts,c.counts)]
            out[p]={'all':total.report(),**{g:c.report() for g,c in groups.items()}}
            for g,c in self.monitor[p].items():
                out[p][g]['monitor']=dict(c,alarm_percent_of_monitored=100*c['alarms']/c['monitored'] if c['monitored'] else None)
        return out
