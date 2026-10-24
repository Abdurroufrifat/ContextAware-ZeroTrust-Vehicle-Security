"""Fixed decision-level injections and three illustrative policy baselines."""
from array import array
import bisect
import math

SCENARIOS=('nominal','wheel_bias_plus3','wheel_drift_to3','wheel_replay_1s_retimed','wheel_replay_1s_original_receipt','wheel_dropout_hold','coordinated_bias_plus3')
POLICIES=('unrestricted','age_only','age_and_agreement')
PHASES=('all','before','during','after','unavailable')


def finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def mask(w,p,t,receipt):
    if not finite(w) or type(t) is not int or type(receipt) is not int or receipt>t:
        return 0
    bits=1
    if 0<=t-receipt<=100000:
        bits|=2
        if finite(p) and abs(w-p)<=1.0:
            bits|=4
    return bits


def transform(rows,scenario):
    if scenario not in SCENARIOS:
        raise ValueError('Unknown frozen scenario: '+scenario)
    timed=[(r['pose_utime'],i) for i,r in enumerate(rows) if type(r.get('pose_utime')) is int]
    times=[t for t,i in timed]; indices=[i for t,i in timed]
    first=times[0] if times else None
    hold=bisect.bisect_left(times,first+5000000)-1 if first is not None else -1
    events=[]
    for i,r in enumerate(rows):
        t=r.get('pose_utime'); w=r.get('wheel_speed_mps'); p=r.get('pose_speed_mps')
        receipt=r.get('wheel_utime'); source=i
        active=bool(scenario!='nominal' and type(t) is int and first is not None and first+5000000<=t<first+15000000)
        if active:
            if scenario in ('wheel_bias_plus3','coordinated_bias_plus3'):
                w=w+3 if finite(w) else None
                if scenario=='coordinated_bias_plus3':
                    p=p+3 if finite(p) else None
            elif scenario=='wheel_drift_to3':
                w=w+3*((t-first-5000000)/10000000) if finite(w) else None
            elif scenario.startswith('wheel_replay_1s_') or scenario=='wheel_dropout_hold':
                at=hold if scenario=='wheel_dropout_hold' else bisect.bisect_right(times,t-1000000)-1
                source=indices[at] if at>=0 else None
                old=rows[source] if source is not None else {}
                w=old.get('wheel_speed_mps'); receipt=old.get('wheel_utime')
                if scenario=='wheel_replay_1s_retimed' and receipt is not None:
                    receipt=t
        events.append([i,active,source,receipt,w,p,mask(w,p,t,receipt)])
    return events


def phase(t,first):
    if type(t) is not int or first is None:
        return 'unavailable'
    return 'before' if t<first+5000000 else 'during' if t<first+15000000 else 'after'


class Cell:
    def __init__(self):
        self.n=0;self.accepted=0;self.unscorable=0;self.offered_large=0
        self.errors=array('d')

    def add(self,event,reference,accepted):
        self.n+=1; w=event[4]
        error=abs(w-reference) if finite(w) and finite(reference) else None
        if event[6]&1 and error is not None and error>2:
            self.offered_large+=1
        if accepted:
            self.accepted+=1
            if error is None:
                self.unscorable+=1
            else:
                self.errors.append(error)

    def report(self):
        errors=sorted(self.errors); n=len(errors)
        pos=(n-1)*.95 if n else 0; lo=int(pos)
        p95=errors[lo]+(errors[min(lo+1,n-1)]-errors[lo])*(pos-lo) if n else None
        bad=sum(e>2 for e in errors)
        coverage=100*self.accepted/self.n if self.n else None
        return dict(requests=self.n,accepted=self.accepted,coverage_percent=coverage,
                    scored_accepted=n,accepted_unscorable=self.unscorable,
                    mae_reference_disagreement_mps=math.fsum(errors)/n if n else None,
                    p95_reference_disagreement_mps=p95,max_reference_disagreement_mps=max(errors) if n else None,
                    accepted_disagreement_gt2_count=bad,
                    offered_disagreement_gt2_count=self.offered_large,
                    blocked_disagreement_gt2_count=self.offered_large-bad,
                    block_rate_on_large_disagreement_percent=100*(self.offered_large-bad)/self.offered_large if self.offered_large else None,
                    diagnostic_gates_pass=bool(self.n and coverage>=95 and p95 is not None and p95<=.5 and bad==0 and self.unscorable==0))


class Bank:
    def __init__(self):
        self.cells={p:{g:Cell() for g in PHASES} for p in POLICIES}

    def add(self,event,reference,group):
        for j,p in enumerate(POLICIES):
            accepted=bool(event[6]&(1<<j))
            self.cells[p]['all'].add(event,reference,accepted)
            self.cells[p][group].add(event,reference,accepted)

    def report(self):
        return {p:{g:self.cells[p][g].report() for g in PHASES} for p in POLICIES}
