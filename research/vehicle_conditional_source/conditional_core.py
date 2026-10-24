"""Conditional ablation only: a simulated protected copy is not an independent sensor."""
from array import array
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_can_attack_audit'))
import attack_core as old

CASES=('nominal','wheel_bias_plus3','wheel_drift_to3','wheel_replay_1s_retimed','wheel_replay_1s_original_receipt','wheel_dropout_hold','coordinated_bias_plus3','coordinated_u0','coordinated_u075','protected_copy_missing','protected_copy_stale','protected_copy_compromised','protected_bound_violated','unestablished_real_evidence')
POLICIES=('legacy_agreement','conditional_anchor_only','conditional_bounded_candidate')
GROUPS=('all','before','during','after','unavailable')


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def fresh(stamp,t):return type(stamp) is int and type(t) is int and 0<=stamp<=t and t-stamp<=100000


def eligible(a,u,stamp,t,assume):
    return assume is True and finite(a) and a>=0 and finite(u) and 0<=u<=.5 and fresh(stamp,t)


def choose(c,cr,a,u,ar,t,assume):
    if not eligible(a,u,ar,t,assume):return None,0
    if finite(c) and c>=0 and fresh(cr,t) and abs(c-a)+u<=.5:return c,1
    return a,2


def events(rows,case):
    if case not in CASES:raise ValueError('Unknown frozen case')
    base=case if case in old.SCENARIOS else 'coordinated_bias_plus3'
    original=old.transform(rows,base);out=[]
    u=0 if case=='coordinated_u0' else .75 if case=='coordinated_u075' else .25
    for i,(r,event) in enumerate(zip(rows,original)):
        clean=r.get('wheel_speed_mps');t=r.get('pose_utime')
        a=float(clean)+u*(i%2) if finite(clean) else None
        ar=r.get('wheel_utime');assume=case!='unestablished_real_evidence'
        if event[1]:
            if case=='protected_copy_missing':a=None;ar=None
            elif case=='protected_copy_stale':ar=t-100001
            elif case=='protected_copy_compromised':a=a+3 if finite(a) else None
            elif case=='protected_bound_violated':a=a+1 if finite(a) else None
        legacy=event[4] if event[6]&4 else None
        anchor=a if eligible(a,u,ar,t,assume) else None
        chosen,code=choose(event[4],event[3],a,u,ar,t,assume)
        reasons=[1 if legacy is not None else 0,2 if anchor is not None else 0,code]
        violated=bool(finite(a) and finite(clean) and abs(a-clean)>u+1e-12)
        out.append([event,a,u,ar,assume,[legacy,anchor,chosen],reasons,violated])
    return out


class Cell:
    def __init__(self):
        self.n=0;self.k=0;self.unscorable=0;self.errors=array('d');self.trace_bad=0;self.trace_max=None;self.counts=[0,0,0]
    def add(self,value,code,reference,clean):
        self.n+=1;self.counts[code]+=1
        if value is None:return
        self.k+=1
        if finite(reference):self.errors.append(abs(value-reference))
        else:self.unscorable+=1
        if finite(clean):
            delta=abs(value-clean);self.trace_bad+=delta>.5+1e-12
            self.trace_max=delta if self.trace_max is None else max(self.trace_max,delta)
    def report(self):
        e=sorted(self.errors);n=len(e);position=(n-1)*.95 if n else 0;lo=int(position)
        p95=e[lo]+(e[min(lo+1,n-1)]-e[lo])*(position-lo) if n else None
        coverage=100*self.k/self.n if self.n else None;bad=sum(x>2 for x in e)
        return dict(requests=self.n,accepted=self.k,coverage_percent=coverage,scored_accepted=n,
                    accepted_unscorable=self.unscorable,mae_reference_disagreement_mps=math.fsum(e)/n if n else None,
                    p95_reference_disagreement_mps=p95,max_reference_disagreement_mps=max(e) if n else None,
                    accepted_disagreement_gt2_count=bad,trace_bound_violation_count=self.trace_bad,
                    max_deviation_from_original_wheel_mps=self.trace_max,
                    selection_counts=dict(abstain=self.counts[0],candidate=self.counts[1],protected_copy=self.counts[2]),
                    diagnostic_gates_pass=bool(self.n and coverage>=95 and p95 is not None and p95<=.5 and bad==0 and self.unscorable==0))


class Metrics:
    def __init__(self):self.cells={p:{g:Cell() for g in GROUPS} for p in POLICIES}
    def add(self,values,codes,reference,clean,group):
        for i,p in enumerate(POLICIES):
            self.cells[p]['all'].add(values[i],codes[i],reference,clean)
            self.cells[p][group].add(values[i],codes[i],reference,clean)
    def report(self):return {p:{g:self.cells[p][g].report() for g in GROUPS} for p in POLICIES}
