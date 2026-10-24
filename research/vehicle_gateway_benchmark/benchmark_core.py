"""Fixed synthetic protocol controls; native vehicle evidence is not generated."""
import json,math,secrets,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_policy_gateway'))
import policy_gateway as p

CASES=('nominal','candidate_bias','candidate_tamper','anchor_tamper','candidate_replay','anchor_replay','candidate_context','anchor_context','anchor_missing','signed_coordinated_bias','anchor_assumption_missing','anchor_key_revoked','risk_elevated','risk_lockdown','unsupported_operation','anchor_measurement_stale','receiver_clock_rollback','database_snapshot_rollback')
POLICIES=('unsigned_candidate_control','authenticated_candidate_control','integrated_gateway')
GROUPS=('all','before','during','after')


def phase(i,n):return 'before' if i<n//4 else 'during' if i<3*n//4 else 'after'


def trial(case,policy,root,n=128):
    if case not in CASES or policy not in POLICIES:raise ValueError('Unknown frozen case/policy')
    if type(n) is not int or n<8 or n>4096 or n%4:raise ValueError('Invalid fixture length')
    root=p.auth.safe(root);root.mkdir(parents=True,exist_ok=True)
    context=p.auth.Context('benchmark-vehicle','benchmark-session','case'+str(CASES.index(case))+'-policy'+str(POLICIES.index(policy)))
    ka=secrets.token_bytes(32);kb=secrets.token_bytes(32);state=root/'state.sqlite';backup=root/'initial.sqlite'
    p.auth.create_state(state,context)
    guard=p.auth.Guard(state,context,[p.auth.KeyRule('candidate','ka','wheel_speed',ka),p.auth.KeyRule('anchor','kb','protected_speed',kb,case!='anchor_key_revoked')])
    source_rules=[p.SourcePolicy('candidate','wheel_speed','candidate','domain-A',None,0,2**63-1),p.SourcePolicy('anchor','protected_speed','anchor','domain-B',250,0,2**63-1,case!='anchor_assumption_missing')]
    gateway=p.Gateway(guard,source_rules);shutil.copyfile(state,backup);saved=None
    for i in range(n):
        group=phase(i,n);active=group=='during';reference=5000+20*i;now=1000000+20000*i
        if active and case=='receiver_clock_rollback':now=1000000+20000*(n//4-1)-1
        def packet(role,value):
            capture='other-capture' if active and case==role+'_context' else context.capture_id
            stamp=now-100001 if role=='anchor' and active and case=='anchor_measurement_stale' else now
            body=dict(schema=1,source_id=role,key_id='ka' if role=='candidate' else 'kb',vehicle_id=context.vehicle_id,session_id=context.session_id,capture_id=capture,channel='wheel_speed' if role=='candidate' else 'protected_speed',seq=i+1,measured_us=stamp,speed_mmps=value)
            return p.Packet(json.dumps(p.auth.sign(body,ka if role=='candidate' else kb)),now)
        candidate=packet('candidate',reference+200+(3000 if active and case in ('candidate_bias','signed_coordinated_bias') else 0))
        anchor=packet('anchor',reference+250*(i%2)+(3000 if active and case=='signed_coordinated_bias' else 0))
        if i==n//4-1:saved=(candidate,anchor)
        candidate_replay=active and case in ('candidate_replay','database_snapshot_rollback')
        anchor_replay=active and case in ('anchor_replay','database_snapshot_rollback')
        if active and case=='database_snapshot_rollback' and i==n//4:
            # Deliberate attack on this temporary simulation database only.
            # Existing guard connections are closed between decisions.
            shutil.copyfile(backup,state)
        if candidate_replay:candidate=p.Packet(saved[0].wire,now)
        if anchor_replay:anchor=p.Packet(saved[1].wire,now)
        if active and case in ('candidate_tamper','anchor_tamper'):
            which=candidate if case=='candidate_tamper' else anchor
            message=json.loads(which.wire);message['body']['speed_mmps']+=3000
            changed=p.Packet(json.dumps(message),now)
            if case=='candidate_tamper':candidate=changed
            else:anchor=changed
        if active and case=='anchor_missing':anchor=None
        risk='elevated' if active and case=='risk_elevated' else 'lockdown' if active and case=='risk_lockdown' else 'normal'
        operation='drive' if active and case=='unsupported_operation' else 'research_speed'
        start=time.perf_counter_ns()
        if policy=='integrated_gateway':result=gateway.evaluate(candidate,anchor,now,risk,operation)
        elif policy=='authenticated_candidate_control':
            message=p.auth.parse_wire(candidate.wire);checked=guard.verify(message,now,candidate.received_us)
            value=message['body']['speed_mmps'] if checked['authenticated_report'] else None
            result=p.decision('candidate' if value is not None else 'abstain','authentication_only_control',checked['status'],value=value,risk=risk)
        else:
            value=json.loads(candidate.wire)['body']['speed_mmps']
            result=p.decision('candidate','unsigned_value_control',value=value,risk=risk)
        latency=time.perf_counter_ns()-start;value=result['research_speed_mmps']
        replay_used=bool(value is not None and ((candidate_replay and result['selection']=='candidate') or
                           (policy=='integrated_gateway' and anchor_replay and result['anchor_status']=='authenticated')))
        yield dict(case=case,policy=policy,index=i,phase=group,reference_mmps=reference,output_mmps=value,
                   selection=result['selection'],candidate_status=result['candidate_status'],anchor_status=result['anchor_status'],
                   reason=result['reason'],latency_ns=latency,replayed_evidence_used=replay_used,
                   restricted_context_output=bool(value is not None and active and case in ('risk_lockdown','unsupported_operation')),
                   physical_speed_authorized=result['physical_speed_authorized'])


def quantile(values,q):
    if not values:return None
    ordered=sorted(values);at=(len(ordered)-1)*q;low=math.floor(at)
    return ordered[low]+(ordered[min(low+1,len(ordered)-1)]-ordered[low])*(at-low)


class Cell:
    def __init__(self):self.n=0;self.errors=[];self.latency=[];self.selection=dict(candidate=0,anchor=0,abstain=0);self.replays=0;self.restricted=0
    def add(self,event):
        self.n+=1;self.latency.append(event['latency_ns']);self.selection[event['selection']]+=1
        if event['output_mmps'] is not None:self.errors.append(abs(event['output_mmps']-event['reference_mmps']))
        self.replays+=event['replayed_evidence_used'];self.restricted+=event['restricted_context_output']
    def report(self):
        k=len(self.errors);coverage=100*k/self.n if self.n else None;p95=quantile(self.errors,.95);bad=sum(x>2000 for x in self.errors)
        return dict(requests=self.n,accepted=k,coverage_percent=coverage,mae_synthetic_error_mps=math.fsum(self.errors)/k/1000 if k else None,
                    p95_synthetic_error_mps=p95/1000 if p95 is not None else None,max_synthetic_error_mps=max(self.errors)/1000 if k else None,
                    accepted_synthetic_error_gt2_count=bad,selection_counts=self.selection.copy(),replayed_evidence_used_count=self.replays,
                    restricted_context_output_count=self.restricted,latency_samples=len(self.latency),
                    median_decision_latency_ms=quantile(self.latency,.5)/1000000 if self.latency else None,
                    p95_decision_latency_ms=quantile(self.latency,.95)/1000000 if self.latency else None,
                    max_decision_latency_ms=max(self.latency)/1000000 if self.latency else None,
                    diagnostic_gates_pass=bool(self.n and coverage>=95 and p95 is not None and p95<=500 and bad==0))


class Metrics:
    def __init__(self):self.cells={g:Cell() for g in GROUPS}
    def add(self,event):
        self.cells['all'].add(event);self.cells[event['phase']].add(event)
    def report(self):return {g:cell.report() for g,cell in self.cells.items()}
