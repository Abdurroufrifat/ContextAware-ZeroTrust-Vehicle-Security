"""Offline integration component; no physical-speed or vehicle authority is granted."""
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_evidence_auth'))
import auth_guard as auth


@dataclass(frozen=True)
class SourcePolicy:
    source_id:str
    channel:str
    role:str
    domain_id:str
    uncertainty_mmps:int|None
    valid_from_us:int
    valid_until_us:int
    research_assumption:bool=False


@dataclass(frozen=True)
class Packet:
    wire:str|bytes
    received_us:int


def decision(selection,reason,candidate_status='not_checked',anchor_status='not_checked',value=None,risk=None):
    return dict(selection=selection,reason=reason,research_speed_mmps=value,
                research_speed_mps=value/1000 if value is not None else None,
                candidate_status=candidate_status,anchor_status=anchor_status,
                risk_context=risk if type(risk) is str and risk in ('normal','elevated','lockdown') else None,
                research_simulation_only=True,physical_speed_authorized=False,
                measurement_truth_established=False,independent_source_trust_established=False,
                physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)


def in_interval(policy,body,now):
    return policy.valid_from_us<=body['measured_us']<=now<=policy.valid_until_us


class Gateway:
    def __init__(self,guard,policies):
        if not isinstance(guard,auth.Guard):raise ValueError('Provision an existing authenticated-report guard')
        self.guard=guard;self._roles={};sources=set();domains=set()
        for policy in tuple(policies):
            if not isinstance(policy,SourcePolicy):raise ValueError('Invalid source policy')
            if policy.role not in ('candidate','anchor') or policy.role in self._roles:raise ValueError('Duplicate/unknown role')
            if not all(auth.identifier(x) for x in (policy.source_id,policy.channel,policy.domain_id)):raise ValueError('Invalid policy identifier')
            if policy.source_id in sources or policy.domain_id in domains:raise ValueError('Sources must have distinct IDs and declared domains; independence is still unproved')
            if type(policy.research_assumption) is not bool:raise ValueError('Research assumption must be an explicit boolean')
            if not auth.unsigned(policy.valid_from_us) or not auth.unsigned(policy.valid_until_us) or policy.valid_from_us>policy.valid_until_us:raise ValueError('Invalid source-policy interval')
            if policy.role=='anchor':
                if type(policy.uncertainty_mmps) is not int or not 0<=policy.uncertainty_mmps<=500:raise ValueError('Invalid uncertainty/error budget')
            elif policy.uncertainty_mmps is not None:raise ValueError('Candidate has no protected uncertainty claim')
            if not any(rule.source_id==policy.source_id and rule.channel==policy.channel for rule in guard.rules.values()):raise ValueError('Policy source/channel absent from report allowlist')
            self._roles[policy.role]=policy;sources.add(policy.source_id);domains.add(policy.domain_id)
        if set(self._roles)!={'candidate','anchor'}:raise ValueError('One candidate and one anchor role are required')

    def _authenticate(self,packet,role,now):
        if packet is None:return None,'missing'
        if not isinstance(packet,Packet):return None,'malformed_packet'
        # Immutable wire only: a private parsed body cannot be changed by the caller
        # between MAC verification, replay commit and policy use.
        try:message=auth.parse_wire(packet.wire)
        except (ValueError,TypeError):return None,'malformed_wire'
        policy=self._roles[role];body=message['body']
        if body['source_id']!=policy.source_id or body['channel']!=policy.channel:return None,'role_mismatch'
        checked=self.guard.verify(message,now,packet.received_us)
        return (body if checked['authenticated_report'] is True else None),checked['status']

    def evaluate(self,candidate_packet,anchor_packet,now_us,risk='normal',operation='research_speed'):
        if operation!='research_speed':return decision('abstain','physical_operation_unavailable',risk=risk)
        if type(risk) is not str or risk not in ('normal','elevated','lockdown'):return decision('abstain','unsupported_risk_context')
        if risk=='lockdown':return decision('abstain','lockdown_context',risk=risk)
        if not auth.unsigned(now_us):return decision('abstain','invalid_receiver_time',risk=risk)
        candidate,cs=self._authenticate(candidate_packet,'candidate',now_us)
        anchor,as_=self._authenticate(anchor_packet,'anchor',now_us)
        if anchor is None:return decision('abstain','anchor_'+as_,cs,as_,risk=risk)
        protected=self._roles['anchor']
        if not protected.research_assumption:return decision('abstain','anchor_assumption_not_supplied',cs,as_,risk=risk)
        if not in_interval(protected,anchor,now_us):return decision('abstain','anchor_calibration_outside_interval',cs,as_,risk=risk)
        center=anchor['speed_mmps']
        if risk=='elevated':return decision('anchor','elevated_context_anchor_only',cs,as_,center,risk)
        if candidate is not None and not in_interval(self._roles['candidate'],candidate,now_us):candidate=None;cs='policy_interval_expired'
        if candidate is None:return decision('anchor','candidate_unavailable_fallback',cs,as_,center,risk)
        if abs(candidate['speed_mmps']-center)+protected.uncertainty_mmps<=500:
            return decision('candidate','research_candidate_within_budget',cs,as_,candidate['speed_mmps'],risk)
        return decision('anchor','candidate_outside_budget_fallback',cs,as_,center,risk)
