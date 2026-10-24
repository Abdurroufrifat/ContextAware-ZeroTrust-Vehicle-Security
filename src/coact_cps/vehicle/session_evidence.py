"""Inactive gate: upstream authorization and physical witness independence are assumptions.

No capability issuance, vehicle control, or sensor truth is provided here. Access
and serialization of this object are trusted. Positive witnesses in tests are synthetic.
"""
from dataclasses import asdict,dataclass,field
import hashlib
import math
from pathlib import Path
from threading import RLock
from .policy import VehicleRequest,digest
from .witness_certificate import WitnessConfig,WitnessVerifier,integer

def _finite_number(value):
    if type(value) not in (int,float):return False
    try:return math.isfinite(value)
    except (OverflowError,ValueError):return False

class RetainedAlert:
    def __init__(self):self._retained=False
    @property
    def retained(self):return self._retained
    def observe(self,alarm):
        if type(alarm) is not bool:raise ValueError('boolean_alarm_required')
        self._retained=self._retained or alarm
        return self._retained

@dataclass(frozen=True)
class EvidenceDecision:
    permit:bool
    reason:str
    state:str
    retained_alarm:bool
    current_alarm:bool
    score_known:bool
    epoch:int
    confirmations:int
    witness_reason:str

@dataclass
class _State:
    state:str='UNADMITTED'
    alert:RetainedAlert=field(default_factory=RetainedAlert)
    epoch:int=0
    last_end:int=-1
    last_now:int=-1
    scope:str=''
    confirmations:int=0
    first_sample:int=-1
    first_motion_end:int=-1
    last_certificate:str=''
    last_sample:int=-1

class SessionEvidenceGate:
    """One evaluation per fresh motion interval. Repeated PEP checks are not this API.

    Caller supplies a trusted local score and verified upstream base result.
    Issuer compromise, durable state, and capability enforcement are outside this wrapper.
    """
    @property
    def threshold(self):return self._threshold
    @property
    def policy_digest(self):return self._policy_digest
    def __init__(self,threshold,witness_public_keys=None,witness_config=None):
        if not _finite_number(threshold) or threshold<0:
            raise ValueError('finite_nonnegative_threshold_required')
        self._threshold=float(threshold)
        config=WitnessConfig() if witness_config is None else witness_config
        if not isinstance(config,WitnessConfig):raise ValueError('typed_witness_config_required')
        self._verifier=None if witness_public_keys is None else WitnessVerifier(witness_public_keys,config)
        names=('session_evidence.py','witness_certificate.py','policy.py')
        code={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in names}
        self._policy_digest=digest(dict(version='coact-session-evidence-v1',threshold=self.threshold,
            witness_config=asdict(config),witness_keys={} if self._verifier is None else self._verifier.key_fingerprints,
            confirmation_epochs=5,confirmation_span_ms=1000,continuous_witness=True,code=code))
        self._states={};self._lock=RLock()
    def _state(self,vehicle):
        if vehicle not in self._states:
            if len(self._states)>=256:raise ValueError('session_capacity')
            self._states[vehicle]=_State()
        return self._states[vehicle]
    @staticmethod
    def _reset_progress(s):s.confirmations=0;s.first_sample=-1;s.first_motion_end=-1
    def _quarantine(self,vehicle,s):
        if s.state!='REVOKED':
            if s.state!='QUARANTINED':
                s.epoch+=1
                if self._verifier:self._verifier.invalidate(vehicle)
            s.state='QUARANTINED'
        self._reset_progress(s)
    def challenge(self,request,now_ms):
        if not isinstance(request,VehicleRequest):raise ValueError('typed_request_required')
        with self._lock:
            s=self._state(request.vehicle_id)
            if self._verifier is None:raise ValueError('witness_unconfigured')
            return self._verifier.challenge(request,self.policy_digest,s.epoch,now_ms)
    def revoke(self,vehicle):
        if type(vehicle) is not str or not vehicle:raise ValueError('vehicle_required')
        with self._lock:
            s=self._state(vehicle);s.state='REVOKED';s.epoch+=1;self._reset_progress(s)
            if self._verifier:self._verifier.invalidate(vehicle)
    def evaluate(self,request,*,now_ms,start_ms,end_ms,gnss,wheel,score,base_permit,witness=None,quality=True):
        known=_finite_number(score)
        alarm=bool(known and score>self.threshold)
        if not isinstance(request,VehicleRequest):
            return EvidenceDecision(False,'invalid_request','UNADMITTED',False,alarm,known,-1,0,'not_checked')
        with self._lock:
            try:s=self._state(request.vehicle_id)
            except ValueError:return EvidenceDecision(False,'session_capacity','UNADMITTED',False,alarm,known,-1,0,'not_checked')
            def decision(permit,reason,witness_reason='not_checked'):
                return EvidenceDecision(permit,reason,s.state,s.alert.retained,alarm,known,s.epoch,s.confirmations,witness_reason)
            if s.state=='REVOKED':return decision(False,'revoked')
            # A trusted local alarm is retained even if other inputs are unusable.
            if alarm:
                s.alert.observe(True);self._quarantine(request.vehicle_id,s)
            if (not all(integer(v) for v in (now_ms,start_ms,end_ms)) or not start_ms<end_ms<=now_ms
                or end_ms-start_ms>500 or now_ms-end_ms>500 or end_ms<=s.last_end or now_ms<s.last_now):
                self._quarantine(request.vehicle_id,s);return decision(False,'clock_invalid')
            gap=s.last_end>=0 and start_ms!=s.last_end
            scope=digest({k:v for k,v in asdict(request).items() if k not in ('request_id','nonce')})
            if gap or (s.scope and s.scope!=scope):
                self._reset_progress(s)
                if s.state=='ADMITTED':self._quarantine(request.vehicle_id,s)
            s.last_end=end_ms;s.last_now=now_ms;s.scope=scope
            if type(quality) is not bool or not quality:
                if s.state=='ADMITTED':self._quarantine(request.vehicle_id,s)
                self._reset_progress(s);return decision(False,'motion_quality')
            if not known:
                if s.state=='ADMITTED':self._quarantine(request.vehicle_id,s)
                self._reset_progress(s);return decision(False,'score_unknown')
            s.alert.observe(alarm)
            if alarm:self._quarantine(request.vehicle_id,s);return decision(False,'score_alarm')
            if type(base_permit) is not bool or not base_permit:
                if s.state=='ADMITTED':self._quarantine(request.vehicle_id,s)
                self._reset_progress(s);return decision(False,'base_denied')
            if self._verifier is None:self._reset_progress(s);return decision(False,'witness_unconfigured')
            check=self._verifier.check(witness,request=request,policy=self.policy_digest,epoch=s.epoch,
                now=now_ms,motion_ms=end_ms,gnss=gnss,wheel=wheel)
            if not check.accepted or check.certificate_id==s.last_certificate or check.sample_ms<=s.last_sample:
                if s.state=='ADMITTED':self._quarantine(request.vehicle_id,s)
                self._reset_progress(s)
                return decision(False,'witness_rejected',check.reason if not check.accepted else 'certificate_reused')
            s.last_certificate=check.certificate_id;s.last_sample=check.sample_ms
            if s.state=='QUARANTINED':
                if not s.confirmations:
                    s.first_sample=check.sample_ms;s.first_motion_end=end_ms
                s.confirmations+=1
                if s.confirmations>=5 and check.sample_ms-s.first_sample>=1000 and end_ms-s.first_motion_end>=1000:
                    s.alert=RetainedAlert();s.state='ADMITTED';s.epoch+=1
                    self._verifier.invalidate(request.vehicle_id);self._reset_progress(s)
                    return decision(False,'recovery_completed_new_evidence_required',check.reason)
                return decision(False,'awaiting_recovery',check.reason)
            s.state='ADMITTED'
            return decision(True,'permit',check.reason)
