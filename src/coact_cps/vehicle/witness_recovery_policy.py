"""Conditional recovery and continuous grant gate using an external speed witness."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .integrity_policy import IntegrityPolicyEngine
from .integrity_window import HorizonMonitor
from .native_detectors import Observer
from .policy import VehicleRequest, digest
from .witness_certificate import WitnessCheck, WitnessVerifier


@dataclass
class _WitnessState:
    epoch: int = 0
    count: int = 0
    start: int = -1
    last_end: int = -1
    last_certificate: str = ''
    wire: str | None = None
    request_digest: str = ''
    check: WitnessCheck = WitnessCheck(False,'witness_not_checked')
    released: bool = False
    enrolled_by_witness: bool = False


class WitnessRecoveryPolicyEngine(IntegrityPolicyEngine):
    def __init__(self, source_keys, capability_key, permissions, model_config,
                 thresholds, horizon_config, witness_public_keys, witness_config=None):
        super().__init__(source_keys,capability_key,permissions,model_config,
                         thresholds,'horizon_dual',horizon_config)
        self._verifier = WitnessVerifier(witness_public_keys,witness_config)
        self._witness_states = {}
        self._current_request = None
        self._current_now = -1
        self._policy_digest = digest(dict(version='coact-external-witness-recovery-v1',
            inherited=self.policy_digest, witness_config=asdict(self._verifier.config),
            witness_keys=self._verifier.key_fingerprints,
            confirmation_epochs=5, confirmation_span_ms=1000, release_quorum=3,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            verifier_sha256=hashlib.sha256(Path(__file__).with_name('witness_certificate.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()))

    def _witness(self, vehicle):
        if vehicle not in self._witness_states:
            self._witness_states[vehicle] = _WitnessState()
        return self._witness_states[vehicle]

    def challenge(self, request, now_ms):
        if not isinstance(request,VehicleRequest):
            raise ValueError('typed_request_required')
        with self._lock:
            state = self._witness(request.vehicle_id)
            state.wire = None
            state.request_digest = digest(asdict(request))
            return self._verifier.challenge(request,self.policy_digest,state.epoch,now_ms)

    def _enter(self, vehicle):
        extension = self._extension(vehicle)
        state = self._witness(vehicle)
        if not extension.quarantined:
            extension.quarantined = True
            state.epoch += 1
            self._verifier.invalidate(vehicle)
        state.count = 0
        state.start = -1
        state.last_certificate = ''

    def _motion(self, vehicle, bodies):
        extension = self._extension(vehicle)
        primary = self._states[vehicle]
        state = self._witness(vehicle)
        end = bodies['imu']['signals']['interval_end_ms']
        start = bodies['imu']['signals']['interval_start_ms']
        g = float(bodies['gnss']['signals']['speed_mps'])
        w = float(bodies['wheel']['signals']['speed_mps'])
        check = self._verifier.check(state.wire,request=self._current_request,
            policy=self.policy_digest,epoch=state.epoch,now=self._current_now,motion_ms=end,
            gnss=g-self._native_model.gnss_offset,wheel=w)
        state.check = check
        distinct = end != state.last_end
        previous_x = None if primary.observer.x is None else primary.observer.x.copy()
        previous_p = None if primary.observer.p is None else primary.observer.p.copy()
        previous_updates = primary.updates
        alarm,cold,z,statistic = super()._motion(vehicle,bodies)
        if not check.accepted:
            self._enter(vehicle)
        elif alarm and not extension.quarantined:
            self._enter(vehicle)
            # This certificate belonged to the previous epoch; new evidence is needed.
            check = WitnessCheck(False,'witness_new_quarantine_epoch')
            state.check = check
        if extension.quarantined:
            # Never learn measurement corrections while quarantined.
            primary.observer.x = previous_x
            primary.observer.p = previous_p
            primary.updates = previous_updates
            if check.accepted and distinct:
                if state.count and start != state.last_end:
                    state.count = 0
                    state.start = -1
                if state.count == 0:
                    state.start = end
                if check.certificate_id != state.last_certificate:
                    state.count += 1
                    state.last_certificate = check.certificate_id
                if state.count >= 5 and end-state.start >= 1000:
                    primary.observer = Observer(self._native_model)
                    primary.observer.initialize(g,w,end)
                    for detector in primary.statistics.values():
                        detector.reset()
                    primary.quorum = 0
                    extension.monitor = HorizonMonitor(self._native_model.imu_offset,**self._horizon_config)
                    extension.shadow = Observer(self._native_model)
                    extension.shadow.initialize(g,w,end)
                    extension.quarantined = False
                    extension.released = True
                    extension.recovery_resets += 1
                    state.released = True
                    state.enrolled_by_witness = True
                    state.count = 0
                    state.start = -1
                    state.epoch += 1
                    self._verifier.invalidate(vehicle)
                    alarm,cold = False,True
                else:
                    alarm,cold = True,True
            else:
                alarm,cold = True,True
            if extension.quarantined:
                primary.quorum = 0
        if distinct:
            state.last_end = end
        # A reset clears the historical horizon. Independent fresh witness checks
        # continue, while the new horizon rebuilds; primary three-epoch quorum stays.
        if not extension.quarantined and check.accepted and state.enrolled_by_witness:
            cold = primary.quorum < 3
        primary.alarm = alarm
        primary.cache_result = alarm,cold,z,statistic
        return primary.cache_result

    def evaluate(self, request, frames, now_ms, witness=None):
        if not isinstance(request,VehicleRequest):
            return super().evaluate(request,frames,now_ms)
        with self._lock:
            state = self._witness(request.vehicle_id)
            state.released = False
            state.check = WitnessCheck(False,'witness_not_checked')
            request_hash = digest(asdict(request))
            if state.request_digest != request_hash:
                state.wire = None
            state.request_digest = request_hash
            if witness is not None:
                state.wire = witness
            self._current_request,self._current_now = request,now_ms
            decision = super().evaluate(request,frames,now_ms)
            if decision.outcome == 'DENY':
                state.count = 0 if not state.check.accepted else state.count
                # Identity/source failures happen before _motion; discard progress.
                if state.check.reason == 'witness_not_checked':
                    state.count = 0
                    state.start = -1
                    self._enter(request.vehicle_id)
            return decision

    def diagnostics(self, vehicle):
        with self._lock:
            out = super().diagnostics(vehicle)
            state = self._witness_states.get(vehicle,_WitnessState())
            primary = self._states.get(vehicle)
            out.update(witness_epoch=state.epoch,witness_confirmation_epochs=state.count,
                witness_confirmation_start_ms=state.start,witness_release=state.released,
                witness_check=asdict(state.check), motion_quorum=primary.quorum if primary else 0)
            return out
