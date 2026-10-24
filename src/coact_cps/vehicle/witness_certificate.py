"""Request-bound external speed evidence. Signatures do not establish physical truth."""
from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import asdict, dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .policy import VehicleRequest, canonical_json, digest, strict_loads

DOMAIN = b'COACT-CPS-WITNESS-12\x00'


def integer(x):
    return type(x) is int and 0 <= x <= 2**53 - 1


def number(x, low, high):
    return type(x) in (int, float) and math.isfinite(x) and low <= x <= high


@dataclass(frozen=True)
class WitnessConfig:
    maximum_age_ms: int = 250
    challenge_lifetime_ms: int = 1000
    maximum_uncertainty_mps: float = .25
    agreement_mps: float = .5
    maximum_acceleration_mps2: float = 8.

    def __post_init__(self):
        if any(not integer(x) or not 1 <= x <= 10000 for x in
               (self.maximum_age_ms, self.challenge_lifetime_ms)):
            raise ValueError('invalid_witness_time_bounds')
        if not all(number(x, 0, 30) for x in
                   (self.maximum_uncertainty_mps,self.agreement_mps,self.maximum_acceleration_mps2)):
            raise ValueError('invalid_witness_numeric_bounds')


@dataclass(frozen=True)
class WitnessCheck:
    accepted: bool
    reason: str
    certificate_id: str = ''
    sample_ms: int = -1
    sequence: int = -1
    speed_mps: float = 0.
    comparison_bound: float = 0.
    physical_bound: float = 0.


class WitnessVerifier:
    """One outstanding challenge per vehicle; caller serializes access.

    Exact same-certificate rechecks permit PDP/PEP use, not new confirmations.
    Restart replay protection is challenge entropy, not persistent counters.
    """
    def __init__(self, public_keys, config=None):
        if type(public_keys) is not dict or not 1 <= len(public_keys) <= 32:
            raise ValueError('bounded_witness_keyring_required')
        self.config = config or WitnessConfig()
        if not isinstance(self.config, WitnessConfig):
            raise ValueError('typed_witness_config_required')
        self._keys = {}
        for issuer, key in public_keys.items():
            if type(issuer) is not str or not 1 <= len(issuer) <= 128 or type(key) is not bytes or len(key) != 32:
                raise ValueError('invalid_witness_public_key')
            self._keys[issuer] = Ed25519PublicKey.from_public_bytes(key)
        self.key_fingerprints = {k: hashlib.sha256(v).hexdigest() for k,v in public_keys.items()}
        self._pending = {}
        self._watermarks = {}
        self._clock = -1

    def challenge(self, request, policy, epoch, now):
        if not isinstance(request,VehicleRequest) or type(policy) is not str or not policy or not integer(epoch) or not integer(now):
            raise ValueError('invalid_witness_challenge_input')
        if now < self._clock:
            raise ValueError('witness_clock_rollback')
        self._clock = now
        # Prune expired challenges; do not evict a live one silently.
        self._pending = {k:v for k,v in self._pending.items() if v['expires_ms'] >= now}
        if request.vehicle_id not in self._pending and len(self._pending) >= 256:
            raise ValueError('witness_challenge_capacity')
        c = dict(challenge=secrets.token_hex(32), vehicle_id=request.vehicle_id,
                 request_digest=digest(asdict(request)), policy_digest=policy,
                 epoch=epoch, issued_ms=now, expires_ms=now+self.config.challenge_lifetime_ms)
        self._pending[request.vehicle_id] = c
        return dict(c)

    def invalidate(self, vehicle):
        self._pending.pop(vehicle,None)

    def check(self, wire, *, request, policy, epoch, now, motion_ms, gnss, wheel):
        try:
            if not isinstance(request,VehicleRequest) or not integer(epoch) or not integer(now) or not integer(motion_ms):
                raise ValueError('invalid_witness_check_input')
            if now < self._clock:
                raise ValueError('witness_clock_rollback')
            self._clock = now
            if type(wire) is not str or len(wire) > 8192:
                raise ValueError('missing_or_oversized_witness')
            packet = strict_loads(wire)
            if type(packet) is not dict or set(packet) != {'body','signature'} or type(packet['body']) is not dict:
                raise ValueError('invalid_witness_envelope')
            b = packet['body']
            required = {'schema','issuer','challenge','vehicle_id','request_digest','policy_digest',
                        'epoch','issued_ms','expires_ms','sample_ms','speed_mps','uncertainty_mps','sequence'}
            if set(b) != required or b['schema'] != 'coact-speed-witness-v1':
                raise ValueError('invalid_witness_schema')
            if type(b['issuer']) is not str or b['issuer'] not in self._keys:
                raise ValueError('unknown_witness_issuer')
            sig = packet['signature']
            if type(sig) is not str or len(sig) != 128:
                raise ValueError('invalid_witness_signature')
            self._keys[b['issuer']].verify(bytes.fromhex(sig), DOMAIN+canonical_json(b).encode())
            expected = self._pending.get(request.vehicle_id)
            if expected is None or any(type(b[k]) is not type(v) or b[k] != v for k,v in expected.items()):
                raise ValueError('witness_challenge_mismatch')
            if (b['request_digest'] != digest(asdict(request)) or b['policy_digest'] != policy or
                    b['epoch'] != epoch or not b['issued_ms'] <= now <= b['expires_ms']):
                raise ValueError('witness_request_policy_epoch_or_expiry')
            if not integer(b['sample_ms']) or not integer(b['sequence']):
                raise ValueError('invalid_witness_time_or_sequence')
            if not b['issued_ms'] <= b['sample_ms'] <= motion_ms <= now:
                raise ValueError('witness_time_binding_failed')
            age = now - b['sample_ms']
            cfg = self.config
            if age > cfg.maximum_age_ms:
                raise ValueError('witness_expired')
            if not number(b['speed_mps'],0,100) or not number(b['uncertainty_mps'],0,cfg.maximum_uncertainty_mps):
                raise ValueError('invalid_witness_speed_or_uncertainty')
            if not number(gnss,-30,100) or not number(wheel,0,100):
                raise ValueError('invalid_witness_comparison_speed')
            identity = hashlib.sha256(canonical_json(packet).encode()).hexdigest()
            key = request.vehicle_id,b['issuer']
            previous = self._watermarks.get(key)
            if previous is not None and identity != previous[2] and (
                    b['sequence'] <= previous[0] or b['sample_ms'] <= previous[1]):
                raise ValueError('witness_sequence_or_sample_replay')
            # A witness cannot reduce the enrolled error guarantee by claiming a
            # smaller uncertainty. The trusted cap governs the physical bound.
            uncertainty = cfg.maximum_uncertainty_mps + cfg.maximum_acceleration_mps2*age/1000
            bound = cfg.agreement_mps + uncertainty
            if max(abs(gnss-b['speed_mps']),abs(wheel-b['speed_mps'])) > bound:
                raise ValueError('witness_motion_disagreement')
            self._watermarks[key] = b['sequence'],b['sample_ms'],identity
            return WitnessCheck(True,'witness_verified',identity,b['sample_ms'],b['sequence'],
                                b['speed_mps'],bound,cfg.agreement_mps+2*uncertainty)
        except (ValueError,TypeError,KeyError,OverflowError,RecursionError,InvalidSignature) as exc:
            return WitnessCheck(False,str(exc) or 'witness_signature_failed')
