import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from coact_cps.vehicle.policy import VehicleRequest, canonical_json

DOMAIN = b'COACT-CPS-WITNESS-12\x00'


def signed(key, challenge_body, *, speed=10.0, sample=1000, sequence=1, uncertainty=.1, **extra):
    body = dict(challenge_body, schema='coact-speed-witness-v1', issuer='roadside-test',
                sample_ms=sample, speed_mps=speed, uncertainty_mps=uncertainty,
                sequence=sequence)
    body.update(extra)
    sig = key.sign(DOMAIN + canonical_json(body).encode()).hex()
    return canonical_json(dict(body=body, signature=sig))


@pytest.fixture
def setup():
    from coact_cps.vehicle.witness_certificate import WitnessVerifier
    key = Ed25519PrivateKey.generate()
    v = WitnessVerifier({'roadside-test': key.public_key().public_bytes_raw()})
    r = VehicleRequest('r1','n1','native-reader','native-ego','vehicle.telemetry.longitudinal','read')
    c = v.challenge(r,'policy',0,1000)
    return v,key,r,c


def check(setup, wire, **overrides):
    v,key,r,c = setup
    kw=dict(request=r, policy='policy', epoch=0, now=1000, motion_ms=1000,
            gnss=10., wheel=10.)
    kw.update(overrides)
    return v.check(wire,**kw)


def test_valid_certificate_can_be_rechecked_for_own_request(setup):
    v,key,r,c=setup
    wire=signed(key,c)
    a=check(setup,wire);b=check(setup,wire,now=1001)
    assert a.accepted and b.accepted
    assert a.certificate_id==b.certificate_id and a.physical_bound==pytest.approx(1.)


def test_reported_uncertainty_cannot_shrink_enrolled_error_bound(setup):
    v,key,r,c=setup
    a=check(setup,signed(key,c,uncertainty=0),gnss=10.7,wheel=10.7)
    assert a.accepted and a.comparison_bound==pytest.approx(.75)
    assert a.physical_bound==pytest.approx(1.)


@pytest.mark.parametrize('change',[{'vehicle_id':'wrong'},{'request_digest':'wrong'},
    {'policy_digest':'other'},{'epoch':1},{'issuer':'unknown'}, {'speed_mps':True},
    {'speed_mps':101},{'uncertainty_mps':.251},{'sample_ms':1001},{'sequence':True},
    {'sequence':-1},{'extra':'x'},{'schema':'other'},{'challenge':'x'}])
def test_signed_invalid_claims_rejected(setup,change):
    v,key,r,c=setup
    assert not check(setup,signed(key,c,**change)).accepted


def test_tampered_and_wrong_signature_rejected(setup):
    v,key,r,c=setup
    p=json.loads(signed(key,c));p['body']['speed_mps']=11
    assert not check(setup,canonical_json(p)).accepted
    assert not check(setup,signed(Ed25519PrivateKey.generate(),c)).accepted


@pytest.mark.parametrize('wire',['{}','[]','null','{"body":{},"body":{},"signature":"x"}',
    '{"body":{"speed_mps":NaN},"signature":"x"}', 'x'*20000])
def test_malformed_evidence_fails_closed(setup,wire):
    assert not check(setup,wire).accepted


def test_expiry_and_future_motion_and_clock_rollback(setup):
    v,key,r,c=setup
    wire=signed(key,c)
    assert not check(setup,wire,now=999,motion_ms=999).accepted
    assert check(setup,wire,now=1100,motion_ms=1100).accepted
    assert not check(setup,wire,now=1099,motion_ms=1099).accepted
    assert not check(setup,wire,now=1251,motion_ms=1251).accepted


def test_challenge_supersession_and_sequence_sample_replay(setup):
    v,key,r,c=setup
    old=signed(key,c)
    assert check(setup,old).accepted
    nc=v.challenge(r,'policy',0,1200)
    assert not check(setup,old,now=1200,motion_ms=1200).accepted
    assert not check(setup,signed(key,nc,sample=1200),now=1200,motion_ms=1200).accepted
    assert check(setup,signed(key,nc,sample=1200,sequence=2),now=1200,motion_ms=1200).accepted


def test_physical_disagreement_rejected_with_authentic_signature(setup):
    v,key,r,c=setup
    a=check(setup,signed(key,c),gnss=15,wheel=15)
    assert not a.accepted and a.reason=='witness_motion_disagreement'


def test_compromised_witness_is_explicit_counterexample(setup):
    v,key,r,c=setup
    assert check(setup,signed(key,c,speed=15),gnss=15,wheel=15).accepted


def test_configuration_is_validated_and_keys_copied():
    from coact_cps.vehicle.witness_certificate import WitnessConfig, WitnessVerifier
    for kw in [dict(maximum_age_ms=0),dict(maximum_acceleration_mps2=float('nan')),
               dict(agreement_mps=-1),dict(maximum_uncertainty_mps=True)]:
        with pytest.raises(ValueError):WitnessConfig(**kw)
    with pytest.raises(ValueError):WitnessVerifier({'bad':b'bad'})
