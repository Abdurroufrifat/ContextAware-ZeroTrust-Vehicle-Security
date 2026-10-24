from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest
from run_vehicle_native_stage6 import packets
from test_vehicle_witness_certificate12 import signed
import test_vehicle_window_recovery as legacy


def make_engine():
    from coact_cps.vehicle.witness_recovery_policy import WitnessRecoveryPolicyEngine
    c=legacy.WindowRecoveryTests().config()
    keys={s:bytes([i+1])*32 for i,s in enumerate(SOURCES)}
    key=Ed25519PrivateKey.generate()
    engine=WitnessRecoveryPolicyEngine(keys,b'c'*32,
        {('native-reader','native-ego','vehicle.telemetry.longitudinal','read')},
        ModelConfig(**c['model']),c['thresholds']['candidate'],
        c['session_config']['stage7_horizon'],
        {'roadside-test':key.public_key().public_bytes_raw()})
    return engine,keys,key


def request(f, suffix=''):
    return VehicleRequest(f'r{f.frame}{suffix}',f'n{f.frame}{suffix}','native-reader',
                          'native-ego','vehicle.telemetry.longitudinal','read')


def drive(frames, *, missing=(), revoked=(), compromised=False, repeat=False):
    e,keys,key=make_engine();rows=[]
    for f in frames:
        q=request(f);c=e.challenge(q,f.time_ms)
        speed=f.wheel_speed if compromised else 10.
        wire=None if f.frame in missing else signed(key,c,speed=speed,sample=f.time_ms,sequence=f.frame+1)
        d=e.evaluate(q,packets(f,4*f.frame+1,keys,revoked=f.frame in revoked),f.time_ms,wire)
        diag=e.diagnostics('native-ego');rows.append((d,diag))
        if repeat and diag['witness_confirmation_epochs']:
            before=diag['witness_confirmation_epochs']
            e.evaluate(q,packets(f,4*f.frame+2,keys),f.time_ms,wire)
            assert e.diagnostics('native-ego')['witness_confirmation_epochs']==before
    return e,keys,key,rows


def test_honest_stream_allows_and_no_witness_fails_closed():
    fs=legacy.WindowRecoveryTests().frames()[:180]
    *_,rows=drive(fs)
    assert rows[-1][0].outcome=='ALLOW'
    *_,rows=drive(fs,missing=range(180))
    assert all(d.outcome!='ALLOW' for d,x in rows)


def test_initial_horizon_warmup_is_preserved():
    *_,rows=drive(legacy.WindowRecoveryTests().frames()[:80])
    assert any(not x['horizon_ready'] for d,x in rows)
    assert all(d.outcome!='ALLOW' for d,x in rows if not x['horizon_ready'])


def test_coherent_forgery_rejected_with_honest_independent_witness():
    fs=[replace(f,gnss_speed=f.gnss_speed+5,wheel_speed=f.wheel_speed+5)
        if f.frame>=100 else f for f in legacy.WindowRecoveryTests().frames()[:350]]
    *_,rows=drive(fs)
    assert all(d.outcome!='ALLOW' for d,x in rows[100:])


def test_honest_recovery_and_distinct_epoch_quorum():
    fs=[replace(f,gnss_speed=f.gnss_speed+5,wheel_speed=f.wheel_speed+5)
        if 100<=f.frame<160 else f for f in legacy.WindowRecoveryTests().frames()[:300]]
    *_,rows=drive(fs,repeat=True)
    releases=[i for i,(d,x) in enumerate(rows) if x['witness_release']]
    assert len(releases)==1 and releases[0]>=165
    i=releases[0]
    assert [d.outcome for d,x in rows[i:i+4]]==['CHALLENGE']*3+['ALLOW']
    assert rows[-1][0].outcome=='ALLOW'


def test_revoked_identity_cannot_release():
    fs=legacy.WindowRecoveryTests().frames()[:180]
    *_,rows=drive(fs,missing=range(100),revoked=range(100,180))
    assert all(d.outcome=='DENY' and not x['witness_release'] for d,x in rows)


def test_pep_rechecks_witness_and_identity_and_one_use():
    fs=legacy.WindowRecoveryTests().frames()[:180]
    e,keys,key,rows=drive(fs)
    f=fs[-1];q=rows[-1][0].request;cap=e.issue_capability(rows[-1][0])
    use=e.consume_capability(cap,q,packets(f,4*f.frame+2,keys,revoked=True),f.time_ms)
    assert not use.authorized
    # New honest epochs restore quorum after revocation; old capability stays request-bound.
    e,keys,key,rows=drive(fs)
    q=rows[-1][0].request;cap=e.issue_capability(rows[-1][0])
    assert e.consume_capability(cap,q,packets(f,4*f.frame+2,keys),f.time_ms).authorized
    assert not e.consume_capability(cap,q,packets(f,4*f.frame+3,keys),f.time_ms).authorized


def test_use_after_witness_age_limit_denied_even_before_capability_expiry():
    fs=legacy.WindowRecoveryTests().frames()[:180]
    e,keys,key,rows=drive(fs)
    f=legacy.WindowRecoveryTests().frames()[180];q=request(f,'aged');c=e.challenge(q,f.time_ms)
    # Source time is current, witness sample has to be at/after challenge, so delay
    # initial appraisal to use most of witness lifetime first.
    cert=signed(key,c,sample=f.time_ms,sequence=999)
    d=e.evaluate(q,packets(f,4*f.frame+2,keys),f.time_ms+200,cert)
    assert d.outcome=='ALLOW'
    cap=e.issue_capability(d)
    assert not e.consume_capability(cap,q,packets(f,4*f.frame+3,keys),f.time_ms+251).authorized


def test_compromised_witness_exposes_trust_assumption():
    fs=[replace(f,gnss_speed=f.gnss_speed+5,wheel_speed=f.wheel_speed+5)
        for f in legacy.WindowRecoveryTests().frames()[:180]]
    *_,rows=drive(fs,compromised=True)
    assert rows[-1][0].outcome=='ALLOW'


def test_new_challenge_invalidates_old_certificate():
    e,keys,key=make_engine();f=legacy.WindowRecoveryTests().frames()[0];q=request(f)
    c=e.challenge(q,f.time_ms);cert=signed(key,c,sample=f.time_ms)
    e.challenge(q,f.time_ms)
    assert e.evaluate(q,packets(f,1,keys),f.time_ms,cert).outcome=='DENY'
