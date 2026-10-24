"""Synthetic protocol fixtures only; signatures do not establish physical truth."""
from dataclasses import replace
import json
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from coact_cps.vehicle.policy import VehicleRequest,canonical_json


def setup_gate():
    from coact_cps.vehicle.session_evidence import SessionEvidenceGate
    key=Ed25519PrivateKey.generate()
    return SessionEvidenceGate(2.,{'test-witness':key.public_key().public_bytes_raw()}),key,VehicleRequest('r','nonce','reader','vehicle','telemetry','read')


def wire(g,key,r,t,seq,**changes):
    b=dict(g.challenge(r,t),schema='coact-speed-witness-v1',issuer='test-witness',sample_ms=t,speed_mps=10.,uncertainty_mps=.1,sequence=seq)
    b.update(changes)
    return canonical_json(dict(body=b,signature=key.sign(b'COACT-CPS-WITNESS-12\x00'+canonical_json(b).encode()).hex()))


def evaluate(g,r,t,score=0.,**kw):
    v=dict(now_ms=t,start_ms=t-200,end_ms=t,gnss=10.,wheel=10.,score=score,base_permit=True);v.update(kw)
    return g.evaluate(r,**v)


def test_retained_alert_survives_quiet_scores():
    from coact_cps.vehicle.session_evidence import RetainedAlert
    a=RetainedAlert();assert not a.observe(False);assert a.observe(True);assert a.observe(False)


def test_no_keys_or_missing_certificate_cannot_admit():
    from coact_cps.vehicle.session_evidence import SessionEvidenceGate
    g,k,r=setup_gate();assert evaluate(SessionEvidenceGate(2.),r,1000).reason=='witness_unconfigured'
    assert not evaluate(g,r,1000).permit


def test_signed_quiet_admission_still_requires_base_permission():
    g,k,r=setup_gate();assert evaluate(g,r,1000,witness=wire(g,k,r,1000,1)).permit
    d=evaluate(g,r,1200,witness=wire(g,k,r,1200,2),base_permit=False)
    assert not d.permit and d.state=='QUARANTINED'


def test_alarm_has_priority_and_quiet_does_not_clear():
    g,k,r=setup_gate();evaluate(g,r,1000,witness=wire(g,k,r,1000,1))
    d=evaluate(g,r,1200,score=3.,witness=wire(g,k,r,1200,2))
    assert not d.permit and d.retained_alarm and d.reason=='score_alarm'
    for t in (1400,1600,1800):
        d=evaluate(g,r,t);assert not d.permit and d.retained_alarm


def test_recovery_needs_distinct_span_and_new_epoch_evidence():
    g,k,r=setup_gate();evaluate(g,r,1000,witness=wire(g,k,r,1000,1));evaluate(g,r,1200,score=3.)
    for seq,t in enumerate(range(1400,2401,200),3):
        d=evaluate(g,r,t,witness=wire(g,k,r,t,seq));assert not d.permit
        if t<2400:assert d.retained_alarm and d.state=='QUARANTINED'
    assert d.reason=='recovery_completed_new_evidence_required' and not d.retained_alarm
    assert evaluate(g,r,2600,witness=wire(g,k,r,2600,10)).permit


@pytest.mark.parametrize('failure',['missing','tampered','wrong_request','false_speed','stale'])
def test_unusable_evidence_never_restores_access(failure):
    g,k,r=setup_gate();c=wire(g,k,r,1000,1)
    if failure=='missing':c=None
    if failure=='tampered':
        p=json.loads(c);p['body']['speed_mps']=11.;c=canonical_json(p)
    if failure=='wrong_request':r=replace(r,request_id='other')
    if failure=='false_speed':c=wire(g,k,r,1000,2,speed_mps=17.)
    if failure=='stale':assert not evaluate(g,r,1300,witness=c,start_ms=1000).permit;return
    assert not evaluate(g,r,1000,witness=c).permit


def test_reused_evidence_and_clock_rollback_do_not_clear_memory():
    g,k,r=setup_gate();c=wire(g,k,r,1000,1);assert evaluate(g,r,1000,witness=c).permit
    d=evaluate(g,r,1200,witness=c);assert not d.permit and d.state=='QUARANTINED'
    assert not evaluate(g,r,1100).permit


@pytest.mark.parametrize('score',[None,float('nan'),float('inf'),True])
def test_unknown_after_admission_quarantines(score):
    g,k,r=setup_gate();assert evaluate(g,r,1000,witness=wire(g,k,r,1000,1)).permit
    d=evaluate(g,r,1200,score=score);assert not d.permit and d.state=='QUARANTINED'


def test_gap_and_scope_change_reset_recovery():
    g,k,r=setup_gate();evaluate(g,r,1000,score=3.)
    assert evaluate(g,r,1200,witness=wire(g,k,r,1200,1)).confirmations==1
    assert evaluate(g,r,1600,witness=wire(g,k,r,1600,2)).confirmations==1
    r=replace(r,resource='other');d=evaluate(g,r,1800,witness=wire(g,k,r,1800,3))
    assert d.confirmations==1 and d.retained_alarm and not d.permit


def test_new_nonce_same_scope_can_collect_confirmations():
    g,k,r=setup_gate();evaluate(g,r,1000,score=3.)
    for seq,t in enumerate(range(1200,2201,200),1):
        q=replace(r,request_id=str(seq),nonce=str(seq));d=evaluate(g,q,t,witness=wire(g,k,q,t,seq))
    assert d.reason=='recovery_completed_new_evidence_required'


def test_revocation_and_restart_are_fail_closed():
    from coact_cps.vehicle.session_evidence import SessionEvidenceGate
    g,k,r=setup_gate();c=wire(g,k,r,1000,1);assert evaluate(g,r,1000,witness=c).permit
    g.revoke(r.vehicle_id);assert evaluate(g,r,1200,witness=wire(g,k,r,1200,2)).reason=='revoked'
    restarted=SessionEvidenceGate(2.,{'test-witness':k.public_key().public_bytes_raw()})
    assert not evaluate(restarted,r,1200,witness=c).permit


def test_long_honest_session_and_legitimate_change():
    g,k,r=setup_gate()
    for seq,t in enumerate(range(1000,11001,200),1):
        speed=10.+min(2.,max(0.,(t-6000)/1000))
        assert evaluate(g,r,t,witness=wire(g,k,r,t,seq,speed_mps=speed),gnss=speed,wheel=speed).permit


def test_false_alarm_denial_cost_without_recovery():
    g,k,r=setup_gate();evaluate(g,r,1000,witness=wire(g,k,r,1000,1))
    assert not evaluate(g,r,1200,score=2.01).permit
    assert all(not evaluate(g,r,t).permit for t in range(1400,5001,200))


def test_threshold_and_inputs_are_validated():
    from coact_cps.vehicle.session_evidence import SessionEvidenceGate
    for t in [True,float('nan'),float('inf'),-1]:
        with pytest.raises(ValueError):SessionEvidenceGate(t)
    g,k,r=setup_gate();assert not evaluate(g,r,1000,base_permit=1).permit
    assert not evaluate(g,r,1200,quality=False).permit


def test_wrong_config_is_not_silently_replaced():
    from coact_cps.vehicle.session_evidence import SessionEvidenceGate
    for c in (False,0,{},'bad'):
        with pytest.raises(ValueError):SessionEvidenceGate(2.,witness_config=c)


def test_pre_quarantine_certificate_cannot_confirm_recovery():
    g,k,r=setup_gate();evaluate(g,r,1000,witness=wire(g,k,r,1000,1));c=wire(g,k,r,1200,2)
    evaluate(g,r,1200,score=3.,witness=c);d=evaluate(g,r,1400,witness=c)
    assert not d.permit and d.retained_alarm and d.confirmations==0


def test_alarm_is_retained_when_quality_is_invalid_before_admission():
    g,k,r=setup_gate()
    d=evaluate(g,r,1000,score=3.,quality=False)
    assert d.retained_alarm and d.state=='QUARANTINED'
    d=evaluate(g,r,1200,witness=wire(g,k,r,1200,1))
    assert not d.permit and d.retained_alarm and d.confirmations==1


def test_recovery_requires_motion_span_not_only_old_sample_span():
    g,k,r=setup_gate();evaluate(g,r,1200,score=3.)
    old=wire(g,k,r,1200,1)
    d=evaluate(g,r,1401,start_ms=1400,witness=old)
    for seq,t in enumerate((1601,1801,2001,2201),2):d=evaluate(g,r,t,witness=wire(g,k,r,t,seq))
    assert not d.permit and d.retained_alarm and d.state=='QUARANTINED'
    d=evaluate(g,r,2401,witness=wire(g,k,r,2401,6))
    assert d.reason=='recovery_completed_new_evidence_required'


def test_policy_configuration_is_read_only_while_evidence_is_pending():
    g,k,r=setup_gate();certificate=wire(g,k,r,1000,1);original=g.policy_digest
    with pytest.raises(AttributeError):g.threshold=4.
    with pytest.raises(AttributeError):g.policy_digest='other'
    assert g.policy_digest==original
    d=evaluate(g,r,1000,score=3.,witness=certificate)
    assert not d.permit and d.retained_alarm


def test_oversized_score_is_unknown_and_invalid_threshold_is_rejected():
    from coact_cps.vehicle.session_evidence import SessionEvidenceGate
    g,k,r=setup_gate();d=evaluate(g,r,1000,score=10**1000)
    assert not d.permit and not d.score_known and d.reason=='score_unknown'
    with pytest.raises(ValueError):SessionEvidenceGate(10**1000)
