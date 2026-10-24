import numpy as np
import pytest

def context(labels):
    n=len(labels);t=np.arange(1,n+1)*100
    return dict(time_ms=t,start_ms=t-100,positions=np.arange(n),labels=np.asarray(labels,bool),base_grant=np.ones(n,bool),period=np.array(['active' if v else 'after' for v in labels]))

def test_retention_carries_across_segment_edges():
    from coact_cps.vehicle.session_evidence import RetainedAlert
    from run_vehicle_session_evidence import replay_segment
    state=RetainedAlert();a=replay_segment(context([1,1,0]),np.array([3.,0.,0.]),2.,state)
    b=replay_segment(context([0,0]),np.array([0.,0.]),2.,state)
    assert a['instant_alarm'].tolist()==[True,False,False]
    assert a['retained_alert'].all() and not b['retained_grant'].any()
    assert b['stateless_grant'].all() and not b['instant_alarm'].any()
    assert not a['strict_grant'].any() and not b['strict_grant'].any()

def test_post_attack_cost_and_unknowns():
    from coact_cps.vehicle.session_evidence import RetainedAlert
    from run_vehicle_session_evidence import replay_segment,policy_metrics
    c=context([1,1,0,0]);b=replay_segment(c,np.array([3.,0.,0.,np.nan]),2.,RetainedAlert())
    m=policy_metrics(c,b['retained_grant'],b['known'])
    assert m['attack_granted_epochs']==0 and m['post_attack_eligible_normal_denied']==2
    assert not b['stateless_grant'][-1]

def test_audit_rejects_rearming_and_missing_evidence_permit():
    from coact_cps.vehicle.session_evidence import RetainedAlert
    from run_vehicle_session_evidence import replay_segment
    from audit_vehicle_session_evidence import verify_block
    c=context([1,1,0]);b=replay_segment(c,np.array([3.,0.,0.]),2.,RetainedAlert());verify_block(c,b,False)
    b['retained_grant'][-1]=True
    with pytest.raises(AssertionError):verify_block(c,b,False)
    b=replay_segment(c,np.array([3.,0.,0.]),2.,RetainedAlert());b['strict_grant'][-1]=True
    with pytest.raises(AssertionError):verify_block(c,b,False)


def test_post_attack_cost_includes_next_segment_before_window():
    from coact_cps.vehicle.session_evidence import RetainedAlert
    from run_vehicle_session_evidence import replay_segment,policy_metrics
    from audit_vehicle_session_evidence import metrics
    a=RetainedAlert();replay_segment(context([1,1]),np.array([3.,0.]),2.,a)
    second=context([0,0]);second['period']=np.array(['before','before'])
    b=replay_segment(second,np.array([0.,0.]),2.,a,prior_attack_seen=True)
    m=policy_metrics(second,b['retained_grant'],b['known'],prior_attack_seen=True)
    assert m['post_attack_eligible_normal_epochs']==2 and m['post_attack_eligible_normal_denied']==2
    assert m['within_segment_after_eligible_normal_epochs']==0
    assert metrics(second,b['retained_grant'],b['known'],prior_attack_seen=True)==m


def test_accounting_history_never_changes_policy_grants():
    from coact_cps.vehicle.session_evidence import RetainedAlert
    from run_vehicle_session_evidence import replay_segment
    c=context([0,0]);a=replay_segment(c,np.array([0.,0.]),2.,RetainedAlert(),prior_attack_seen=False)
    b=replay_segment(c,np.array([0.,0.]),2.,RetainedAlert(),prior_attack_seen=True)
    for field in ('known','instant_alarm','retained_alert','stateless_grant','retained_grant','strict_grant'):
        np.testing.assert_array_equal(a[field],b[field])
