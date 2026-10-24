"""Tampering tests for the independent saved-decision reconstruction."""
import numpy as np
import pytest


def fixture():
    time=np.arange(1,7,dtype=np.int64)*100
    c=dict(time_ms=time,start_ms=time-100,positions=np.arange(1,6),
           labels=np.array([False,False,True,True,True,True]),base_grant=np.ones(6,dtype=bool))
    score=np.array([np.nan,2.,0.,0.,2.]);alarm=np.array([False,True,False,False,True]);known=np.isfinite(score)
    grant=np.array([False,False,False,True,True,False]);effective=np.array([True,False,True,True,True,True])
    b=dict(score=score,alarm=alarm,known=known,grant=grant,effective_base=effective,threshold=np.array(1.),positions=c['positions'])
    return c,b


def test_independent_metrics_count_unknown_denial_and_escape():
    import audit_vehicle_joint_phase2 as a
    c,b=fixture();r=a.independent_metrics(c,b)
    assert r['normal_total_denied']==2
    assert r['unknown_scored_epochs']==1
    assert r['attack_longest_grant_run_epochs']==2
    assert r['attack_granted_observed_span_ms']==200
    assert r['events'][0]['granted_epochs_after_first_denial']==2


@pytest.mark.parametrize('field',['grant','alarm','known','effective_base'])
def test_independent_audit_rejects_changed_decision_even_after_rehash(field):
    import audit_vehicle_joint_phase2 as a
    c,b=fixture();b[field]=b[field].copy();b[field][1]=~b[field][1]
    with pytest.raises(AssertionError):a.independent_metrics(c,b)


def test_independent_feature_formula_uses_right_endpoint_integral():
    import audit_vehicle_joint_phase2 as a
    t=np.array([100,220,410,600,710,900],dtype=np.int64);s=np.r_[0,t[:-1]]
    accel=np.arange(6,dtype=float);speed=np.cumsum(accel*(t-s)/1000)
    j,p=a.independent_features(np.column_stack((speed,speed,accel)),t,s)
    np.testing.assert_allclose(j[5,3:],0,atol=1e-12)
    assert len(p)==0


def test_independent_threshold_must_equal_frozen_model_threshold():
    import audit_vehicle_joint_phase2 as a
    _,b=fixture()
    lock={'thresholds':{'joint_history_gru6__7':2.}}
    with pytest.raises(AssertionError):a.verify_score_contract(b,'joint_history_gru6',7,lock,{})


def test_original_neural_scores_are_bound_to_saved_nodes_and_profile():
    import audit_vehicle_joint_phase2 as a
    b=dict(score=np.array([8.]),residual=np.ones((1,3)),threshold=np.array(2.))
    refs={'7':dict(threshold=2.,profile=dict(median=[0.,0.,0.],scale=[1.,1.,1.]))}
    with pytest.raises(AssertionError):a.verify_score_contract(b,'original_nn_frozen',7,{},refs)


def test_physics_audit_resets_matched_positions_after_native_scope():
    import audit_vehicle_joint_phase2 as a
    import run_vehicle_joint_phase2 as r
    c,_=fixture();report=r.physics_reference(c)
    assert report['native']['scored_epochs']==6
    assert report['matched']['scored_epochs']==5
    a.verify_physics_reference(c,report)
