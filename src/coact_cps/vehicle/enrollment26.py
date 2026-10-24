"""Offline normal-enrollment residual calibration; no online adaptation."""
import numpy as np


def _normal(nodes,labels):
    x,y=np.asarray(nodes),np.asarray(labels)
    if (x.ndim!=2 or x.shape[1]!=3 or len(x)<2 or not np.isfinite(x).all()
            or y.shape!=(len(x),) or y.dtype.kind!='b' or y.any()):
        raise ValueError('finite residuals from exclusively presumed-normal enrollment required')
    return x


def enroll_profile(nodes,labels):
    x=_normal(nodes,labels)
    return dict(median=np.median(x,axis=0).tolist(),scale=np.maximum(
        np.quantile(x,0.75,axis=0)-np.quantile(x,0.25,axis=0),0.01).tolist())


def score(nodes,profile):
    x=np.asarray(nodes);med,scale=np.asarray(profile['median']),np.asarray(profile['scale'])
    if (x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or med.shape!=(3,)
            or scale.shape!=(3,) or not np.isfinite(med).all()
            or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError('finite residuals and positive three-channel calibration required')
    return ((x-med)/scale).max(axis=1)


def calibrate_threshold(nodes,labels,profile):
    return float(np.quantile(score(_normal(nodes,labels),profile),0.95))


def screen(normal_alarm,original_alarm,normal_grants,original_grants,worst_tpr_loss,
           additional_missed_events,longer_attack_run,added_attack_span):
    checks=dict(normal_alarm_at_most_5_percent=normal_alarm<=0.05,
                alarm_reduction_at_least_1pp=original_alarm-normal_alarm>=0.01,
                normal_grants_not_lower=normal_grants>=original_grants,
                each_case_tpr_loss_at_most_5pp=worst_tpr_loss<=0.05,
                no_new_missed_events=additional_missed_events==0,
                no_longer_attack_grant_runs=not longer_attack_run,
                no_case_mean_added_attack_observed_span=not added_attack_span)
    return dict(**checks,passed=all(checks.values()),eligible_for_test=False,
                status='DEVELOPMENT_ONLY_NOT_PROMOTED',fresh_validation_required=True)
