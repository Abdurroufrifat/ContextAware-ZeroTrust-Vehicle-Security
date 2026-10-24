import math,pytest
from baseline_core import merge_sets, agreement, predict_sets, s3e_update, closest_point, zocbf_command, cbf_slack
def test_disconnected_agreements_are_not_collapsed_to_intersection_or_hull():
 assert agreement([(0,10),(0,2),(8,10)],1)==[(0,2),(8,10)]
def test_time_update_preserves_both_branches_and_clips_zero():
 result=predict_sets([(0,.1),(2,3)],-4,.1,1)
 assert result==[(0,0),(1.6,2.7)]
def test_s3e_filters_out_attack_inconsistent_with_honest_pair():
 result=s3e_update([(0,30)],[(10,10.2),(9.9,10.1),(20,21)],0,.1,1)
 assert result==[(10,10.1)]
def test_missing_domains_reduce_available_redundancy_without_truth_reset():
 assert agreement([(1,2),(4,5)],1)==[(1,2),(4,5)]
 assert agreement([(1,2)],1)==[]
 assert s3e_update([],[(1,2),(1,2),(8,9)],0,.1,1)==[]
def test_point_can_be_in_hull_without_fabricating_a_narrow_set():
 assert closest_point([(0,2),(8,10)])==5
def test_cbf_returns_least_integer_change_among_all_feasible_commands():
 for ego,lead,gap in [(8,6,15),(15,12,20),(0,0,5),(20,0,9)]:
  wanted=2000
  got,slack=zocbf_command(wanted,ego,lead,gap)
  feasible=[u for u in range(-4000,wanted+1) if cbf_slack(u,ego,lead,gap)>=0]
  assert got==(max(feasible) if feasible else -4000)
  assert math.isclose(slack,cbf_slack(got,ego,lead,gap),abs_tol=1e-12)
def test_cbf_checks_current_gap_even_when_lead_is_faster():
 u,margin=zocbf_command(2000,0,15,1.9)
 assert u==-4000 and margin<0
def test_invalid_attack_budget_is_rejected():
 with pytest.raises(ValueError):agreement([(0,1),(0,1)],-1)

def test_cbf_catches_intersample_headway_dip_before_terminal_recovery():
 assert cbf_slack(-4000,3.15,0,5.152)<0
