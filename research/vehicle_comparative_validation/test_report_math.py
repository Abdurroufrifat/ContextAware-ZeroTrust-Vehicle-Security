import pytest
from report_math import summarize,paired,cluster_effect
def row(seed,method,mae=1,selected=200):
 return {"profile":"p","case":"c","seed":seed,"method":method,"group":"primary","mae_mps":mae,"selected":selected,"queries":200,
 "over_0_5":0,"collision_trial":False,"collision_episodes":0,"progress_m":2,"minimum_gap_m":3,"seconds_gap_below_2m":0,
 "fallback_queries":0,"unrecoverable_queries":0,"changed_actuator_frames":0,"changed_actuator_frames_accepted":0,"maximum_decision_seconds":.1}
def test_query_weighting_does_not_average_unequal_trial_means():
 s=summarize([row(1,"a",1,200),row(2,"a",9,1)])
 assert s["selected_mae_mps"]==pytest.approx(209/201)
 assert s["availability"]==201/400
def test_pairs_reject_missing_or_duplicate_worlds():
 rows=[row(1,"a"),row(1,"b"),row(2,"a")]
 with pytest.raises(AssertionError):paired(rows,"a","b","primary")
 with pytest.raises(AssertionError):paired([row(1,"a"),row(1,"a"),row(1,"b")],"a","b","primary")
def test_seed_cluster_effect_does_not_treat_queries_as_replicates():
 rows=[{"seed":s,"x":s} for s in range(1,7) for _ in range(10)]
 e=cluster_effect(rows,"x")
 assert e["seed_clusters"]==6 and e["mean_left_minus_right"]==3.5
 assert e==cluster_effect(rows,"x")
