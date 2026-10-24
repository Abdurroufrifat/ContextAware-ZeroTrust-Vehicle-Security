import numpy as np,pytest
from crossmodal_replay import latest_completed,make_cases
def test_latest_completed_is_causal_and_inclusive():
 np.testing.assert_array_equal(latest_completed([10,20],[9,10,19,20]),[-1,0,0,1])
def test_later_invalid_frame_supersedes_older_valid():
 k=latest_completed([10,20],[20]);quality=np.array([True,False])
 assert not quality[k[0]]
def fixture():
 q=np.array([0,1_000_000_000,2_000_000_000],dtype=np.int64)
 return q,np.array([8.,8.,8.]),np.array([8.,8.,8.]),np.ones(3,bool),np.zeros(3,dtype=np.int64),np.zeros(3,dtype=np.int64)
def test_common_attack_and_single_attack_are_distinct():
 c=make_cases(*fixture())
 assert np.all(c["nominal"]["accepted"])
 assert not np.any(c["radar_plus3"]["accepted"])
 assert not np.any(c["optical_plus3"]["accepted"])
 assert np.all(c["both_plus3"]["accepted"])
 np.testing.assert_allclose(c["both_plus3"]["prediction"],11.)
def test_honest_replay_expired_forged_replay_retains_limitation():
 c=make_cases(*fixture())
 assert not np.any(c["radar_replay_honest"]["accepted"])
 assert not c["radar_replay_forged"]["supported"][0]
 assert c["radar_replay_forged"]["accepted"][1:].all()
def test_missing_and_just_stale_optical_abstain():
 c=make_cases(*fixture())
 assert not np.any(c["missing_optical"]["accepted"])
 assert not np.any(c["stale_optical"]["accepted"])
def test_query_before_completed_history_not_wrapped():
 np.testing.assert_array_equal(latest_completed([100],[0,99,100]),[-1,-1,0])
