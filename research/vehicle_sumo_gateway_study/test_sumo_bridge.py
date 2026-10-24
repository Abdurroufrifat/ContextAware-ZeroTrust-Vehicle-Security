import importlib,json,pytest
def api():return importlib.import_module("sumo_bridge")
def numeric(**kw):
 d=dict(policy="clock",candidate=10000,anchor=10000,stamp=910000,now=1000000,clock=5000,history=[(900000,1000000,0)],risk="normal")
 d.update(kw);return api().numeric(**d)
def test_nonzero_clock_90ms_recovers_with_history():
 assert numeric()["value_mmps"] is None
 r=numeric(policy="tube");assert r["value_mmps"]==10000 and r["selection"]=="candidate"
def test_uncertain_age_beyond_100ms_rejects():
 assert numeric(policy="tube",stamp=900000)["value_mmps"] is None
def test_static_baseline_is_explicitly_age_agnostic():
 assert numeric(policy="static",candidate=13000)["value_mmps"]==10000
def test_lockdown_rejects_before_motion_selection():
 assert numeric(policy="tube",risk="lockdown")["value_mmps"] is None
def test_elevated_uses_anchor_not_candidate():
 assert numeric(policy="tube",candidate=10100,risk="elevated")["selection"]=="propagated_anchor"
def test_missing_command_history_rejects():
 assert numeric(policy="tube",history=[])["value_mmps"] is None
def test_excessive_history_not_silently_truncated():
 with pytest.raises(ValueError):api().history_window([(k,k+1,0) for k in range(20)],0,20)
def test_history_window_rejects_gap():
 with pytest.raises(ValueError):api().history_window([(0,10,0),(11,20,0)],0,20)
def test_history_window_covers_requested_time():
 assert api().history_window([(0,10,0),(10,20,2000)],5,20)==[(0,10,0),(10,20,2000)]
@pytest.mark.parametrize("v",[None])
def test_missing_estimate_brakes(v):assert api().idm_command(v,20,100)==-4000
def test_idm_free_road_acceleration_is_clipped():
 assert api().idm_command(0,20,100000)==2000
def test_idm_closing_gap_brakes():
 assert api().idm_command(20,0,1)==-4000
def test_idm_nonfinite_input_rejected():
 with pytest.raises(ValueError):api().idm_command(float("nan"),10,20)
def test_control_bounds():
 for v in (0,1,10,20,30):
  for gap in (.1,2,20,100):assert -4000<=api().idm_command(v,10,gap)<=2000
@pytest.mark.parametrize("policy",["static","clock","tube"])
def test_adapter_never_overrides_failed_authentication(policy):
 decision={"selection":"abstain","reason":"anchor_bad_tag","anchor_status":"bad_tag","command_status":"authenticated","candidate_status":"authenticated","research_speed_mmps":None}
 assert api().adapt_verified(policy,decision,None,None,1000000,5000)["research_speed_mmps"] is None
def test_unknown_policy_rejected():
 with pytest.raises(ValueError):numeric(policy="invented")
