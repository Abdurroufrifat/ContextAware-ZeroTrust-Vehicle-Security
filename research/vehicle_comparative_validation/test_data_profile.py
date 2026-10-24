import pytest
from public_profile import extract_series, quantile
def row(t,s):return {"time_ns":t,"x":s,"y":0,"z":0}
def test_causal_age_never_uses_future_command():
 r=extract_series([150_000_000,350_000_000],[row(i*100_000_000,i) for i in range(6)])
 assert r["age_us"]==[None,None,50000,150000,50000,150000]
def test_constant_acceleration_has_zero_velocity_second_difference():
 r=extract_series([0],[row(i*100_000_000,2+3*i) for i in range(6)])
 assert r["innovation_mps"]==[None,None,0,0,0,0]
def test_decreasing_receipt_times_fail_instead_of_reordering_history():
 with pytest.raises(ValueError):extract_series([10,9],[row(0,1),row(100_000_000,1)])
 with pytest.raises(ValueError):extract_series([0],[row(100_000_000,1),row(0,1)])
def test_duplicate_commands_are_preserved_in_metadata_and_causal_choice():
 r=extract_series([0,0,50_000_000],[row(0,1),row(100_000_000,1)])
 assert r["command_rows"]==3 and r["duplicate_command_stamps"]==1 and r["age_us"]==[0,50000]
def test_empirical_quantile_uses_fixed_nearest_rank():
 assert quantile([4,1,3,2],.5)==2
 assert quantile([4,1,3,2],.95)==4
