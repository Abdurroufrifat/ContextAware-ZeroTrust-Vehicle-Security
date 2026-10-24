import pytest
import report_core as r
def test_aggregation_uses_selected_denominator_and_trial_units():
    trials=[{"mode":"m","case":"nominal","queries":200,"selected":100,"mae_mps":.2,"over_0_5":3,"over_2":1,
             "collision_events":2,"collision_trial":True,"progress_m":100.,"minimum_gap_m":1.,"fallback_queries":40,"unrecoverable_queries":2},
            {"mode":"m","case":"nominal","queries":100,"selected":50,"mae_mps":.4,"over_0_5":4,"over_2":0,
             "collision_events":0,"collision_trial":False,"progress_m":150.,"minimum_gap_m":3.,"fallback_queries":10,"unrecoverable_queries":0}]
    out=r.aggregate(trials)[0]
    assert out["trials"]==2 and out["queries"]==300 and out["selected"]==150
    assert out["mae_mps"]==pytest.approx(40/150)
    assert out["collision_trials"]==1 and out["collision_events"]==2
    assert out["mean_progress_m"]==125 and out["minimum_gap_m"]==1
    assert out["fallback_fraction"]==pytest.approx(1/6)
def test_empty_selection_is_not_zero_error():
    trial={"mode":"m","case":"attack","queries":10,"selected":0,"mae_mps":None,"over_0_5":0,"over_2":0,"collision_events":0,"collision_trial":False,"progress_m":0.,"minimum_gap_m":2.,"fallback_queries":10,"unrecoverable_queries":0}
    assert r.aggregate([trial])[0]["mae_mps"] is None
def test_modes_and_conditions_stay_separate():
    template={"queries":1,"selected":0,"mae_mps":None,"over_0_5":0,"over_2":0,"collision_events":0,"collision_trial":False,"progress_m":0.,"minimum_gap_m":2.,"fallback_queries":1,"unrecoverable_queries":0}
    out=r.aggregate([{**template,"mode":"a","case":"x"},{**template,"mode":"b","case":"x"},{**template,"mode":"a","case":"y"}])
    assert len(out)==3
