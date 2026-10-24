from pathlib import Path
import importlib.util,numpy as np,pytest
def core():
    p=Path(__file__).with_name("reference_core.py")
    assert p.exists(),"reference_core implementation missing"
    s=importlib.util.spec_from_file_location("reference_tested",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_status_tokens_parse_without_guessing_good():
    m=core();assert m.parse_status("status: 3")==3;assert m.parse_status("status: 6")==6;assert m.parse_status("status: 99")==99
@pytest.mark.parametrize("value",["","status: 3.0","GOOD","status: nan"])
def test_malformed_status_rejected(value):
    with pytest.raises(ValueError):core().parse_status(value)
def test_status_is_causal_and_age_bound_is_inclusive():
    codes,age=core().latest_status([9,10,110_000_010,110_000_011],[10,210_000_010],[3,6])
    np.testing.assert_array_equal(codes,[-1,3,-1,-1])
    c,a=core().latest_status([100_000_010,100_000_011],[10,210_000_010],[3,6])
    np.testing.assert_array_equal(c,[3,-1]);assert a[0]==100_000_000
def test_future_status_is_not_backfilled():
    c,a=core().latest_status([99,100],[100,200],[6,3]);np.testing.assert_array_equal(c,[-1,6])
def test_unknown_bound_is_not_zero_error():
    out=core().sensitivity([0,2],[0,1],None)
    assert not out["reference_bound_known"] and out["true_error_p95_upper"] is None and not out["conditional_full_admitted_set_upper_within_0p5"]
def test_triangle_bound_and_counts_on_known_rows():
    out=core().sensitivity([0,3,5],[0,1,2],.2)
    assert abs(out["true_error_p95_upper"]-3.1)<1e-12
    assert abs(out["true_error_p95_lower"]-2.7)<1e-12
    assert out["definitely_above2"]==1 and out["possibly_above2"]==2
def test_missing_reference_blocks_complete_certificate():
    out=core().sensitivity([0.,1],[0.,np.nan],.1)
    assert out["scored"]==1 and out["unknown_reference"]==1
    assert out["scored_subset_upper_within_0p5"] and not out["conditional_full_admitted_set_upper_within_0p5"]
def test_observed_failure_has_no_nonnegative_reference_budget():
    out=core().sensitivity([1.,1.],[0.,0.],.1)
    assert out["max_uniform_epsilon_for_scored_p95"] is None and not out["scored_subset_upper_within_0p5"]
def test_zero_bound_cannot_make_unavailable_predictions_valid():
    out=core().sensitivity([np.nan,1],[0.,1],0.)
    assert out["admitted"]==1 and out["scored"]==1
def test_no_scores_do_not_pass():
    out=core().sensitivity([1],[np.nan],0)
    assert out["scored"]==0 and out["true_error_p95_upper"] is None and not out["conditional_full_admitted_set_upper_within_0p5"]
@pytest.mark.parametrize("bound",[-1,float("nan"),float("inf")])
def test_invalid_bounds_rejected(bound):
    with pytest.raises(ValueError):core().sensitivity([0],[0],bound)
