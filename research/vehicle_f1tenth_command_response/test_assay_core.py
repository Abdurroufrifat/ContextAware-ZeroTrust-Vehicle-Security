import pytest
from assay_core import pair_samples

def test_causal_alignment_never_uses_future_command():
    out=pair_samples([(100,1.0),(300,9.0)],[(50,1.,0.,0.),(100,3.,4.,0.),(250,2.,0.,0.),(300,0.,0.,12.)],100)
    assert out[0]["status"]=="unmatched"
    assert out[1]["reference_mps"]==5.0
    assert out[1]["command_mps"]==1.0
    assert out[2]["command_mps"]==1.0
    assert out[2]["status"]=="stale"
    assert out[3]["command_mps"]==9.0

def test_age_boundary_inclusive():
    out=pair_samples([(0,2.)],[(100,2.,0.,0.),(101,2.,0.,0.)],100)
    assert [x["status"] for x in out]==["fresh","stale"]

def test_speed_magnitude_handles_reverse_and_world_frame():
    out=pair_samples([(1,-5.)],[(1,0.,-3.,4.)],100)
    assert out[0]["error_mps"]==0.
    assert out[0]["command_signed_mps"]==-5.

@pytest.mark.parametrize("commands,refs",[
 ([(2,1.),(1,2.)],[(3,1.,0.,0.)]),
 ([(1,1.),(1,2.)],[(3,1.,0.,0.)]),
 ([(1,float("nan"))],[(3,1.,0.,0.)]),
 ([(1,1.)],[(3,float("inf"),0.,0.)]),
 ([(1,1.)],[(3,1.,0.,0.),(2,1.,0.,0.)]),
])
def test_invalid_stream_rejected(commands,refs):
    with pytest.raises(ValueError):
        pair_samples(commands,refs,100)

def test_empty_commands_keep_unmatched_reference():
    assert pair_samples([],[(5,0.,0.,0.)],100)[0]["status"]=="unmatched"

def test_duplicate_reference_time_is_preserved():
    assert len(pair_samples([(0,1.)],[(1,1.,0.,0.),(1,2.,0.,0.)],100))==2


def test_transverse_command_rejected_before_magnitude_comparison():
    import assay_core
    with pytest.raises(ValueError):
        assay_core.longitudinal_commands([{"time_ns":1,"linear_x":1.,"linear_y":.1,"linear_z":0.}])

def test_longitudinal_command_adapter_preserves_sign_and_time():
    import assay_core
    assert assay_core.longitudinal_commands([{"time_ns":2,"linear_x":-3.,"linear_y":0.,"linear_z":0.}])==[(2,-3.)]
