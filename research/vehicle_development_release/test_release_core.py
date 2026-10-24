from pathlib import Path
import importlib.util,math,hashlib,json,numpy as np,pytest
def core():
    p=Path(__file__).with_name("release_core.py")
    assert p.exists(),"release_core implementation missing"
    s=importlib.util.spec_from_file_location("release_tested",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_rounding_above_half_cannot_pass():
    row=core().exact_sensitivity([.25],[0.],math.nextafter(.25,math.inf))
    assert row["true_error_p95_upper_display"]==.5
    assert row["true_error_p95_upper_exact"]=="9007199254740993/18014398509481984"
    assert not row["scored_subset_upper_within_0p5"]
def test_exact_half_is_inclusive():
    row=core().exact_sensitivity([.25],[0.],.25)
    assert row["scored_subset_upper_within_0p5"] and row["conditional_full_admitted_set_upper_within_0p5"]
def test_above_two_is_counted_despite_float_rounding():
    row=core().exact_sensitivity([2.],[0.],math.nextafter(0.,1.))
    assert row["possibly_above2"]==1 and row["definitely_above2"]==0
def test_unknown_bound_never_becomes_zero():
    row=core().exact_sensitivity([1],[1],None)
    assert row["true_error_p95_upper_exact"] is None and not row["scored_subset_upper_within_0p5"]
def test_missing_reference_blocks_full_admitted_flag():
    row=core().exact_sensitivity([0.,1.],[0.,np.nan],.1)
    assert row["admitted"]==2 and row["scored"]==1 and row["unknown_reference"]==1
    assert row["scored_subset_upper_within_0p5"] and not row["conditional_full_admitted_set_upper_within_0p5"]
def test_empty_scores_do_not_pass():
    row=core().exact_sensitivity([1.],[np.nan],0.)
    assert not row["conditional_full_admitted_set_upper_within_0p5"]
@pytest.mark.parametrize("name",["../x","C:/outside/x","/outside/x","a/../../outside","data/Library03/x.bin","RAV4/x.bin","Urban02/NIGHT/x.bin","vehicle_final_evaluation/x.npz"])
def test_unsafe_artifact_paths_rejected(tmp_path,name):
    with pytest.raises(ValueError):core().artifact_path(tmp_path,name)
def test_manifest_accepts_identity_and_rejects_tampering(tmp_path):
    m=core();p=tmp_path/"x.json";p.write_text("{}")
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/"manifest.json").write_text(json.dumps({"x.json":digest}))
    assert len(m.verify_manifest(tmp_path,"manifest.json"))==1
    p.write_text("changed")
    with pytest.raises(ValueError):m.verify_manifest(tmp_path,"manifest.json")

def test_resolved_reserved_junction_rejected(tmp_path):
    import os
    m=core();target=tmp_path/"Library03";target.mkdir();link=tmp_path/"safe_alias"
    if os.name=="nt":
        import _winapi
        _winapi.CreateJunction(str(target),str(link))
    else:link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError):m.artifact_path(tmp_path,"safe_alias")
def test_scenario_identity_accepts_matching_historical_row():
    m=core();prior={"sequence":"Library01","group":"all","scope":"original_radar","epsilon_assumption_mps":.1}
    assert m.validate_scenario_identity({"row":0,**prior},prior,0) is None
@pytest.mark.parametrize("key,value",[("row",99),("sequence","Other"),("group","segment_0"),("scope","current_inertial_mask"),("epsilon_assumption_mps",.2)])
def test_scenario_identity_rejects_substituted_row(key,value):
    m=core();prior={"sequence":"Library01","group":"all","scope":"original_radar","epsilon_assumption_mps":.1}
    row={"row":0,**prior};row[key]=value
    with pytest.raises(ValueError):m.validate_scenario_identity(row,prior,0)
