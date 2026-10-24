from pathlib import Path
import shutil
R=Path(__file__).resolve().parent;archive=R/"pre_review_fix";archive.mkdir()
for name in ["release_core.py","verify_release.py","test_release_core.py","experiment_lock.json","independent_verification.json"]:
    shutil.copyfile(R/name,archive/name)
tests='''
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
'''
with (R/"test_release_core.py").open("a",encoding="utf-8") as f:f.write(tests)
print("Archived pre-review source/lock/report; appended seven regression cases.")
