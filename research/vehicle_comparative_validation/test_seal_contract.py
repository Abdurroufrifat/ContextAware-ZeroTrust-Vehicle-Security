"""Exercise final publication order without touching research artifacts."""
from pathlib import Path
import json,hashlib,subprocess,sys
import pytest
SOURCE=Path(__file__).with_name("seal_release.py")
@pytest.mark.parametrize("fail_manifest",[True,False])
def test_completion_marker_is_published_only_after_manifest_checks(tmp_path,fail_manifest):
    b=tmp_path/"research"/"study";b.mkdir(parents=True);(b/"runtime").mkdir()
    shim='''from pathlib import Path
import json,hashlib,os
B=Path(__file__).resolve().parent;P=B.parent.parent
def sha(p):
 if FAIL and p.name=="development_manifest.json":raise RuntimeError("injected final manifest failure")
 return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 t=p.with_name(p.name+".tmp");t.write_text(json.dumps(v,indent=2));os.replace(t,p)
'''.replace("FAIL",repr(fail_manifest))
    (b/"cv_common.py").write_text(shim);(b/"seal_release.py").write_text(SOURCE.read_text())
    result={"trials":[{} for _ in range(1512)]};(b/"results.json").write_text(json.dumps(result))
    h=hashlib.sha256((b/"results.json").read_bytes()).hexdigest()
    objects={"verification.json":{"passed":True,"counts":{"decisions":302400},"input_results_sha256":h},
      "policy_replay.json":{"passed":True,"decisions":302400,"input_results_sha256":h},
      "report_verification.json":{"passed":True,"input_results_sha256":h},
      "runtime/preflight_tests.json":{"passed":True,"verification_lock_digest":hashlib.sha256(b"{}").hexdigest()},
      "experiment_lock.json":{},"verification_lock.json":{},
      "assembly_status.json":{"status":"AWAITING_REPORT_QA"}}
    for name,value in objects.items():(b/name).write_text(json.dumps(value))
    for name in ("COACT_CPS_COMPARATIVE.py","COMPARATIVE_SIMULATION_START.md"):(tmp_path/name).write_text("fixture")
    r=subprocess.run([sys.executable,str(b/"seal_release.py")],cwd=b,capture_output=True,text=True)
    assert (r.returncode!=0)==fail_manifest,r.stdout+r.stderr
    assert json.loads((b/"assembly_status.json").read_text())["status"]=="AWAITING_REPORT_QA"
    assert (b/"final_seal.json").exists()==(not fail_manifest)
    if not fail_manifest:
        seal=json.loads((b/"final_seal.json").read_text())
        assert seal["status"]=="COMPLETE_FOR_COMPARATIVE_SIMULATION_SCOPE"
        assert seal["package_manifest_sha256"]==hashlib.sha256((b/"package_manifest.json").read_bytes()).hexdigest()
