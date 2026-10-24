from common import *
install_guard()
import contextlib,time
runtime=R/"runtime";runtime.mkdir(exist_ok=True)
for key in ("TEMP","TMP","TORCH_HOME","HF_HOME","XDG_CACHE_HOME","MPLCONFIGDIR"):
    directory=runtime/key;directory.mkdir(exist_ok=True);os.environ[key]=str(directory)
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
for path in (P,P/"scripts",P/"src",R,H,M,N,R.parent/"vehicle_crossmodal_doppler",R.parent/"vehicle_optical_timing",R.parent/"vehicle_neural_baseline",R.parent/"vehicle_reference_uncertainty"):
    sys.path.insert(0,str(path))
os.chdir(P)
import pytest
class Counts:
    def __init__(self):self.counts={};self.warnings=[]
    def pytest_terminal_summary(self,terminalreporter):self.counts={k:len(v) for k,v in terminalreporter.stats.items()}
    def pytest_warning_recorded(self,warning_message,when,nodeid,location):self.warnings.append(str(warning_message.message))
old=R.parent/"vehicle_development_release"
targets=[R/"test_integration.py",R/"test_trace_contract.py",R/"test_verifier_contract.py",R.parent/"vehicle_policy_gateway"/"test_gateway.py",R.parent/"vehicle_evidence_auth"/"test_auth.py",R.parent/"vehicle_command_tube"/"test_tube_core.py",R.parent/"vehicle_closed_loop_model"/"test_loop_core.py",R.parent/"vehicle_clock_bound_admission"/"test_clock_core.py",R.parent/"vehicle_equal_input_ablation"/"test_equal_input.py",old/"test_release_core.py",R.parent/"vehicle_reference_uncertainty"/"test_reference.py",
 R.parent/"vehicle_reference_qualification"/"test_reference_core.py",R.parent/"vehicle_inertial_consistency"/"test_inertial_core.py",
 R.parent/"vehicle_inertial_consistency"/"test_diagnostic_details.py",N/"test_neural_core.py",
 R.parent/"vehicle_neural_baseline"/"test_input_contract.py",M/"test_evidence_core.py",
 R.parent/"vehicle_optical_timing"/"test_compact_packet.py",R.parent/"vehicle_crossmodal_doppler"/"test_crossmodal_core.py",
 R.parent/"vehicle_crossmodal_doppler"/"test_crossmodal_replay.py",R.parent/"vehicle_hercules_reference_diagnosis"/"test_source_stress.py",
 H/"test_hercules_core.py",R.parent/"vehicle_independent_source_qualification"/"test_pilot_core.py",
 R.parent/"vehicle_independent_source_qualification"/"test_native_compat.py"]
validation_root=R/"runtime"/"recheck" if "--recheck" in sys.argv else R
validation_root.mkdir(parents=True,exist_ok=True)
summary={}
for name,args in (("focused",list(map(str,targets))),("core",[])):
    plugin=Counts();start=time.perf_counter()
    with (validation_root/(name+"_tests.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        code=pytest.main(args+["--basetemp",str(runtime/(name+"_tmp")),"-p","no:cacheprovider"],plugins=[plugin])
    summary[name]={"exit_code":int(code),"counts":plugin.counts,"warnings":plugin.warnings,"seconds":time.perf_counter()-start}
    print(name,json.dumps(summary[name]),flush=True)
checked(validation_root/"test_validation.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
sys.exit(0 if all(value["exit_code"]==0 for value in summary.values()) else 1)
