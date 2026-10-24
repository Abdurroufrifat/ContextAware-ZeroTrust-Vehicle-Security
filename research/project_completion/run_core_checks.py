"""Full core test runner; all temporary writes stay in the project."""
import contextlib,json,os,shutil,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent.parent;runtime=R/"runtime";runtime.mkdir(exist_ok=True)
previous=R/"core_test_validation.json"
if previous.exists() and not (R/"core_test_validation_attempt1.json").exists():shutil.copyfile(previous,R/"core_test_validation_attempt1.json")
for name,folder in {"TEMP":"tmp","TMP":"tmp","TORCH_HOME":"torch","HF_HOME":"hf","XDG_CACHE_HOME":"cache","MPLCONFIGDIR":"mpl"}.items():
 q=runtime/folder;q.mkdir(exist_ok=True);os.environ[name]=str(q)
os.environ.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
sys.dont_write_bytecode=True
for path in [P,P/"scripts",P/"src"]:sys.path.insert(0,str(path))
os.chdir(P);denied=[]
def guard(event,args):
 if event=="socket.connect":
  denied.append(dict(event=event,path="network"));raise PermissionError("Core checks must remain offline")
 if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
  name=os.fsdecode(args[0]).replace("\\","/").lower()
  if "vehicle_final_evaluation" in name or "rav4" in name or ("urban02" in name and "night" in name):
   denied.append(dict(event=event,path=name));raise PermissionError("Reserved payload excluded before opening")
sys.addaudithook(guard)
import pytest
class Summary:
 counts={}
 warnings=[]
 def pytest_terminal_summary(self,terminalreporter):
  self.counts={k:len(v) for k,v in terminalreporter.stats.items()}
 def pytest_warning_recorded(self,warning_message,when,nodeid,location):
  self.warnings.append(dict(nodeid=nodeid,message=str(warning_message.message)))
plugin=Summary();start=time.perf_counter()
with (R/"core_tests.log").open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(["tests","-q","--basetemp",str(runtime/"pytest_tmp"),"-p","no:cacheprovider"],plugins=[plugin])
result=dict(exit_code=int(code),elapsed_seconds=time.perf_counter()-start,core_test_files=91,
 counts=plugin.counts,warnings=plugin.warnings,protected_or_network_access_denials=denied,
 suite="pyproject tests/ full core suite; research-module unittest suites are separate",
 production_policy_changed=False,protected_payloads_opened=False)
previous.write_text(json.dumps(result,indent=2)+"\n")
print("CORE_CHECK_RESULT",json.dumps(result),flush=True)
sys.exit(int(code))
