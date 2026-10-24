from pathlib import Path
import sys,os,json,time,contextlib
R=Path(__file__).resolve().parent;P=R.parent.parent
runtime=R/"runtime";runtime.mkdir(exist_ok=True)
for k in ["TEMP","TMP","TORCH_HOME","HF_HOME","XDG_CACHE_HOME","MPLCONFIGDIR"]:
 p=runtime/k;p.mkdir(exist_ok=True);os.environ[k]=str(p)
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
for p in [P,P/"scripts",P/"src",R]:sys.path.insert(0,str(p))
os.chdir(P);denials=[]
def guard(event,args):
 if event=="socket.connect":denials.append("network");raise PermissionError("offline verification")
 if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(args[0]).replace("\\","/").lower()
  if "/data/library03/" in p or "vehicle_final_evaluation" in p or "rav4" in p or ("urban02" in p and "night" in p):
   denials.append(p);raise PermissionError("reserved payload")
sys.addaudithook(guard)
import pytest
class Counts:
 def __init__(self):self.counts={};self.warnings=[]
 def pytest_terminal_summary(self,terminalreporter):self.counts={k:len(v) for k,v in terminalreporter.stats.items()}
 def pytest_warning_recorded(self,warning_message,when,nodeid,location):self.warnings.append(str(warning_message.message))
targets=[R/"test_crossmodal_core.py",R/"test_crossmodal_replay.py",R.parent/"vehicle_hercules_reference_diagnosis"/"test_source_stress.py",R.parent/"vehicle_hercules_development"/"test_hercules_core.py",R.parent/"vehicle_independent_source_qualification"/"test_pilot_core.py",R.parent/"vehicle_independent_source_qualification"/"test_native_compat.py"]
results={}
for name,target in [("focused",list(map(str,targets))),("core",["tests"])]:
 plugin=Counts();start=time.perf_counter()
 with (R/(name+"_tests.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  code=pytest.main(target+["-q","--basetemp",str(runtime/(name+"_tmp")),"-p","no:cacheprovider"],plugins=[plugin])
 results[name]={"exit_code":int(code),"counts":plugin.counts,"warnings":plugin.warnings,"seconds":time.perf_counter()-start}
 print(name,json.dumps(results[name]),flush=True)
results["denials"]=denials
(R/"test_validation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
sys.exit(0 if all(results[x]["exit_code"]==0 for x in ["focused","core"]) else 1)
