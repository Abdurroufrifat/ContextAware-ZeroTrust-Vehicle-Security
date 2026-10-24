from sg_common import *
import subprocess,copy
runtime=R/"runtime"/"verifier_probes";runtime.mkdir(parents=True,exist_ok=True)
base=json.loads((R/"results.json").read_text());records=[]
for name in ("omitted_trial","duplicate_trial","reference_relabel","false_physical_claim"):
 data=copy.deepcopy(base)
 if name=="omitted_trial":data["trials"].pop()
 elif name=="duplicate_trial":data["trials"][1]=copy.deepcopy(data["trials"][0])
 elif name=="reference_relabel":data["trials"][0]["native_idm_reference"]=True
 else:data["physical_vehicle_validation_established"]=True
 path=runtime/(name+".json");path.write_text(json.dumps(data),encoding="utf-8")
 run=subprocess.run([sys.executable,str(R/"verify_study.py"),"--results",str(path)],capture_output=True,text=True)
 assert run.returncode!=0 and "AssertionError" in run.stderr,(name,run.stdout,run.stderr)
 records.append({"mutation":name,"rejected":True,"exit_code":run.returncode})
 (runtime/(name+".log")).write_text(run.stdout+run.stderr,encoding="utf-8")
write("verifier_probes.json",{"all_rejected":True,"probes":records})
print("Four verifier mutation probes rejected")
