from pathlib import Path
import os,sys,hashlib,json
R=Path(__file__).resolve().parent;P=R.parent.parent
H=R.parent/"vehicle_hercules_development";T=R.parent/"vehicle_optical_timing";M=R.parent/"vehicle_matched_evidence"
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
def guard(event,args):
    if event=="socket.connect":raise PermissionError("offline diagnostic")
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace("\\","/").lower()
        if "/data/library03/" in p or "vehicle_final_evaluation" in p or "rav4" in p or ("urban02" in p and "night" in p):raise PermissionError("reserved payload")
sys.addaudithook(guard)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((R/name).read_text(encoding="utf-8"))
def write(name,value):(R/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
def verify_lock():
    for name,digest in read("experiment_lock.json").items():assert sha(R/name)==digest,name
