from pathlib import Path
import sys,os,json,hashlib
R=Path(__file__).resolve().parent;P=R.parent.parent
H=R.parent/"vehicle_hercules_development";M=R.parent/"vehicle_matched_evidence"
os.environ.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
def guard(event,args):
    if event=="socket.connect":raise PermissionError("offline experiment")
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]).replace("\\","/").lower()
        if "/data/library03/" in name or "vehicle_final_evaluation" in name or "rav4" in name or ("urban02" in name and "night" in name):raise PermissionError("reserved payload")
sys.addaudithook(guard)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((R/name).read_text(encoding="utf-8"))
def write(name,value):(R/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
def verify_lock():
    for name,digest in read("experiment_lock.json").items():assert sha(R/name)==digest,name
def frames(data):
    return [data["features"][a:b] for a,b in zip(data["offsets"][:-1],data["offsets"][1:])]
