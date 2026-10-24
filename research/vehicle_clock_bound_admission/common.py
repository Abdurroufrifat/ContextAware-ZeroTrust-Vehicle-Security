from pathlib import Path
import hashlib,json,sys,os
R=Path(__file__).resolve().parent
P=R.parent.parent
H=R.parent/"vehicle_hercules_development"
M=R.parent/"vehicle_matched_evidence"
N=R.parent/"vehicle_neural_adaptation"
def forbidden(path):
    parts=str(path).replace("\\","/").lower()
    return "library03" in parts or "vehicle_final_evaluation" in parts or "rav4" in parts or ("urban02" in parts and "night" in parts)
def checked(path):
    p=Path(path)
    if forbidden(p) or forbidden(p.resolve()):raise PermissionError("reserved path")
    if not p.resolve().is_relative_to(P.resolve()):raise PermissionError("outside project")
    return p
def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def write(name,value):
    checked(R/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def install_guard():
    def guard(event,args):
        if event=="socket.connect":raise PermissionError("offline experiment")
        if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)) and forbidden(os.fsdecode(args[0])):
            raise PermissionError("reserved payload")
    sys.addaudithook(guard)
