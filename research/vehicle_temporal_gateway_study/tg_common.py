from pathlib import Path
import os,sys,json,hashlib
R=Path(__file__).resolve().parent;P=R.parent.parent;OLD=R.parent/"vehicle_sumo_gateway_study"
sys.path.insert(0,str(OLD))
from sg_common import checked,install_guard,BINARY,NETCONVERT
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def write(name,obj):checked(R/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
