from pathlib import Path
import sys,json,hashlib,os
R=Path(__file__).resolve().parent;P=R.parent.parent;OLD=R.parent/"vehicle_sumo_gateway_study"
sys.path.insert(0,str(OLD))
from sg_common import checked,install_guard
install_guard()
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def dump(path,obj):checked(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
