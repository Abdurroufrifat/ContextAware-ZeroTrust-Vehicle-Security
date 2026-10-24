from pathlib import Path
import sys,json,hashlib,os
B=Path(__file__).resolve().parent;P=B.parent.parent;OLD=P/"research"/"vehicle_closed_loop_cps"
sys.path.insert(0,str(OLD))
from cps_common import KEYS,checked,install_guard,BINARY
sys.path.insert(0,str(B))
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def dump(path,value):
    target=checked(path);tmp=target.with_name(target.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8");os.replace(tmp,target)
