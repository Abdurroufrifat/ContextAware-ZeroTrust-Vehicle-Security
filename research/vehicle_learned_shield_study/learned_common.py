from pathlib import Path
import sys,json,hashlib,os
R=Path(__file__).resolve().parent;P=R.parent.parent
KW=R.parent/"vehicle_kinematic_witness_study"
sys.path.insert(0,str(KW));sys.path.insert(0,str(R))
from kw_common import checked,install_guard,BINARY
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def write(name,value):
 checked(R/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
