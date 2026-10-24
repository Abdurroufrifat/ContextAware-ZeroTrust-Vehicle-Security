from pathlib import Path
import os,sys,json,hashlib
R=Path(__file__).resolve().parent;P=R.parent.parent
KW=R.parent/"vehicle_kinematic_witness_study";TEMPORAL=R.parent/"vehicle_temporal_gateway_study";OLD=R.parent/"vehicle_sumo_gateway_study"
for path in (KW,TEMPORAL,OLD):sys.path.insert(0,str(path))
from sg_common import checked,install_guard
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def dump(path,obj):checked(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
