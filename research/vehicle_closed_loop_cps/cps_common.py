from pathlib import Path
import sys,json,hashlib,os
R=Path(__file__).resolve().parent
P=R.parent.parent
CS=R.parent/"vehicle_command_scene_learning"
sys.path.insert(0,str(CS))
from cs_common import checked,install_guard,BINARY
KEYS={n:hashlib.sha256(("COACT-SUMO-PUBLIC-TEST-"+n).encode()).digest() for n in ("candidate","anchor","command_logger","scene","radar","actuator")}
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def dump(path,value):checked(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
