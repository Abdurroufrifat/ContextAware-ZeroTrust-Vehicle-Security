from pathlib import Path
import os,sys,json,hashlib
R=Path(__file__).resolve().parent;P=R.parent.parent
BINARY=Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
NETCONVERT=BINARY.with_name("netconvert.exe")
def forbidden(path):
 s=str(path).replace("\\","/").lower()
 return any(x in s for x in ("library03","vehicle_final_evaluation","rav4")) or ("urban02" in s and "night" in s)
def checked(path):
 p=Path(path)
 if forbidden(p) or forbidden(p.resolve()) or not p.resolve().is_relative_to(P.resolve()):raise PermissionError("outside study scope")
 return p
def sha(path):return hashlib.sha256(checked(path).read_bytes()).hexdigest()
def write(name,data):checked(R/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def install_guard(local_sumo=False):
 def guard(event,args):
  if event=="socket.connect":
   address=args[1]
   if not local_sumo or not isinstance(address,tuple) or address[0] not in ("127.0.0.1","::1","localhost"):raise PermissionError("external network disabled")
  if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
   path=os.fsdecode(args[0])
   if forbidden(path) or forbidden(Path(path).resolve()):raise PermissionError("reserved payload")
 sys.addaudithook(guard)
