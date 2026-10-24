import json,subprocess,sys,urllib.request,hashlib,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent
pin=json.loads((R/"backend_pin.json").read_text())
subprocess.run([sys.executable,"-m","pip","install","--no-deps","--no-cache-dir",pin["wheel"]],check=True)
for module,package in [("pydantic","pydantic>=2"),("pydantic_settings","pydantic-settings")]:
 if importlib.util.find_spec(module) is None:
  subprocess.run([sys.executable,"-m","pip","install","--no-cache-dir",package],check=True)
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
cfg=load_config(None)
cfg.registration.max_num_threads=4
print("CONFIG",cfg.model_dump())
u="https://raw.githubusercontent.com/PRBonn/kiss-icp/v1.3.0/cpp/kiss_icp/core/Preprocessing.cpp"
b=urllib.request.urlopen(u).read()
(R/"BACKEND_Preprocessing.cpp").write_bytes(b)
print("CPP",b.decode())
(R/"backend_environment.json").write_text(json.dumps(dict(config=cfg.model_dump(),preprocessing_url=u,preprocessing_sha256=hashlib.sha256(b).hexdigest()),indent=2))
