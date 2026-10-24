"""Install one hash-pinned native wheel locally, without changing NumPy/SciPy dependencies."""
import hashlib,importlib.metadata,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/"runtime";RUNTIME.mkdir(exist_ok=True)
(ROOT/".gitignore").write_text("runtime/\n")
version="1.0.1";name="small_gicp-1.0.1-cp312-cp312-win_amd64.whl"
sha="e9812b1d758808811538dccf0b433181d3464fb9444a932b4b0ec40dab00f6fa"
url="https://files.pythonhosted.org/packages/1e/70/d8ff59d76a1f7a722eeaa35128d2b224238ffdb7ee205add90602327d648/"+name
wheel=RUNTIME/name
if not wheel.exists():
    with urllib.request.urlopen(url,timeout=30) as r: wheel.write_bytes(r.read())
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==sha
with zipfile.ZipFile(wheel) as z:
    licenses=[n for n in z.namelist() if "license" in n.lower()]
    for n in licenses: (ROOT/("BACKEND_"+Path(n).name)).write_bytes(z.read(n))
temp=RUNTIME/"pip_temp";temp.mkdir(exist_ok=True)
env=os.environ.copy();env["TMP"]=str(temp);env["TEMP"]=str(temp)
p=subprocess.run([sys.executable,"-m","pip","install","--no-deps","--no-cache-dir",str(wheel)],
    env=env,capture_output=True,text=True)
(ROOT/"install.log").write_text(p.stdout+p.stderr)
print(p.stdout+p.stderr)
if p.returncode: sys.exit(p.returncode)
import small_gicp
report=dict(package="small-gicp",version=importlib.metadata.version("small-gicp"),wheel=name,
    sha256=sha,url=url,numpy_version=importlib.metadata.version("numpy"),scipy_version=importlib.metadata.version("scipy"),
    dependencies_changed=False,license_members=licenses)
assert report["version"]==version
(ROOT/"backend_install.json").write_text(json.dumps(report,indent=2)+"\n")
(ROOT/"API.txt").write_text(small_gicp.align.__doc__+"\n\n"+small_gicp.preprocess_points.__doc__)
print("API",small_gicp.align.__doc__)
print("RESULT",small_gicp.RegistrationResult.__doc__ if hasattr(small_gicp,"RegistrationResult") else "No exported result class")
