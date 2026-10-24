"""Only scalar-shape compatibility for the locked native Kellner baseline."""
from pathlib import Path
import importlib.util,hashlib,json
R=Path(__file__).resolve().parent
entry=json.loads((R/"author_source_lock.json").read_text())["files"]["src/utils/methods.py"]
p=R/entry["local"]
if hashlib.sha256(p.read_bytes()).hexdigest()!=entry["sha256"]:raise ValueError("native author source identity")
spec=importlib.util.spec_from_file_location("author_native_compat_source",p)
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
class KellnerCompat(author.KellnerMethod):
 def _lstsq(self,vr,theta):
  return super()._lstsq(vr,theta).ravel()
