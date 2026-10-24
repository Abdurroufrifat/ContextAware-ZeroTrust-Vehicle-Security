from pathlib import Path
import urllib.request,zipfile,io,json,hashlib
R=Path(__file__).resolve().parent
url="https://codeload.github.com/robotics-upo/ars548_ros/zip/refs/heads/noetic"
with urllib.request.urlopen(url,timeout=30) as r:b=r.read(8*1024*1024+1)
assert len(b)<=8*1024*1024
(R/"ars548_driver_source.zip").write_bytes(b)
out={"url":url,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"evidence":[]}
with zipfile.ZipFile(io.BytesIO(b)) as z:
 for n in z.namelist():
  if not n.endswith((".cpp",".h",".msg",".md")):continue
  s=z.read(n).decode("utf-8",errors="replace");lines=s.splitlines()
  for i,l in enumerate(lines):
   if any(k in l for k in ["f_RangeRate","f_AzimuthAngle","point.x","velocity =","DetectionPointCloud"]):
    out["evidence"].append({"file":n,"line":i+1,"context":"\n".join(lines[max(0,i-2):i+4])})
(R/"driver_evidence.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
