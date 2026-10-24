from pathlib import Path
import json,urllib.request,hashlib
R=Path(__file__).resolve().parent
urls={
"published_tree":"https://api.github.com/repos/paaraujo/deep-ego-velocity/git/trees/main?recursive=1",
"published_readme":"https://raw.githubusercontent.com/paaraujo/deep-ego-velocity/main/README.md",
"published_license":"https://raw.githubusercontent.com/paaraujo/deep-ego-velocity/main/LICENSE.md",
}
out={}
for k,u in urls.items():
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"CO-ACT-CPS-source-screen"}),timeout=25) as r:b=r.read(4*1024*1024+1)
  assert len(b)<=4*1024*1024
  (R/(k+".json" if k=="published_tree" else k+".md")).write_bytes(b)
  out[k]={"url":u,"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
  if k=="published_tree":
   t=json.loads(b);out[k]["commit"]=t["sha"]
   out[k]["paths"]=[x["path"] for x in t["tree"] if "msc" in x["path"].lower() or "utils" in x["path"].lower()]
 except Exception as e:out[k]={"url":u,"error":str(e)}
 print(k,json.dumps(out[k]),flush=True)
(R/"published_adapter_probe.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
