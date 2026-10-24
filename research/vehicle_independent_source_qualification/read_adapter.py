from pathlib import Path
import urllib.request,json,hashlib
R=Path(__file__).resolve().parent
commit=json.loads((R/"published_adapter_probe.json").read_text())["published_tree"]["commit"]
paths=["src/mscrad4r/pcd/pcd_read.cpp","src/mscrad4r/sync.py","src/mscrad4r/dataset.py","src/mscrad4r/compare.py","src/utils/methods.py","data/mscrad4r/sets.json"]
out={}
for p in paths:
 u=f"https://raw.githubusercontent.com/paaraujo/deep-ego-velocity/{commit}/{p}"
 with urllib.request.urlopen(u,timeout=25) as r:b=r.read(1024*1024+1)
 assert len(b)<=1024*1024
 name="author_"+p.replace("/","_")
 (R/name).write_bytes(b)
 out[p]={"url":u,"local":name,"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
 print("\nFILE",p,"\n",b.decode("utf-8") if any(x in p for x in ["pcd_read","sync.py","sets.json"]) else "\n".join(x for x in b.decode("utf-8").splitlines() if any(w in x.lower() for w in ["doppler","velocity","ransac","x_vel","ref_","theta","angle"])),flush=True)
(R/"author_source_lock.json").write_text(json.dumps({"commit":commit,"files":out},indent=2),encoding="utf-8")
