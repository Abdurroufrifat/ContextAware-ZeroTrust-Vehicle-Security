from pathlib import Path
import urllib.request,zipfile,hashlib,json,time
R=Path(__file__).resolve().parent;commit="2285343bb1e433e3b7dfbc8e200eb969805388c0"
p=R/"author_repository.zip";cap=256*1024**2
if not p.exists():
 req=urllib.request.Request("https://codeload.github.com/paaraujo/deep-ego-velocity/zip/"+commit,headers={"User-Agent":"CO-ACT-CPS-research"})
 with urllib.request.urlopen(req,timeout=45) as r,p.with_suffix(".partial").open("wb") as f:
  if r.headers.get("Content-Length"):assert int(r.headers["Content-Length"])<=cap
  total=0
  while True:
   b=r.read(1024**2)
   if not b:break
   total+=len(b);assert total<=cap;f.write(b)
 p.with_suffix(".partial").replace(p)
rows=[];saved=[]
with zipfile.ZipFile(p) as z:
 assert sum(x.file_size for x in z.infolist())<=1024**3
 for x in z.infolist():
  q=Path(x.filename);assert not q.is_absolute() and ".." not in q.parts
  name="/".join(q.parts[1:])
  if x.is_dir():continue
  rows.append({"path":name,"bytes":x.file_size,"compressed_bytes":x.compress_size,"archive_date":x.date_time})
  if (name.startswith("src/") and q.suffix==".py") or name in ["README.md","LICENSE.md","requirements.txt","data/mscrad4r/sets.json","data/navinst/sets.json"]:
   assert x.file_size<=2*1024**2
   dest=R/"author_source"/name;dest.parent.mkdir(parents=True,exist_ok=True);data=z.read(x);dest.write_bytes(data)
   saved.append({"path":name,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
out={"commit":commit,"zip_bytes":p.stat().st_size,"zip_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"entries":rows,"extracted_source":saved,"weights_extracted":False}
(R/"source_inventory.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print("ZIP_BYTES",out["zip_bytes"],flush=True)
print("WEIGHT_INVENTORY",json.dumps([x for x in rows if x["path"].startswith("checkpoints/")]),flush=True)
print("SOURCE_FILES",json.dumps([x["path"] for x in saved]),flush=True)
