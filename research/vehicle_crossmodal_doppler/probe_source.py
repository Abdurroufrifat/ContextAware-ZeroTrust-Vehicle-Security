from pathlib import Path
import urllib.request,urllib.parse,re,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development"
inv=json.loads((H/"structure_inventory.json").read_text());root=json.loads((H/"source_inventory.json").read_text())
def response(ident,headers=None):
 url="https://drive.google.com/uc?export=download&id="+ident
 r=urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=45)
 if "text/html" not in r.headers.get("Content-Type",""):return r
 s=r.read(1024*1024).decode("utf-8");r.close()
 p=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s))
 if p.get("id")!=ident or p.get("confirm")!="t":raise ValueError("non-download HTML")
 return urllib.request.urlopen(urllib.request.Request("https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(p),headers=headers or {}),timeout=45)
out={}
for seq in ["Library01","Library02"]:
 entry=next(x for x in inv[seq]["sensor_data"]["entries"] if x["title"]=="aeva_stamp.csv CSV")
 with response(entry["id"]) as r:b=r.read(32*1024*1024+1)
 assert len(b)<=32*1024*1024;(R/(seq+"_aeva_stamp.csv")).write_bytes(b)
 stamps=np.array([int(x.strip()) for x in b.decode().splitlines()],dtype=np.int64);assert np.all(np.diff(stamps)>0)
 queries=np.load(H/(seq+"_replay.npz"),allow_pickle=False)["queries"]
 j=np.searchsorted(stamps,queries,side="right")-1;age=queries-stamps[np.maximum(j,0)]
 ready=(j>=0)&(age>=0)&(age<=100_000_000)
 archive=next(x for x in root[seq]["entries"] if x["title"]=="LiDAR.tar.gz Compressed archive")
 with response(archive["id"],{"Range":"bytes=0-1048575"}) as r:
  prefix=r.read(1048577);status=r.status;content_range=r.headers.get("Content-Range");length=r.headers.get("Content-Length")
 assert len(prefix)<=1048576,"range ignored; archive probe aborted"
 (R/(seq+"_archive_prefix.gz")).write_bytes(prefix)
 out[seq]={"timestamp_id":entry["id"],"timestamp_sha256":hashlib.sha256(b).hexdigest(),"frames":len(stamps),"median_gap_ms":float(np.median(np.diff(stamps))*1e-6),"maximum_gap_ms":float(np.max(np.diff(stamps))*1e-6),"requests":len(queries),"zero_compute_fresh":int(sum(ready)),"zero_compute_fraction":float(np.mean(ready)),"archive_id":archive["id"],"range_status":status,"content_range":content_range,"content_length":length,"prefix_bytes":len(prefix),"prefix_sha256":hashlib.sha256(prefix).hexdigest()}
 print(seq,json.dumps(out[seq]),flush=True)
(R/"source_probe.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
