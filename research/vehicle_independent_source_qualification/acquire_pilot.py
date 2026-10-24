from pathlib import Path
import json,urllib.request,urllib.parse,re,hashlib,concurrent.futures
R=Path(__file__).resolve().parent;raw=R/"pilot_data";raw.mkdir(exist_ok=True)
entries=sorted(json.loads((R/"schema_probe.json").read_text())["pcd_entries"],key=lambda x:x["title"])
assert [x["title"].split()[0] for x in entries]==[f"{i:06d}.pcd" for i in range(50)]
manifest={"protocol_sha256":hashlib.sha256((R/"PILOT_PROTOCOL.md").read_bytes()).hexdigest(),"entries":entries,"limits":{"per_file":4*1024*1024,"total":64*1024*1024}}
(R/"pilot_membership_lock.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
def get(url):
 with urllib.request.urlopen(url,timeout=30) as r:b=r.read(4*1024*1024+1)
 if len(b)>4*1024*1024:raise ValueError("download size cap")
 return b
def one(e):
 name=e["title"].split()[0];p=raw/name
 try:
  if p.exists():b=p.read_bytes()
  elif name=="000000.pcd":b=(R/"msc_first_radar.pcd").read_bytes();p.write_bytes(b)
  else:
   url="https://drive.google.com/uc?export=download&id="+e["id"];b=get(url)
   if b.lstrip().startswith(b"<"):
    s=b.decode("utf-8")
    if "Google Drive - Virus scan warning" not in s:raise ValueError("non-PCD HTML")
    params=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s))
    if params.get("id")!=e["id"]:raise ValueError("download identity mismatch")
    b=get("https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(params))
   if not b.startswith((b"VERSION",b"# .PCD")):raise ValueError("not PCD")
   p.write_bytes(b)
  return {"id":e["id"],"file":name,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
 except Exception as ex:return {"id":e["id"],"file":name,"error":str(ex)}
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(one,entries):
  out.append(row);print(row["file"],row.get("bytes",row.get("error")),flush=True)
assert sum(x.get("bytes",0) for x in out)<=64*1024*1024
(R/"pilot_data_lock.json").write_text(json.dumps({"files":out,"total_bytes":sum(x.get("bytes",0) for x in out)},indent=2),encoding="utf-8")
print("downloaded",sum("sha256" in x for x in out),"of",len(out),flush=True)
