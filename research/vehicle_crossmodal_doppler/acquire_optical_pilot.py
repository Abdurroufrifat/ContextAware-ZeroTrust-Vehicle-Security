from pathlib import Path
import urllib.request,urllib.parse,json,re,time,hashlib,concurrent.futures,tarfile
R=Path(__file__).resolve().parent;probes=json.loads((R/"source_probe.json").read_text())
CAP=128*1024*1024
def download_url(ident):
 with urllib.request.urlopen("https://drive.google.com/uc?export=download&id="+ident,timeout=45) as r:
  if "text/html" not in r.headers.get("Content-Type",""):return r.url
  s=r.read(1024*1024).decode()
 params=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s));assert params.get("id")==ident
 return "https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(params)
def one(seq):
 entry=probes[seq];p=R/(seq+"_pilot_prefix.gz");start=time.perf_counter()
 if not p.exists():
  part=p.with_suffix(".partial")
  prefix=(R/(seq+"_archive_prefix.gz")).read_bytes();assert hashlib.sha256(prefix).hexdigest()==entry["prefix_sha256"]
  with part.open("wb") as f:f.write(prefix)
  req=urllib.request.Request(download_url(entry["archive_id"]),headers={"Range":f"bytes={len(prefix)}-{CAP-1}"})
  with urllib.request.urlopen(req,timeout=60) as r,part.open("ab") as f:
   assert r.status==206 and r.headers.get("Content-Range","").startswith(f"bytes {len(prefix)}-{CAP-1}/")
   total=len(prefix)
   while True:
    b=r.read(1024*1024)
    if not b:break
    total+=len(b);assert total<=CAP;f.write(b)
    if total%(32*1024*1024)==0:print(seq,"MiB",total//(1024*1024),flush=True)
  assert part.stat().st_size==CAP;part.replace(p)
 output=R/"pilot_frames"/seq;output.mkdir(parents=True,exist_ok=True);rows=[];terminal=None;total_raw=0
 try:
  with tarfile.open(p,"r|gz") as archive:
   for member in archive:
    if member.isdir():continue
    name=Path(member.name)
    if not member.isfile() or name.is_absolute() or ".." in name.parts or name.suffix!=".bin" or member.size>32*1024*1024:raise ValueError("unsafe member")
    b=archive.extractfile(member).read()
    assert len(b)==member.size
    stamp=int(name.stem);dest=output/(str(stamp)+".bin");dest.write_bytes(b);total_raw+=len(b)
    assert total_raw<=512*1024*1024
    rows.append({"stamp_ns":stamp,"archive_name":member.name,"relative_path":str(dest.relative_to(R)),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()})
    if len(rows)>=64:break
 except (EOFError,tarfile.ReadError) as e:terminal=str(e)
 out={"sequence":seq,"prefix_bytes":p.stat().st_size,"prefix_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"frames":rows,"terminal_truncation":terminal,"elapsed_seconds":time.perf_counter()-start}
 print(seq,"complete pilot frames",len(rows),"elapsed",out["elapsed_seconds"],flush=True);return seq,out
result={}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for seq,out in pool.map(one,["Library01","Library02"]):result[seq]=out
(R/"pilot_data_lock.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print("PILOT_ACQUISITION_FINISHED",flush=True)
