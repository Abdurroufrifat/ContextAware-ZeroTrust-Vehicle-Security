from pathlib import Path
import urllib.request,urllib.parse,json,re,time,hashlib,concurrent.futures,shutil
R=Path(__file__).resolve().parent
assert json.loads((R/"pilot_summary.json").read_text())["full_stream_trigger"] is True
probes=json.loads((R/"source_probe.json").read_text())
assert set(probes)=={"Library01","Library02"}
CAP=8*1024**3;CHUNK=64*1024**2
class PublisherQuotaError(RuntimeError):pass
def url(ident):
 with urllib.request.urlopen("https://drive.google.com/uc?export=download&id="+ident+"&request="+str(time.time_ns()),timeout=45) as r:
  if "text/html" not in r.headers.get("Content-Type",""):return r.url
  s=r.read(1024*1024).decode()
 params=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s));assert params.get("id")==ident
 return "https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(params)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
def one(seq):
 e=probes[seq];total=int(e["content_range"].rsplit("/",1)[1]);assert total<=CAP
 p=R/(seq+"_Aeva.tar.gz");part=p.with_suffix(".downloadpartial");prefix=R/(seq+"_pilot_prefix.gz")
 if not p.exists():
  if not part.exists():
   seed=prefix;prefix_lock=R/"prefix_data_lock.json"
   if prefix_lock.exists():
    old=json.loads(prefix_lock.read_text())[seq];candidate=R/old["file"]
    assert sha(candidate)==old["sha256"];seed=candidate
   shutil.copyfile(seed,part)
  assert part.stat().st_size<=total
  with part.open("rb") as f:assert hashlib.sha256(f.read(prefix.stat().st_size)).hexdigest()==sha(prefix)
  endpoint=url(e["archive_id"]);start=time.perf_counter()
  while part.stat().st_size<total:
   offset=part.stat().st_size;end=min(total,offset+CHUNK)-1
   for attempt in range(4):
    try:
     req=urllib.request.Request(endpoint+"&range_start="+str(offset)+"&request="+str(time.time_ns()),headers={"Range":f"bytes={offset}-{end}","Cache-Control":"no-cache"})
     with urllib.request.urlopen(req,timeout=60) as r:
      if "text/html" in r.headers.get("Content-Type",""):
       page=r.read(16*1024).decode(errors="replace")
       if "Quota exceeded" in page:raise PublisherQuotaError("Google Drive publisher download quota exceeded; stop and retain verified partial bytes")
       raise ValueError("unexpected HTML download response")
      assert r.status==206 and r.headers.get("Content-Range")==f"bytes {offset}-{end}/{total}", (r.status, r.headers.get("Content-Range"), f"bytes {offset}-{end}/{total}")
      with part.open("r+b") as f:
       f.seek(offset);remaining=end-offset+1
       while remaining:
        b=r.read(min(1024**2,remaining))
        if not b:raise EOFError("short download")
        f.write(b);remaining-=len(b)
       assert not r.read(1)
     break
    except Exception as ex:
     with part.open("r+b") as f:f.truncate(offset)
     print(seq,"retry",attempt,type(ex).__name__,str(ex),flush=True)
     if isinstance(ex,PublisherQuotaError) or attempt==3:raise
     endpoint=url(e["archive_id"])
   print(seq,"download",part.stat().st_size,"/",total,"elapsed",round(time.perf_counter()-start,1),flush=True)
  part.replace(p)
 assert p.stat().st_size==total
 out={"file":p.name,"bytes":total,"sha256":sha(p),"source_id":e["archive_id"]}
 (R/(seq+"_full_download.json")).write_text(json.dumps(out,indent=2),encoding="utf-8")
 print(seq,"DOWNLOAD_VERIFIED",flush=True)
 return seq,out
assert sum(int(e["content_range"].rsplit("/",1)[1]) for e in probes.values())<=12*1024**3
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 result=dict(pool.map(one,["Library01","Library02"]))
(R/"full_data_lock.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print("FULL_ACQUISITION_FINISHED",flush=True)
