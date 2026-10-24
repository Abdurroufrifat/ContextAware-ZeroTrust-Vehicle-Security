"""Import only the two explicitly identified development archives; no network."""
from pathlib import Path
import hashlib,json,shutil,gzip,time
R=Path(__file__).resolve().parent
sources={"Library01":Path(r"C:\Users\abdurrouf\Downloads\LiDAR-001.tar.gz"),"Library02":Path(r"C:\Users\abdurrouf\Downloads\LiDAR-001 (1).tar.gz")}
probe=json.loads((R/"source_probe.json").read_text());pilot=json.loads((R/"pilot_data_lock.json").read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
result={};audit={}
assert shutil.disk_usage(R).free>24*1024**3
for seq,source in sources.items():
 start=time.perf_counter();e=probe[seq];expected=int(e["content_range"].rsplit("/",1)[1]);p=pilot[seq]
 assert source.is_file() and source.stat().st_size==expected,(seq,"unexpected archive size")
 with source.open("rb") as f:prefix_hash=hashlib.sha256(f.read(p["prefix_bytes"])).hexdigest()
 assert prefix_hash==p["prefix_sha256"],(seq,"does not match previously locked development source")
 print(seq,"source identity and exact size verified",flush=True)
 total=0
 with gzip.open(source,"rb") as g:
  while True:
   block=g.read(8*1024**2)
   if not block:break
   total+=len(block);assert total<=100*1024**3
 print(seq,"gzip end-of-stream and CRC verified; raw bytes",total,flush=True)
 original_hash=sha(source);target=R/(seq+"_Aeva.tar.gz")
 if target.exists():assert target.stat().st_size==expected and sha(target)==original_hash
 else:
  temp=target.with_suffix(".manual_import")
  shutil.copyfile(source,temp)
  assert temp.stat().st_size==expected and sha(temp)==original_hash
  temp.replace(target)
 result[seq]={"file":target.name,"bytes":expected,"sha256":original_hash,"source_id":e["archive_id"]}
 audit[seq]={"manual_source_path":str(source),"imported_path":str(target),"bytes":expected,"sha256":original_hash,"matched_locked_pilot_prefix":True,"gzip_crc_and_eof_verified":True,"decompressed_tar_bytes":total,"elapsed_seconds":time.perf_counter()-start}
 print(seq,"IMPORTED",round(time.perf_counter()-start,1),"seconds",flush=True)
lock=R/"full_data_lock.json"
if lock.exists():assert json.loads(lock.read_text())==result
else:lock.write_text(json.dumps(result,indent=2),encoding="utf-8")
(R/"manual_import_audit.json").write_text(json.dumps({"sequences":audit,"Library03_payloads_accessed":False,"original_downloads_preserved":True},indent=2),encoding="utf-8")
print("MANUAL_IMPORT_FINISHED",flush=True)
