from pathlib import Path
import tarfile,time,json,hashlib
R=Path(__file__).resolve().parent;p=R/"data"/"Library01"/"Radar"/"Continental.tar.gz"
with tarfile.open(p,"r:gz") as t:
 members=[m for m in t.getmembers() if m.isfile()];ordered=sorted(members,key=lambda m:m.name)
 times=[];hashes=[]
 for m in ordered[:30]:
  start=time.perf_counter();b=t.extractfile(m).read();times.append(time.perf_counter()-start);hashes.append(hashlib.sha256(b).hexdigest())
start=time.perf_counter();count=0
with tarfile.open(p,"r|gz") as t:
 for m in t:
  if m.isfile():t.extractfile(m).read();count+=1
out={"random_seek_first30_seconds":sum(times),"stream_all_files_seconds":time.perf_counter()-start,"files_streamed":count,"archive_positions_first10_sorted":[m.offset_data for m in ordered[:10]],"raw_first30_hashes":hashes}
(R/"archive_performance_probe.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
