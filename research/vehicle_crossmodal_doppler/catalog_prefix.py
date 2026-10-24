from pathlib import Path
import json,tarfile,hashlib
R=Path(__file__).resolve().parent;out={}
for seq in ["Library01","Library02"]:
 p=R/(seq+"_Aeva.tar.partial");assert p.stat().st_size<=512*1024**2
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 rows=[];terminal=None
 try:
  with tarfile.open(p,"r|gz") as archive:
   for m in archive:
    if m.isdir():continue
    n=Path(m.name);assert m.isfile() and not n.is_absolute() and ".." not in n.parts and n.suffix==".bin" and m.size<=32*1024**2
    b=archive.extractfile(m).read();assert len(b)==m.size
    rows.append({"stamp_ns":int(n.stem),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()})
 except (EOFError,tarfile.ReadError) as ex:terminal=type(ex).__name__+": "+str(ex)
 assert terminal is not None and len({x["stamp_ns"] for x in rows})==len(rows)
 out[seq]={"file":p.name,"bytes":p.stat().st_size,"sha256":h.hexdigest(),"complete_members":rows,"truncated_tail":terminal,"full_archive":False}
 print(seq,"complete_frames",len(rows),"span_s",(max(x["stamp_ns"] for x in rows)-min(x["stamp_ns"] for x in rows))*1e-9,flush=True)
(R/"prefix_data_lock.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
