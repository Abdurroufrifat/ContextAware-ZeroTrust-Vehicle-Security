from pathlib import Path
import tarfile,json,hashlib,struct
R=Path(__file__).resolve().parent
lock=json.loads((R/"download_lock.json").read_text());assert len(lock["files"])==14
out={}
for seq in ["Library01","Library02"]:
 report={}
 for row in lock["files"]:
  if row["sequence"]!=seq:continue
  p=R/row["relative_path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]
  if not p.name.endswith(".gz"):
   report[row["name"]]={"bytes":p.stat().st_size,"first_lines":p.read_text().splitlines()[:3]}
 archive=R/"data"/seq/"Radar"/"Continental.tar.gz"
 with tarfile.open(archive,"r:gz") as t:
  files=[m for m in t.getmembers() if m.isfile()]
  report["archive"]={"file_count":len(files),"first":[{"name":m.name,"size":m.size} for m in files[:3]],"total_unpacked_bytes":sum(m.size for m in files),"last":files[-1].name}
  m=next(m for m in files if m.name.endswith(".bin"))
  with t.extractfile(m) as f:b=f.read(29)
  report["first_record"]={"source_file":m.name,"values_x_y_z_velocity_range_rcs_az_el":list(struct.unpack("<fffffBff",b))}
 out[seq]=report
 print(seq,json.dumps(report,indent=2),flush=True)
(R/"layout_inspection.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
