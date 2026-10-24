from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent
s=json.loads((R/"development_summary.json").read_text());failed=[]
def metric(p,y):
 keep=np.isfinite(p)&np.isfinite(y);e=abs(p[keep]-y[keep])
 return {"scored":int(keep.sum()),"mae_mps":float(e.mean()) if len(e) else None,"p95_mps":float(np.quantile(e,.95)) if len(e) else None,"max_mps":float(max(e)) if len(e) else None}
for seq,v in s["sequences"].items():
 a=np.load(R/(seq+"_replay.npz"),allow_pickle=False)
 for seg in v["requests"]["segments_60s"]:
  if seg["ready_fraction"]>=.95 and seg["p95_mps"] is not None and seg["p95_mps"]<=.5 and seg["above_2_mps"]==0:continue
  start=seg["start_request"];end=start+seg["requests"];t0=int(a["queries"][start]);t1=int(a["queries"][end-1])
  mask=(a["stamps"]>=t0)&(a["stamps"]<=t1)&a["quality"]
  failed.append({"sequence":seq,"request_segment":seg,"start_time_ns":t0,"end_time_ns":t1,"duration_seconds":(t1-t0)*1e-9,"frame_candidate_vs_primary":metric(a["speed"][mask],a["reference_primary"][mask]),"frame_candidate_vs_INS":metric(a["speed"][mask],a["reference_ins"][mask]),"frame_primary_vs_INS":metric(a["reference_primary"][mask],a["reference_ins"][mask]),"interpretation":"Post-run diagnostic. Primary and INS references differ in representation and sensor location; this does not prove either is correct. Original reference and failure are retained, without retuning."})
(R/"failed_segments.json").write_text(json.dumps(failed,indent=2),encoding="utf-8")
ready=R/"candidate_readiness.json";obj=json.loads(ready.read_text());obj["all_retained_segments_pass"]=not failed;obj["failed_segments"]=len(failed);ready.write_text(json.dumps(obj,indent=2),encoding="utf-8")
note="\nRetained segment failure: Library01 final302requests (about15seconds),301primary-scored, P95=1.102297507859495m/s, maximum=1.1932529748089449m/s. It fails the unchanged0.5m/s limit despite the good whole-sequence average. Diagnostic reference comparisons are in failed_segments.json; no reference was switched to erase this failure.\n"
find=R/"FINDINGS.md";find.write_text(find.read_text(encoding="utf-8")+note,encoding="utf-8")
status=R.parent.parent/"CURRENT_PROJECT_STATUS.md";text=status.read_text(encoding="utf-8")
text=text.replace("Reference is interval velocity derived",note.strip()+"\nReference is interval velocity derived",1)
text=text.replace("Next: evaluate this frozen candidate","Next: diagnose and address the retained final-segment failure without tuning on reserved data, then evaluate this frozen candidate",1)
status.write_text(text,encoding="utf-8")
c=R/"completion.json";obj=json.loads(c.read_text());obj["all_retained_segments_pass"]=not failed;obj["current_status_sha256"]=hashlib.sha256(status.read_bytes()).hexdigest();c.write_text(json.dumps(obj,indent=2),encoding="utf-8")
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print(json.dumps(failed,indent=2))
