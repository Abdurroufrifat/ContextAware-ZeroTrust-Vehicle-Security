from pathlib import Path
R=Path(__file__).resolve().parent;p=R/"finish_study.py";s=p.read_text()
s=s.replace('total_frames=sum(s["frames"] for s in summary["sequences"].values())','segment_passes=sum(s["ready_fraction"]>=.95 and s["p95_mps"] is not None and s["p95_mps"]<=.5 and s["above_2_mps"]==0 for s in segments)\ntotal_frames=sum(s["frames"] for s in summary["sequences"].values())')
old='All{len(segments)}retained60-second segments meet >=95% replay availability and P95<=0.5m/s, with zero scored discrepancies>2m/s: {all(s["ready_fraction"]>=.95 and s["p95_mps"] is not None and s["p95_mps"]<=.5 and s["above_2_mps"]==0 for s in segments)}.'
s=s.replace(old,'Retained60-second segments meeting >=95% replay availability, P95<=0.5m/s and zero scored discrepancies>2m/s: {segment_passes}/{len(segments)}. Every segment remains in development_summary.json.')
s=s.replace('All11retained60s segments pass the three reported numeric criteria.','Retained60s segments passing the three reported numeric criteria: {segment_passes}/{len(segments)}.')
p.write_text(s,encoding="utf-8")
src=R.parent/"vehicle_independent_source_qualification"/"audit_package.py"
code=src.read_text().replace('"pilot_measurements_previously_verified":50','"development_inputs_previously_verified":14')
(R/"audit_package.py").write_text(code,encoding="utf-8")
print("Closure reports computed segment outcomes and explicit file identities.")
