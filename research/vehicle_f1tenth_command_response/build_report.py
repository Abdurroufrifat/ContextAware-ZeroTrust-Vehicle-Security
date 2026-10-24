from pathlib import Path
import json
P=Path(__file__).resolve().parent
a=json.loads((P/"assessment.json").read_text())
v=json.loads((P/"verification.json").read_text())
raw=json.loads((P/"raw_verification.json").read_text())
assert v["passed"] and raw["passed"]
lines=["# Physical recorded command-response results","",
"Three fixed public TU/e F1TENTH hardware runs were compared without fitting a time lag or selecting by performance. Source: Badakis, Galanis and Zhang, DOI 10.5281/zenodo.12536536, CC BY 4.0. Publisher size/MD5 and local SHA-256 checks verify the complete raw bags.",
"",
"The commanded desired longitudinal speed magnitude is compared with the separate VICON world-frame velocity magnitude. Causal bag receipt times select the last command; VICON header timestamps are retained for inspection. Direction and device-level synchronization are not qualified.",
"",
"| Run | Reference samples | Fresh / stale / unmatched | All matched MAE (m/s) | P95 (m/s) | >0.5 / >2 m/s |",
"|---|---:|---:|---:|---:|---:|"]
for r in a["runs"]:
 m=r["all_matched"];s=r["status_counts"]
 lines.append("| "+r["run"]+" | "+str(r["reference_count"])+" | "+str(s["fresh"])+" / "+str(s["stale"])+" / "+str(s["unmatched"])+" | "+format(m["mae_mps"],".6f")+" | "+format(m["p95_mps"],".6f")+" | "+str(m["over_0_5"])+" / "+str(m["over_2"])+" |")
lines.extend(["",
"All rows remain in the recorded results. Fresh and all-matched summaries are separate. These discrepancies measure command tracking; they are neither policy-estimation errors nor attack-detection outcomes. Runs are not pooled into a population claim.",
"",
"Independent verification checked "+str(v["reference_rows_checked"])+" reference rows, causal alignment and all reported discrepancy metrics. A second parser decoded the selected ROS message payloads directly with Python struct and matched every command and VICON component against the extracted CSV files.",
"",
"This establishes a real recorded desired-command/external-motion interface. It does not establish measured executed acceleration, the lead gap/speed scene interface, calibrated worst-case reference or clock bounds, or a new experiment in which CO-ACT-CPS controlled the physical vehicle. The current 10 ms SUMO motion identity and its sensor/execution error assumptions are not certified on this hardware.",
"",
"To close the full physical-control objective, an instrumented lab must supply the measurements in PHYSICAL_ACQUISITION_SPEC.md and run the frozen policy in that environment. Code, replay and additional web searches cannot substitute for that intervention.",
"",
"See RESEARCH_SCOPE.md for supported novelty and EAAI-scope claims. Existing successful simulation results and earlier failed physical/reference studies remain unchanged. No manuscript was created.",
"",
"Verification command (PowerShell):",
"",
"    & 'D:\\CO-ACT-CPS\\.venv\\Scripts\\python.exe' 'D:\\CO-ACT-CPS\\phase2_joint_history\\research\\vehicle_f1tenth_command_response\\run_all.py'",""])
(P/"RESULTS.md").write_text("\n".join(lines).replace("\\n","\n"),encoding="utf-8")
print("report written")
