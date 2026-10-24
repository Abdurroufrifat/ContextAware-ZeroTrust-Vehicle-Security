from pathlib import Path
import json,csv,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent.parent;T=R.parent/"vehicle_optical_timing"
v=json.loads((R/"independent_verification.json").read_text());assert v["passed"]
b=json.loads((R/"baseline_summary.json").read_text())["sequences"];a=json.loads((R/"ablation_summary.json").read_text())["sequences"];ref=json.loads((R/"reference_triangulation.json").read_text())["sequences"]
def write_csv(name,rows):
 with (R/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
baseline_rows=[{"sequence":seq,"method":name,**m} for seq,x in b.items() for name,m in x["all_attempt_metrics"].items()]
query_rows=[{"sequence":seq,"method":name,**m} for seq,x in a.items() for name,m in x["nominal_all_grid"].items()]
attack_rows=[{"case":case,"method":name,**m} for case,x in v["case_totals"].items() for name,m in x.items()]
window=ref["Library01"]["original_failed_302_requests"]
reference_rows=[{"pair":name,**m} for name,m in window["pairs"].items()]
for name,rows in [("baseline_accuracy.csv",baseline_rows),("query_ablation.csv",query_rows),("attack_ablation.csv",attack_rows),("reference_failed_window.csv",reference_rows)]:write_csv(name,rows)
bt="\n".join(f'| {x["sequence"]} | {x["method"]} | {x["scored"]}/{x["attempts"]} | {x["mae_mps"]:.6f} | {x["p95_mps"]:.6f} | {x["above2"]} |' for x in baseline_rows)
qt="\n".join(f'| {x["sequence"]} | {x["method"]} | {100*x["availability"]:.2f}% | {x["p95_mps"]:.6f} | {x["unknown_reference"]} |' for x in query_rows)
at="\n".join(f'| {x["case"]} | {x["method"]} | {x["admitted"]}/{x["supported"]} | {x["above2"]}/{x["scored"]} | {x["unknown_reference"]} |' for x in attack_rows)
rt="\n".join(f'| {x["pair"]} | {x["scored"]}/{x["attempts"]} | {x["p95_mps"]:.6f} | {x["unknown_pair"]} |' for x in reference_rows)
ci="\n".join(f'- {seq}, candidate minus {name}: paired MAE difference {x["difference"]:.6f}m/s, conditional block-resampling interval [{x["low"]:.6f},{x["high"]:.6f}].' for seq,y in b.items() for name,x in y["paired_mae_intervals"].items())
text="""# Matched motion-component, policy-ablation and reference evidence

2026-10-21. This batch is complete. Overall project completion and journal-readiness claims remain unsupported.

## Reproduced author-code classical comparator

Pinned source: https://github.com/paaraujo/deep-ego-velocity , commit2285343bb1e433e3b7dfbc8e200eb969805388c0. Re-executed its AlternativeMethod classical3DRANSAC comparator (300trials,0.1residual threshold,seed42), ordinary3Dleast-squares and original radar/IMU planar Huber on the original every20th-frame cohort:564attempts,562common reference-scored frames. All reproduced predictions agree with the prior frozen outputs within1e-10m/s. Two unknown primary labels remain visible.

This is reproduction of a classical comparator distributed with the2024paper repository, NOT the paper's proposed deep neural network or a complete published AV zero-trust defense. All three use the same filtered radar points; the candidate additionally uses IMU attitude. Results cannot isolate architectural superiority from that extra information. Fit-only timing excludes shared geometry/context preparation and is not system latency.

| Sequence | Method | Scored/attempts | MAE(m/s) | P95(m/s) | Recorded>2m/s |
|---|---|---:|---:|---:|---:|
"""+bt+"""

Post-hoc paired circular moving-block bootstrap:5ordered common-cohort frames (approximately5s),1000draws, fixed per-sequence seeds. These conditional intervals assume the selected block structure is meaningful within the observed sequence. They are not independent-route replication, unseen-environment confidence or proof of general superiority.
"""+ci+"""

## Full-query ablation with coverage

All11258original queries retained. Nominal each-method availability is shown alongside conditional disagreement; unknowns are not counted as correct. A separate common-admission table in ablation_summary.json proves that the guard's radar-valued predictions are identical to radar-only predictions wherever jointly admitted. It is therefore incorrect to describe rejection/cohort changes as improved benign speed estimation.

| Sequence | Method | Availability | Recorded primary P95(m/s) | Admitted unknown primary |
|---|---|---:|---:|---:|
"""+qt+"""

Coverage and service are from the existing compact-representation replay, assuming complete source-byte buffers before packet construction. The original gzip-path availability failure, physical transport/clock uncertainty and all original accuracy failures are unchanged.

## Identical supported-pair interface cases

| Case | Method | Admitted/supported | Admitted recorded primary>2m/s/scored | Unknown primary |
|---|---|---:|---:|---:|
"""+at+"""

This table uses identical source-ready support masks for all methods in each case;1second replay also requires prior paired support. These counts are conditional interface-ablation results, not operational coverage under arbitrary raw attacks and not certified physical false-acceptance rates. Both modalities/common clock/shared gyro remain compromise paths. Report simultaneous shifts and forged timestamps together with single-source results. Locally defined ablations are not attributed to external papers.

## Reference triangulation on the original failed window

Exactly302queries, original start6000; no new lag or label selection. All signals stay on their existing source/query associations. Optical is a predictor/guard input and is not independent evaluation ground truth. Pose and raw-position derivatives share navigation ancestry with direct INS and can differ in location/processing; neither reference is declared correct.

| Pair | Scored/attempts | P95disagreement(m/s) | Unknown pair |
|---|---:|---:|---:|
"""+rt+"""

The optical estimate also disagrees with both navigation reference representations above the0.5m/s study threshold in this window. Agreement with one signal therefore cannot justify replacing another label or erasing the original P951.1022975m/s failure. The reference gap remains unresolved; no scoring-code change or calibration adjustment is supported by this comparison. Pairwise disagreement is not an estimate of each source's true physical error.

## Literature and remaining claims

published_method_registry.json distinguishes verified classical code, local controls and recent related systems. The2025context-aware zero-trust AV paper already covers the broad framing; its publisher full-method request still returns403. CAD-TMS2026is a vehicular network/context/reputation system with code/data upon request, not a motion-regression baseline. RAIL2026is a control-response system needing planner/TTC/control cues absent from this replay. No substituted-input adaptation is called an exact reproduction and no cross-dataset paper score is inserted into these tables. This targeted screen does not establish novelty or the absence of closer literature.

Primary sources: https://research.unsw.edu.au/people/dr-marwa-keshk/publications?type=journalarticles ; https://link.springer.com/article/10.1007/s12083-026-02247-1 ; https://www.usenix.org/conference/vehiclesec26/presentation/wasif-rail .

## Verification and finite remaining requirements

731core and81focused tests pass,4core skips, one existing legacy calibration warning. The10new metrics/ablation tests first failed because their module was missing, then passed. Independent replay recomputes all564baseline predictions against saved originals,4paired intervals,11258query rows,60method/case cells and30reference-pair cells. Frozen source/results/failure hashes remain unchanged.

This evidence supports a limited offline component/conditional-policy study. It does not establish a novel whole AV security method, a reproduced deep-model ranking, physical safety, source authenticity, clock bounds or human-centered effectiveness. No manuscript, new training, production promotion, final holdout, external message, install, commit or push.

Next work that changes readiness: establish an evaluation contract supported by reference/frame/clock evidence or explicitly restrict claims to proxy disagreement; validate compatible published neural/full-defense inputs before any such reproduction claim; then freeze a justified final-confirmation protocol. Existing protected payloads stay closed under the current stop rule. More pass markers or reference swapping cannot supply the missing evidence.
"""
(R/"FINDINGS.md").write_text(text,encoding="utf-8")
decision={"matched_evidence_batch_complete":True,"classical_author_comparator_reproduced":True,"baseline_attempts":564,"common_scored_frames":562,"queries":11258,"paired_ablation_method_case_cells":60,"reference_pair_cells":30,"all_original_outputs_and_failures_retained":True,"benign_guard_is_new_speed_estimator":False,"reference_gap_resolved":False,"neural_model_reproduced":False,"full_published_AV_defense_reproduced":False,"novelty_established":False,"overall_project_complete":False,"physical_witness_admitted":False,"Library03_payloads_accessed":False,"production_changed":False}
(R/"RESEARCH_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
readiness="""# Remaining evidence requirements

The matched-comparison batch is finished; this is not a claim that every requirement of the full research project is satisfied.

| Requirement | Current evidence | Remaining requirement |
|---|---|---|
| Classical motion comparator | Pinned author3DRANSAC, OLS and radar/IMU reproduced on564fixed frames;562scored | Component comparison complete within explicit extra-IMU/fit-only timing limits |
| Matched policy controls | Radar-only/optical-only/guard;11258query grid; ten identical-support interface cases | Keep conditional scope and common-mode/forged-time failures |
| Evaluation reference | Multiple navigation representations and optical/radar estimates disagree in old failed window | Independently justified reference/frame/clock bounds, or a predeclared limited proxy-agreement evaluation claim |
| Compatible published neural/full security comparison | Classical comparator is not deep model; recent AV network/control methods require different inputs | Verify native preprocessing/checkpoint membership/task compatibility before executing/claiming a full reproduction |
| Novelty | Broad context-aware/multisource zero trust already exists | Demonstrate a specific supported algorithmic/system contribution against compatible prior methods |
| Final confirmation | Current protocol has retained unresolved development failures and unknowns | Freeze justified scope/stop rule before touching reserved payloads; no silent threshold or label changes |

Production witness remains disabled. Human review/study is not added; human-centered effectiveness is not claimed.
"""
(R/"READINESS.md").write_text(readiness,encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_MATCHED.md"
if not backup.exists():shutil.copyfile(status,backup)
prefix="""Latest: matched author-code/classical baseline reproduction, full-query ablation and reference triangulation completed,2026-10-21.

Reproduced564fixed classical-comparator frames;562reference-scored. Candidate radar/IMU P95=.299051/.202939m/s; author3DRANSAC=.338696/.241079; OLS=.728428/.460668. Candidate has extra IMU; this is not reproduction of the proposed deep neural model or a complete published AV defense.
Full11258query grid: radar-only/optical-only/agreement guard compared with coverage and identical-support interface cases. Guard and radar speed estimates are identical on common admitted rows; observed benefit is conditional attack rejection, not new benign estimation accuracy.
Original302query failure remains. Optical P95disagrees .890430m/s with primary pose derivative and .643688m/s with direct INS in that window; no source is certified true and no label swap is justified.
731core/81focused tests pass,4skip, one existing legacy warning. Independent564frame/11258query/60case-cell/30reference-cell/4interval arithmetic/provenance replay passes.
Overall project complete:NO. Reference/clock/source evidence, compatible full neural/security reproduction, specific novelty and justified final-confirmation scope remain open. Production witness disabled; Library03 and older protected payloads untouched.
Evidence: research/vehicle_matched_evidence/FINDINGS.md, READINESS.md, RESEARCH_DECISION.json, independent_verification.json and CSV result tables.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"files":len(files),"baseline_attempts":564,"queries":11258,"focused_passed":81,"core_passed":731,"core_skipped":4,"current_status_sha256":hashlib.sha256(status.read_bytes()).hexdigest()}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(decision,indent=2));print(json.dumps(completion,indent=2))
