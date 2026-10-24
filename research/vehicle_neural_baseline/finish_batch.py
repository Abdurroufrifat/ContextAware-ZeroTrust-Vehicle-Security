from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent.parent;M=R.parent/"vehicle_matched_evidence"
v=json.loads((R/"independent_verification.json").read_text());assert v["passed"]
a=json.loads((R/"model_admission_audit.json").read_text());assert all(x["strict_state_dict_loaded"] and x["synthetic_forward_passed"] for x in a["models"])
table="\n".join(f'| {x["id"]} | {x["parameters"]:,} | {x["input_shape"]} | Yes | '+("; ".join(sorted({r["field"]+": "+r["reason"] for r in x["target_admission"]["reasons"]})))+" |" for x in a["models"])
text="""# Native neural architecture/weight audit and benchmark input admission

2026-10-21. This bounded admission audit is complete. It does not report a real-dataset neural performance comparison or establish full project completion.

## What was validated

Retrieved the pinned author source archive at commit2285343bb1e433e3b7dfbc8e200eb969805388c0 from https://github.com/paaraujo/deep-ego-velocity . Preserved the MIT license, archive/source hashes and selected weight hashes. Source snapshot archive metadata is dated June2024. Checkpoint names were selected from author default test configurations before any tensor outputs; no outcome-based checkpoint selection.

The unchanged RadarNet architecture strictly loads all three selected genuine pretrained state dictionaries on CPU using torch.load(weights_only=True). Author test parameter assignments are read as constants from source. Only the pure author mapping method is executed on artificial records; dataset file loaders, eval(line), pickle entrypoints and training routines are never invoked. All three return finite1x1outputs, repeat deterministically and preserve all parameters and buffers in inference mode. Independent verification reloads each pair and rejects a deliberately wrong channel count. These are synthetic tensor/implementation checks, not a reproduction of reported publication scores, training or performance.

| Native model | Parameters | Input tensor shape | Strict load/synthetic pass | Current target-data admission blocker |
|---|---:|---|---|---|
"""+table+"""

Two3Dmodels require additional measurement channels: MSC-RAD4R uses native power, NavINST uses SNR. The inspected HeRCULES Continental29byte format has x,y,z,radialvelocity,range,unsigned-byteRCS,azimuth,elevation. It contains neither power nor SNR. No channel is relabelled, zero-filled or reconstructed from evaluation labels.

The RadarScenes2Dmodel maps sensor-frame x/y, raw radial velocity and RCS. RadarScenes documents RCS in dBsm: https://radar-scenes.com/dataset/structure/ . The inspected official HeRCULES reader preserves an unsigned byte, and does not establish a conversion to the model's dBsm input. This is an unresolved mapping, not proof that such documentation cannot exist elsewhere. Tensor dimensions and similar field names do not establish semantic compatibility. A guessed signed-byte reinterpretation would not be justified by the inspected source.

Source/model requirements also include output-target meaning and location. RadarScenes training labels are rear-axle vehicle-frame vx; this project's primary proxy is radar-pose speed magnitude. A valid cross-dataset experiment must justify that correspondence before a ranking. It is not fixed by taking an absolute value or silently changing references.

The source main scripts contain Python3.12SyntaxWarnings for invalid escapes in unused plotting strings. They were retained, not edited; source_audit_console.log and the independent audit record them. They did not prevent strict checkpoint loading or synthetic inference.

## Implemented admission check

input_contract.py checks required quantity, units, sensor coordinate frame and an evidenced semantic mapping. It accepts hypothetical properly documented compatible channels and rejects missing/unknown channels, RCS renamed as SNR, wrong frames and reference speed masquerading as radial measurements. The metadata must be reviewed against native preprocessing; a caller's asserted flag is not cryptographic authentication or physical calibration. Native units not specified in the inspected author code still require documented cross-sensor conversion evidence. This is benchmark admission logic, not a production sensor-authentication service.

All three current HeRCULES admissions fail for explicit reasons. No new neural target-data predictions, neural ranking, cross-dataset error numbers, additional point-cloud payload reads, fine-tuning or retraining were produced. The compatible classical comparator and existing source-stress/ablation results remain the valid comparisons.

## Evaluation scope and preserved scientific failures

EVALUATION_SCOPE.json explicitly permits recorded disagreement to named reference proxies and conditional interface/compact-replay outcomes. It does not certify true physical speed error, physical false-acceptance probability, source trust, clock bounds, deployment performance or human-centered effectiveness.

The original Library01last302query window still has301primary scores and P951.102297507859495m/s disagreement. Four original reference endpoint unknowns remain unknown. Optical is now a predictor and cannot be evaluation truth for itself; navigation-derived reference forms share ancestry. No scoring label, offset, model, threshold or previous result was changed. Existing development stop rules are retained; Library03/older protected payloads remain closed and the physical witness stays disabled.

## Verification and next usable path

731configured core tests and92focused tests pass,4core skips and one existing legacy calibration warning remain. The11new input-contract tests first failed because their module was missing, then passed. Independent verification checks immutable source/weight identities, three strict native architecture-weight pairs, three wrong-channel rejections, blocked semantic admissions and unchanged prior results. Author code syntax notes are separately retained.

Next usable neural comparison requires documented compatible input channels and target mapping, or a separately frozen, explicitly labelled neural adaptation using available Doppler/geometry. A new adaptation is not an exact pretrained-paper reproduction. Current source admission no longer depends on guessing whether weights exist: weights execute correctly; the target data contract is the actual blocker. Reference/frame/clock justification, specific novelty and justified final confirmation remain open. No manuscript, human review workflow, production change, installation, external message, commit or push.
"""
(R/"FINDINGS.md").write_text(text,encoding="utf-8")
decision={"admission_audit_complete":True,"native_architecture_weight_pairs_validated":3,"synthetic_tensor_checks_passed":3,"target_dataset_admissions":{x["id"]:x["target_admission"]["admitted"] for x in a["models"]},"target_neural_performance_comparison_executed":False,"blockers":["MSC3D native power missing","NavINST3D SNR missing","RadarScenes2D equivalent RCS dBsm encoding unverified","Native output target/frame/location correspondence requires justification"],"evaluation_claim_scope_recorded":True,"reference_gap_resolved":False,"previous_failures_retained":True,"overall_project_complete":False,"physical_witness_admitted":False,"Library03_payloads_accessed":False,"production_changed":False}
(R/"RESEARCH_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_NEURAL_ADMISSION.md"
if not backup.exists():shutil.copyfile(status,backup)
prefix="""Latest: pretrained neural source/weight validation and input-admission audit complete,2026-10-21.

Three unchanged native RadarNet/checkpoint pairs load strictly with weights_only=True on CPU and pass synthetic inference; independent wrong-channel tests reject all3. This validates model implementation/weight compatibility, not target-dataset performance.
Real HeRCULES neural scoring is blocked: MSC3D needs native power; NavINST3D needs SNR; neither channel is in current locked records. RadarScenes2D needs RCS(dBsm), whose equivalence to unsigned-byte HeRCULES encoding is unverified. Native output target/frame/location must also match. No guessed channels/conversions or neural ranking produced.
Input-admission tests enforce quantity/unit/frame/evidenced mapping. EVALUATION_SCOPE.json retains proxy-disagreement/conditional replay claims only; it does not relax old development gates or supply physical truth. Original failed302query segment/four unknown endpoints preserved.
731core/92focused tests pass,4skip, one legacy calibration warning. Native plotting-string SyntaxWarnings retained separately. Independent source/weight/tensor/admission/state audit passes.
Next usable path: documented native input/target compatibility or a separately frozen, explicitly labelled Doppler/geometry neural adaptation. Reference/clock/source justification, novelty and final-confirmation scope remain open. Overall project complete:NO. Physical witness disabled; Library03/older reserved payloads untouched.
Evidence: research/vehicle_neural_baseline/FINDINGS.md, model_admission_audit.json, EVALUATION_SCOPE.json, RESEARCH_DECISION.json and independent_verification.json.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"top_level_files":len(files),"native_models_validated":3,"valid_target_neural_comparison":False,"focused_passed":92,"core_passed":731,"core_skipped":4,"current_status_sha256":hashlib.sha256(status.read_bytes()).hexdigest()}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(decision,indent=2));print(json.dumps(completion,indent=2))
