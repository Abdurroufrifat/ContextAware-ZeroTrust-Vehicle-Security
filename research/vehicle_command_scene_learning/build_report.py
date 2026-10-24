from cs_common import *
install_guard()
import gzip,csv,math,statistics,itertools
from fractions import Fraction as F
cfg=json.loads((R/"PROTOCOL.json").read_text());result=json.loads((R/"results.json").read_text())
assert json.loads((R/"verification.json").read_text())["passed"]
app={x["trial_id"]:x for x in json.loads((R/"scene_appraisal.json").read_text())};trials=[]
for t in result["trials"]:
 with gzip.open(checked(R/t["trace"]),"rt",encoding="utf-8") as f:rows=[json.loads(line) for line in f]
 errors=[x["selected_disagreement_mps"] for x in rows if x["selected_disagreement_mps"] is not None]
 active=[x["selected_disagreement_mps"] for x in rows if x["active"] and x["selected_disagreement_mps"] is not None]
 widths=[];oldwidths=[]
 for row in rows:
  d=row["decision"]
  if d.get("refined_budget_satisfied"):
   widths.append(float(F(d["refined_high_n"],d["refined_high_d"])-F(d["refined_low_n"],d["refined_low_d"]))/1000)
   oldwidths.append(float(F(d["witness_high_n"],d["witness_high_d"])-F(d["witness_low_n"],d["witness_low_d"]))/1000)
   assert widths[-1]<=oldwidths[-1]+1e-12
 a=app[t["trial_id"]]
 trials.append({**t,"selected_mae_mps":math.fsum(errors)/len(errors),"active_selected":len(active),"active_gt05":sum(x>.5 for x in active),
  "valid_certificate_selected":a["selected_with_valid_joint_certificate"],"valid_certificate_gt05":a["joint_certificate_errors_gt05"],
  "scene_valid_selected":a["selected_with_valid_scene"],"scene_valid_gt05":a["selected_error_gt05_with_valid_scene"],
  "refined_points":len(widths),"mean_refined_width_mps":statistics.mean(widths) if widths else None,
  "mean_original_width_at_refined_points_mps":statistics.mean(oldwidths) if oldwidths else None})
write("enriched_trials.json",trials)
def aggregate(group):
 n=sum(x["selected"] for x in group);q=sum(x["queries"] for x in group)
 return dict(trials=len(group),queries=q,selected=n,availability=n/q,
  selected_mae_mps=sum(x["selected_mae_mps"]*x["selected"] for x in group)/n,
  errors_gt05=sum(x["selected_error_gt05"] for x in group),errors_gt2=sum(x["selected_error_gt2"] for x in group),
  valid_certificate_selected=sum(x["valid_certificate_selected"] for x in group),valid_certificate_gt05=sum(x["valid_certificate_gt05"] for x in group),
  scene_valid_selected=sum(x["scene_valid_selected"] for x in group),scene_valid_gt05=sum(x["scene_valid_gt05"] for x in group),
  mean_progress_m=statistics.mean(x["progress_m"] for x in group),collision_events=sum(x["collision_events"] for x in group),
  max_trial_p95_decision_ms=max(x["decision_wall_p95_s"] for x in group)*1000)
aggregates=[]
for profile,case,policy in itertools.product(cfg["profiles"],cfg["cases"],cfg["policies"]):
 group=[x for x in trials if (x["profile"],x["case"],x["policy"])==(profile,case,policy)];assert len(group)==3
 aggregates.append(dict(profile=profile,case=case,policy=policy,**aggregate(group)))
write("aggregate_results.json",aggregates)
for name,rows in (("all_trials.csv",trials),("aggregate_results.csv",aggregates)):
 with checked(R/name).open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs=[]
for profile,case,seed in itertools.product(cfg["profiles"],cfg["cases"],cfg["seeds"]):
 group={x["policy"]:x for x in trials if (x["profile"],x["case"],x["seed"])==(profile,case,seed)}
 old,mid,learned=[group[k] for k in cfg["policies"]]
 pairs.append(dict(profile=profile,case=case,seed=seed,
  learned_minus_original_mae=learned["selected_mae_mps"]-old["selected_mae_mps"],
  learned_minus_midpoint_mae=learned["selected_mae_mps"]-mid["selected_mae_mps"],
  learned_minus_original_selected=learned["selected"]-old["selected"],
  learned_minus_midpoint_selected=learned["selected"]-mid["selected"],
  learned_minus_original_progress=learned["progress_m"]-old["progress_m"],
  learned_minus_midpoint_progress=learned["progress_m"]-mid["progress_m"]))
write("paired_results.json",pairs)
sensitivity=[]
for seed in cfg["seeds"]:
 group=[x for x in pairs if x["seed"]==seed]
 sensitivity.append(dict(seed=seed,paired_scenarios=len(group),
  mean_learned_minus_original_mae=statistics.mean(x["learned_minus_original_mae"] for x in group),
  mean_learned_minus_midpoint_mae=statistics.mean(x["learned_minus_midpoint_mae"] for x in group)))
write("seed_sensitivity.json",dict(blocks=sensitivity,independent_rng_seeds=3,shared_noise_across_profiles_cases=True,
 inference="Finite-grid descriptive paired effects, no independent-cell bootstrap/population confidence claim."))
totals={p:aggregate([x for x in trials if x["policy"]==p]) for p in cfg["policies"]}
nominal={p:aggregate([x for x in trials if x["policy"]==p and x["case"]=="nominal"]) for p in cfg["policies"]}
badscene={p:aggregate([x for x in trials if x["policy"]==p and x["case"]=="scene_common_compromise"]) for p in cfg["policies"]}
assessment=dict(totals=totals,nominal=nominal,common_scene_compromise=badscene,
 learned_better_original_nominal=nominal["refined_learned"]["selected_mae_mps"]<nominal["kinematic"]["selected_mae_mps"],
 learned_better_midpoint_nominal=nominal["refined_learned"]["selected_mae_mps"]<nominal["refined_midpoint"]["selected_mae_mps"],
 learned_better_original_all_cases=totals["refined_learned"]["selected_mae_mps"]<totals["kinematic"]["selected_mae_mps"],
 learned_better_midpoint_all_cases=totals["refined_learned"]["selected_mae_mps"]<totals["refined_midpoint"]["selected_mae_mps"],
 paired_cells=81,learning_better_original_cells=sum(x["learned_minus_original_mae"]<0 for x in pairs),
 learning_better_midpoint_cells=sum(x["learned_minus_midpoint_mae"]<0 for x in pairs),
 conditional_interval_proof="Discrete10ms right-endpoint containment, joint scene/execution premises, exact rational arithmetic; no physical qualification.",
 global_first_ever_claimed=False,native_recent_framework_superiority_claimed=False,physical_vehicle_control_validated=False,
 complete_special_issue_compliance_certified=False,production_promotion=False)
write("contribution_assessment.json",assessment)
selection=json.loads((R/"model_selection.json").read_text())
lines=["# Command-refined scene learning results","",
 "This frozen computational study tests a new development hypothesis following the earlier unsuccessful learned-estimator study. Earlier results remain unchanged.",
 "",f"Selected model: {selection['selected']}; {selection['training_rows']} training and {selection['validation_rows']} validation rows. Training excludes declared broken scene/command-execution premises and uses only relative physical features.",
 "Fresh test grid: three new driving profiles, three new seeds, nine cases, three policies; 243 closed-loop trials and48,600 decisions. No fresh test result was used for model selection.",
 "","| Policy | Nominal selected / queries | Nominal MAE m/s | All-case selected / queries | All-case MAE m/s | Errors >0.5 with valid certificate |",
 "|---|---:|---:|---:|---:|---:|"]
for p in cfg["policies"]:
 a=totals[p];n=nominal[p]
 lines.append(f"| {p} | {n['selected']} / {n['queries']} | {n['selected_mae_mps']:.6f} | {a['selected']} / {a['queries']} | {a['selected_mae_mps']:.6f} | {a['valid_certificate_gt05']} / {a['valid_certificate_selected']} |")
lines+=["",f"Learned estimate has lower selected MAE than the original policy in {assessment['learning_better_original_cells']}/81 paired cells and the refined midpoint in {assessment['learning_better_midpoint_cells']}/81.",
 "All-case averages include deliberately broken trust/execution assumptions and policy abstentions. Availability, case-wise outcomes and premise qualification must be considered alongside error.",
 "","## Formal contribution and limits","",
 "MODEL_CONTRACT.md derives the discrete speed identity, lead endpoint envelopes, signed-command correction and integer point projection. The refined interval is contained in the old interval under additional execution assumptions; its admissible point set is larger. This does not prove collision avoidance, stability, physical sensor truth or a general nonlinear-vehicle theorem.",
 "The old scene certificate is kept inside scene_decision. Modified estimates use a distinct joint scene-and-command certificate. The independent verifier derives lead bounds with forward/backward constraints and manually replays forest trees; it imports no production estimator or controller.",
 "","## Relation to the EAAI call","",
 "The official publisher scope requires novel AI engineering work, public-data validation and reproducibility. No mandatory physical-car test or new mathematical principle was verified. The full special-issue page remains403, so complete call compliance cannot be certified.",
 "Existing public HeRCULES/Boreas recorded-data evidence remains separate. It does not directly validate the scene/command model because the required independently qualified channels and bounds are absent. A physical-car/control-validation claim remains unsupported.",
 "Known prior work already covers residual learning, context-aware zero trust, set estimation and safety filters. prior_method_matrix.json describes specific differences without a broad first-ever or native-framework superiority claim.",
 "The all-source-forgery indistinguishability argument states why software alone cannot prove truthful physical state without an uncompromised information source. Common-scene-compromise, forged-log and execution-error cases remain reported.",
 "No human-effectiveness claim, production deployment or guaranteed journal acceptance is made.",
 "","## Reproduce","",
 "Run research/vehicle_command_scene_learning/run_all.py with the existing project Python interpreter. It verifies the frozen experiment and prior evidence; it does not retrain or rerun the driving grid.",
 "Raw outputs: simulator_outputs/. Per-case aggregates: aggregate_results.csv. Paired effects: paired_results.json. Premise checks: scene_appraisal.json. Row exclusions: training_row_audit.json. Source/model hashes: experiment_lock.json and model_lock.json.",
 "","The strict signed-input application is CommandSceneSession in command_scene_session.py. It accepts the existing four-field signed record, rejects reference/operation fields, and checks selected-model/source integrity. replay_application.py checks54 profile/case/mode trajectories at seed307 (10,800 decisions); these are integration replays, not extra driving evidence."]
checked(R/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
checked(P/"COMMAND_SCENE_RESULTS.md").write_text("\n".join(lines[:14])+"\n\nFull proof, scope and verification: research/vehicle_command_scene_learning/README.md.\n",encoding="utf-8")
print("REPORT_COMPLETE",json.dumps(assessment,allow_nan=False),flush=True)

controller_scope_note='\nController scope: the IDM-form longitudinal controller receives uncorrupted simulator lead speed and gap, as in the frozen predecessor experiment. Attacks alter admission input streams. In common-scene-compromise, the signed scene stream presented to the estimator is corrupted; separate controller context remains ideal. Zero collisions in this grid are not evidence of whole-perception-stack resilience.\n'
for target in (R/"README.md",P/"COMMAND_SCENE_RESULTS.md"):
 target.write_text(target.read_text()+controller_scope_note,encoding="utf-8")
