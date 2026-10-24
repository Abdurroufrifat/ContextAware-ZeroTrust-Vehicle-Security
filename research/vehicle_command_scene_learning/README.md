# Command-refined scene learning results

This frozen computational study tests a new development hypothesis following the earlier unsuccessful learned-estimator study. Earlier results remain unchanged.

Selected model: forest; 6268 training and 3134 validation rows. Training excludes declared broken scene/command-execution premises and uses only relative physical features.
Fresh test grid: three new driving profiles, three new seeds, nine cases, three policies; 243 closed-loop trials and48,600 decisions. No fresh test result was used for model selection.

| Policy | Nominal selected / queries | Nominal MAE m/s | All-case selected / queries | All-case MAE m/s | Errors >0.5 with valid certificate |
|---|---:|---:|---:|---:|---:|
| kinematic | 1791 / 1800 | 0.064048 | 15300 / 16200 | 0.117939 | 0 / 14211 |
| refined_midpoint | 1791 / 1800 | 0.020789 | 15300 / 16200 | 0.079039 | 0 / 13144 |
| refined_learned | 1791 / 1800 | 0.018703 | 15300 / 16200 | 0.076661 | 0 / 13145 |

Learned estimate has lower selected MAE than the original policy in 81/81 paired cells and the refined midpoint in 79/81.
All-case averages include deliberately broken trust/execution assumptions and policy abstentions. Availability, case-wise outcomes and premise qualification must be considered alongside error.

## Formal contribution and limits

MODEL_CONTRACT.md derives the discrete speed identity, lead endpoint envelopes, signed-command correction and integer point projection. The refined interval is contained in the old interval under additional execution assumptions; its admissible point set is larger. This does not prove collision avoidance, stability, physical sensor truth or a general nonlinear-vehicle theorem.
The old scene certificate is kept inside scene_decision. Modified estimates use a distinct joint scene-and-command certificate. The independent verifier derives lead bounds with forward/backward constraints and manually replays forest trees; it imports no production estimator or controller.

## Relation to the EAAI call

The official publisher scope requires novel AI engineering work, public-data validation and reproducibility. No mandatory physical-car test or new mathematical principle was verified. The full special-issue page remains403, so complete call compliance cannot be certified.
Existing public HeRCULES/Boreas recorded-data evidence remains separate. It does not directly validate the scene/command model because the required independently qualified channels and bounds are absent. A physical-car/control-validation claim remains unsupported.
Known prior work already covers residual learning, context-aware zero trust, set estimation and safety filters. prior_method_matrix.json describes specific differences without a broad first-ever or native-framework superiority claim.
The all-source-forgery indistinguishability argument states why software alone cannot prove truthful physical state without an uncompromised information source. Common-scene-compromise, forged-log and execution-error cases remain reported.
No human-effectiveness claim, production deployment or guaranteed journal acceptance is made.

## Reproduce

Run research/vehicle_command_scene_learning/run_all.py with the existing project Python interpreter. It verifies the frozen experiment and prior evidence; it does not retrain or rerun the driving grid.
Raw outputs: simulator_outputs/. Per-case aggregates: aggregate_results.csv. Paired effects: paired_results.json. Premise checks: scene_appraisal.json. Row exclusions: training_row_audit.json. Source/model hashes: experiment_lock.json and model_lock.json.

The strict signed-input application is CommandSceneSession in command_scene_session.py. It accepts the existing four-field signed record, rejects reference/operation fields, and checks selected-model/source integrity. replay_application.py checks54 profile/case/mode trajectories at seed307 (10,800 decisions); these are integration replays, not extra driving evidence.

Controller scope: the IDM-form longitudinal controller receives uncorrupted simulator lead speed and gap, as in the frozen predecessor experiment. Attacks alter admission input streams. In common-scene-compromise, the signed scene stream presented to the estimator is corrupted; separate controller context remains ideal. Zero collisions in this grid are not evidence of whole-perception-stack resilience.
