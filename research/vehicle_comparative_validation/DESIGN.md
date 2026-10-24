# Comparative validation design
Date: 24 October 2026. User authorized all three focused research extensions; prior requests delegate routine design choices and ask for continuous execution with few credits.
Goal: strengthen comparative evidence without changing the frozen 540-trial study or retraining the selected forest.
Isolation: new research/vehicle_comparative_validation folder on existing v8-verifier-utility branch. Existing unrelated dirty files and all frozen inputs remain untouched; no commits, branch switch, public release or manuscript.

## Selected approach and alternatives
Implement a new common longitudinal numerical plant matching the existing 10ms right-endpoint SUMO contract, with parity checks against fresh SUMO runs before confirmation. This avoids thousands of socket/process launches while preserving physical feedback. Alternatives: rerun the original study (does not isolate new contributions; rejected); port full RAIL/GUARDIAN platforms (different learned perception, dynamics and threat models, not a fair drop-in baseline; rejected).
Comparators are explicit common-model method adaptations, not native full-framework reproductions or inherited general theorems.

## Methods, shared information and attribution
Eight methods: full, no_learning, no_refinement, no_fusion, no_brake, no_actuator_auth, s3e_brake, ssf_zocbf.
Full uses the frozen refined forest, authenticated scene/radar fusion, braking-reserve projection and actuator authentication.
no_learning uses the refined midpoint. no_refinement uses the original scene midpoint; compare it with no_learning to isolate refinement without a mismatched learned feature representation.
Other ablations remove exactly context fusion, braking filter or actuator-frame authentication from full.
All primary comparators receive the same six-domain signed packet schema and public bounds; no evaluator truth enters policy APIs. Candidate, delayed anchor, and scene-pair kinematic witness form the three scalar ego-speed observation domains. An observation unavailable or rejected by shared admission contributes no interval.
S3E: exact one-dimensional interval-union specialization of time update, per-sensor intersection, p-q=2 agreement and union, q=1 among the three scalar domains. Retain disconnected sets. Actual clipping at zero is a declared monotone plant adaptation. Use the same scene/radar context and braking-reserve controller/actuator gate as full to isolate estimator effects. No scene+radar derived ego channels are counted as independent.
SSF: measurement-consistent union over two-of-three current ego observation intervals; interval-valued relaxed reconstruction specialization. Apply a discrete zero-order headway CBF filter, least change to the nominal integer command, evaluated over every plausible-state interval and common context bounds. Headway1s, alpha0.2 per100ms. Explicit finite-grid adaptation; no unqualified continuous-time proof. Compare ssf_zocbf with s3e_brake as a combined estimator/filter difference, not an isolated causal effect.
Authentication, packet age handling, actuator permissions and plant disturbance authority are common except the deliberate no_actuator_auth ablation. Resetting/recovery and initial information must be documented; empty plausible sets cause braking, not fabricated recovery from truth. Shared-source and q>1 cases are reported separately from in-premise cases.

## New frozen experiment
Primary grid: 3 new profiles x6 new seeds x8 cases x8 methods =1152 new closed-loop trials.
Stress grid: 3 profiles x6 seeds x5 stress cases x4 methods(full,no_learning,s3e_brake,ssf_zocbf)=360 trials.
Total1512 new numerical closed-loop trials,302400 decisions. Old540 not rerun.
New seeds: 1103,1229,1361,1499,1621,1753. Development-only seeds: 2903,3011.
Profiles: urban_cutback lead14 ego13 gap29 and lead acceleration[1,-2,-4,2,0]; highway_recovery lead20 ego19 gap43 [0,-3,1,-4,2]; stop_release lead8 ego7 gap23 [-2,2,-3,2,0].
All profiles20s, five4s segments. Ego initial position60m, length5m. Control100ms, plant10ms. Attack5s<=t<11s.
Primary: nominal,candidate_drift,scene_disinformation,radar_disinformation,scene_dropout,actuator_tamper,adaptive_candidate,shared_disinformation.
Stress: public_anchor_age,public_motion_innovation,burst_loss,lower_braking,execution_break.
Shared sensor-noise streams and source mutation algorithms use matched seed schedules; actual values differ when feedback trajectories differ. No tuning or model selection on confirmation outcomes.
RNG process workers3, OMP/BLAS1. Resumption verifies frozen input and completed-trial hashes; incomplete attempt directories preserved, only absent/unfinished IDs rerun. No silent skipping of errors.

## Public-data provenance
Use existing verified TU/e F1TENTH CSVs (DOI10.5281/zenodo.12536536); bind raw/extraction manifests and source CSV hashes.
Easy recording defines receipt-time command-age and VICON-motion innovation distributions before new tests. Medium/hard recordings supply held-out availability and innovation sequences.
Command age is a causal availability/staleness proxy, not network latency or clock calibration. VICON high-frequency motion innovation includes physical motion/reference noise; transferred perturbations are stress inputs, not a sensor-accuracy estimate or guaranteed hard bound. Original small scene bounds remain model assumptions and out-of-bound stress rows must not receive valid certificates.
Publish exact extraction equations, thresholds, source rows, quantization and transforms. No fitting against confirmation performance. No protected Library03, NIGHT, RAV4 or vehicle_final_evaluation payload access.

## Completion evidence
TDD for interval unions, finite-grid CBF, data timing/innovation extraction, component routing, checkpoint resumption and aggregation.
SUMO parity over all new profiles with seeded command sequences and zero-speed contact.
Freeze sources, forest dependencies, public profiles and protocol before confirmation.
Retain every new trace/trajectory and all failures. Independently verify numerical dynamics, observation-only routing, baseline set operations/filter constraints, frozen forest outputs, summaries and paired/cluster descriptive reporting.
Report matched component effects, baseline assumptions, availability/progress/safety/error, all-case and qualified counts. Any confidence intervals resample seed clusters only and are labelled exploratory with six clusters.
One fresh source reviewer before freeze; critical findings fixed with regression evidence. New viewer/report entry separate from frozen old files.

Refinement-ablation clarification: no_refinement uses the broad scene interval midpoint only when the frozen refined-midpoint arm reports successful refinement. All unavailable/conflict fallback selections remain identical to no_learning. This isolates successful interval refinement, not every behavior of deleting the whole module; report its eligible-row count. It computes refinement eligibility but does not use refined bounds for the ablated point. Full versus no_learning remains the untouched frozen learning-selector comparison.
The ZO-CBF adapter checks current/every10ms headway h>=0, terminal contraction and gap, including the zero-speed crossing; no intersample continuous-time theorem transfer.
