# CO-ACT-CPS comparative validation

All 1,512 new confirmation trials are saved, independently checked and replayed from strict signed input packets. This adds 302,400 decisions to the preserved 540-trial predecessor. No predecessor trial was rerun and no model was trained on confirmation outcomes.

## Primary common-task comparison

| method | trials | contacts | selected MAE (m/s) | availability | mean progress (m) | fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| full | 144 | 12 | 0.550336 | 0.955000 | 165.241372 | 5146 |
| no_learning | 144 | 12 | 0.552800 | 0.955000 | 165.245991 | 5146 |
| no_refinement | 144 | 12 | 0.605929 | 0.955000 | 165.249839 | 5162 |
| no_fusion | 144 | 12 | 0.636260 | 0.955000 | 171.484764 | 2376 |
| no_brake | 144 | 12 | 0.550352 | 0.955000 | 165.235790 | 5146 |
| no_actuator_auth | 144 | 30 | 1.243411 | 0.955000 | 176.652701 | 5936 |
| s3e_brake | 144 | 12 | 0.075700 | 1.000000 | 165.924988 | 4320 |
| ssf_zocbf | 144 | 12 | 0.085236 | 1.000000 | 165.884183 | 4320 |

Each method has 144 matched scenario-seed trials: three new lead profiles, eight conditions, six frozen seeds. All methods use the same plant, exogenous source-noise streams, actuator envelope and query schedule. Their closed-loop trajectories and later observations can differ because their actions differ. Selected MAE includes only available estimates; compare it together with availability, contact and progress.

## Additional public-data-informed stresses

| method | trials | contacts | selected MAE (m/s) | availability | mean progress (m) | fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| full | 90 | 6 | 0.107936 | 0.960833 | 176.230396 | 977 |
| no_learning | 90 | 6 | 0.112049 | 0.960833 | 176.232975 | 979 |
| s3e_brake | 90 | 0 | 0.072923 | 0.758111 | 158.244434 | 4744 |
| ssf_zocbf | 90 | 6 | 0.079151 | 1.000000 | 177.111542 | 344 |

The four stress methods have 90 trials each. Conditions are public command-receipt age, public motion innovation, burst source loss, reduced braking and execution break. These deliberately challenging conditions are not interchangeable with bounded nominal assumptions. Per-case CSV rows retain failures and missing estimates; unequal method populations are not combined.

## Matched component and method effects

| comparison | metric | difference | six-seed interval |
| --- | --- | --- | --- |
| full - no_learning | mae_mps | -0.002488 | [-0.002835, -0.002142] |
| full - no_learning | collision_trial | 0.000000 | [0.0, 0.0] |
| full - no_learning | progress_m | -0.004619 | [-0.006655, -0.002465] |
| no_learning - no_refinement | mae_mps | -0.052703 | [-0.053417, -0.052041] |
| no_learning - no_refinement | collision_trial | 0.000000 | [0.0, 0.0] |
| no_learning - no_refinement | progress_m | -0.003848 | [-0.007926, -0.00044] |
| full - no_fusion | mae_mps | -0.082901 | [-0.088174, -0.078324] |
| full - no_fusion | collision_trial | 0.000000 | [0.0, 0.0] |
| full - no_fusion | progress_m | -6.243392 | [-6.346155, -6.152602] |
| full - no_brake | mae_mps | -0.000016 | [-2.8e-05, -6e-06] |
| full - no_brake | collision_trial | 0.000000 | [0.0, 0.0] |
| full - no_brake | progress_m | 0.005581 | [0.00499, 0.006168] |
| full - no_actuator_auth | mae_mps | -0.665213 | [-0.673379, -0.657325] |
| full - no_actuator_auth | collision_trial | -0.125000 | [-0.125, -0.125] |
| full - no_actuator_auth | progress_m | -11.411329 | [-11.584042, -11.236449] |
| full - s3e_brake | mae_mps | 0.457054 | [0.454198, 0.459827] |
| full - s3e_brake | collision_trial | 0.000000 | [0.0, 0.0] |
| full - s3e_brake | progress_m | -0.683616 | [-0.703274, -0.66394] |
| full - ssf_zocbf | mae_mps | 0.447517 | [0.444494, 0.450511] |
| full - ssf_zocbf | collision_trial | 0.000000 | [0.0, 0.0] |
| full - ssf_zocbf | progress_m | -0.642812 | [-0.660723, -0.623554] |

Differences are left minus right on matched profile/condition/seed worlds. Lower contact rate and error are favorable; greater progress and availability are favorable. A favorable error difference alone is not an overall win. Intervals resample six seed clusters, retaining their cases and profiles. They are exploratory descriptive uncertainty, not independent-query evidence or proof of universal superiority.

Refinement compares no_learning against no_refinement: successful refined-interval selection is replaced by a budget-projected midpoint of the original scene interval. The same fallback algorithm is retained when refinement is unavailable or conflicting. This is an eligibility-conditioned interval ablation, not deletion of the whole module. Eligible queries: 25886. Learning compares full against no_learning; fusion, braking and actuation authentication compare full against their named ablations.

## Saved-artifact verification

    {
      "trials": 1512,
      "decisions": 302400,
      "vehicle_transitions": 6044976,
      "forest_predictions": 121479,
      "operator_cards": 302400,
      "baseline_updates": 93600,
      "baseline_qualified_sets": 88639,
      "baseline_qualified_set_violations": 0,
      "point_certified_qualified": 264717,
      "point_certified_qualified_errors": 0,
      "admitted_contexts": 265364,
      "actual_context_violations": 12049,
      "qualified_followups": 245809,
      "qualified_followup_gap_violations": 0,
      "changed_actuator_frames": 8640,
      "changed_frames_accepted": 1080,
      "refinement_eligible_queries": 25886,
      "public_profile_rows": 5100
    }

The independent verifier checks every numerical transition, protocol identity, complete checkpoint hashes, prescribed source attacks, public heldout split, command-log MAC and accepted-command coverage, interval/filter arithmetic, manual forest outputs and aggregate metrics. A separate fresh-state replay compares every recorded policy decision, issued/accepted command and operator card to the strict packet-only implementation. Replay uses the same algorithm and is separate from independent mathematics.

Conditional qualification uses actual sensor-set containment and stated execution bounds. All context violations, abstentions and contacts remain in overall results; qualified subsets do not erase failures outside those assumptions. Zero qualified violations is a finite benchmark result, not a universal safety theorem.

## Literature and novelty boundary

S3E and secure-safety-filter baselines are transparent common-task adaptations documented in BASELINE_CONTRACT.md. S3E retains disconnected agreement sets and temporal accepted-command prediction; SSF uses current plausible sets and a 10ms finite-grid, zero-order-held headway filter. These are not complete native reproductions of the papers' original systems. RAIL, GUARDIAN and software-defined-vehicle zero trust are conceptual neighbors with different tasks and assumptions. No ranking against their native systems is claimed.

The contribution to assess is the combined signed-source/context-to-command pipeline, explicit assumption-aware refinement, bounded learned selection, independently audited closed-loop attacks and component evidence. Novelty depends on measured component effects and differences from cited methods. No single improved metric or implementation count establishes broad originality.

## Data and simulation scope

Public provenance: TU/e F1TENTH, DOI 10.5281/zenodo.12536536. Easy defines calibration summaries; medium and hard are heldout stress sequences. Command receipt age is a staleness proxy, not measured network delay. Motion second difference includes motion/reference artifacts, not calibrated sensor noise. Large tails and missing values are retained. No protected HeRCULES Library03, Urban02 NIGHT, RAV4 or vehicle_final_evaluation data were opened.

New trials use a 10ms numerical longitudinal plant. Full 20s, all-profile parity against SUMO 1.27.1 matched speed and position exactly under the tested actuator contract. These are not 1,512 native SUMO socket runs. A contact is trajectory gap <=0, not a SUMO warning. Timing is observed wall time under concurrent CPU execution, not a hard real-time guarantee.

The operator viewer replays recorded observation-derived cards with reason, source authentication, control margin and accepted command. It has no evaluator truth input and no human-subject evaluation. The study addresses trustworthy AI, cybersecurity, CPS and explanatory operator support within simulation. It does not establish human effectiveness or cover every call keyword. Journal acceptance and full special-issue compliance require editorial assessment.
