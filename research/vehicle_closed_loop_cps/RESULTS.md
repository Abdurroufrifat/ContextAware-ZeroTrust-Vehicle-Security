# Closed-loop CPS simulation results

The user-approved completion scope is numerical cyber-physical simulation. The physical-hardware objective is excluded. Earlier frozen experiments remain unchanged.

Five policies were evaluated in three driving profiles, three new seeds and twelve conditions: 540 confirmation trials, 108,000 policy decisions. The frozen forest was not retrained. Sensor packets, including attacked lead/gap observations, determine every controller input; the controller receives no evaluator truth.

| Policy | Colliding trials / all trials | Nominal progress (m) | Nominal selected / queries | Nominal speed MAE (m/s) | All-case fallback |
|---|---:|---:|---:|---:|---:|
| raw | 39 / 108 | 207.821 | 1800 / 1800 | 0.050781 | 2.50% |
| sensor_auth | 18 / 108 | 207.756 | 1791 / 1800 | 0.064583 | 6.92% |
| learned | 18 / 108 | 207.800 | 1791 / 1800 | 0.022586 | 6.96% |
| proposed_midpoint | 9 / 108 | 207.798 | 1791 / 1800 | 0.025390 | 16.74% |
| proposed_learned | 9 / 108 | 207.800 | 1791 / 1800 | 0.022492 | 16.74% |

Proposed learned versus proposed midpoint speed MAE: nominal 0.022492 versus 0.025390 m/s; all tested cases 0.464527 versus 0.467017 m/s. Lower is better. The learned estimator is not credited with gains due to actuator authentication or control filtering.

Nominal progress retained versus sensor-authenticated control: 100.02%. Difference in all-case colliding trials versus that baseline: 9.

| Condition | raw | sensor_auth | learned | proposed_midpoint | proposed_learned |
|---|---:|---:|---:|---:|---:|
| nominal | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| speed_bias | 3 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| candidate_bad_tag | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| scene_disinformation | 6 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| scene_replay | 3 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| scene_dropout | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| radar_disinformation | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| common_view_disinformation | 6 / 9 | 6 / 9 | 6 / 9 | 6 / 9 | 6 / 9 |
| actuator_tamper | 9 / 9 | 9 / 9 | 9 / 9 | 0 / 9 | 0 / 9 |
| untrusted_advisory | 9 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| execution_break | 3 / 9 | 3 / 9 | 3 / 9 | 3 / 9 | 3 / 9 |
| context_change | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |

Across all proposed-policy admitted contexts, 1129 / 37050 failed actual state containment, including the tested premise-breaking conditions. These failures remain in the report and all-case collision outcomes.


The all-case table includes common-source semantic corruption and actual execution outside the declared envelope. Collision results and availability tradeoffs are retained. Different policies produce different feedback trajectories; matched scenarios are not identical-state prediction tests.

Independent verification checked 108000 decisions, 216000 queried vehicle states and 2158920 vehicle transitions. It manually evaluated 38261 frozen forest outputs and checked every operator card.

Qualified selected-speed errors over 0.5 m/s at full logged precision: 0 / 71208. Qualified context containment violations: 0 / 34924. Qualified following-interval gap violations: 0 / 34620. These denominators differ and must not be pooled. Independent FCD comparisons allow 2e-6 m/s speed and 3e-6 m gap rounding tolerances; the strict speed-error count uses the full-precision logged error.

Actual changed actuator frames accepted by proposed policies: 0 / 1080. Untrusted advisory instructions accepted by proposed policies: 0 / 1080. A no-op tamper attempt is not counted as a changed frame.

The human-centered feature is an evidence-card interface showing observed context, source status, cause, action and limitations. Technical fidelity was checked against software decisions. No measured human trust, usability, persuasion resistance or general phishing result is claimed.

MODEL_CONTRACT.md specifies sensor, clock, motion and execution assumptions. EAAI_ALIGNMENT.md maps the supplied topics and distinguishes established methods from this project's specific integration. Three RNG seeds support descriptive sensitivity analysis; no population confidence intervals are claimed.

Open operator_view.html for the explanation viewer and closed_loop_results.png for the result figure. aggregate_results.csv, paired_results.json and seed_sensitivity.json contain the detailed comparisons.

The software and computational evidence are packaged for manuscript preparation once final verification passes. Editorial acceptance and first-ever originality are not certification outputs of this project.
