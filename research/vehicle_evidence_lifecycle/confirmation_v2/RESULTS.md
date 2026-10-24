# Joint context-history and execution-authority confirmation

1080 new frozen confirmation trials:6fresh seeds,3profiles,12conditions,5methods,216000decisions. Earlier540/1512studies and55/60development studies remain separate. No confirmation result was used to select this policy.

| Method | Trials | Contacts | Selected MAE m/s | Availability | Mean progress m |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 216 | 6 | 0.277911 | 1.000000 | 135.654017 |
| robust_current | 216 | 42 | 0.311656 | 1.000000 | 224.061824 |
| nominal_authority | 216 | 12 | 1.080535 | 1.000000 | 154.947245 |
| no_learning | 216 | 6 | 0.361833 | 1.000000 | 135.654944 |
| always_brake | 216 | 0 | 0.091735 | 1.000000 | 29.032045 |

Interpret selected error together with availability, uncertainty, progress and fallback. A numeric speed may be reported while its0.5m/s point budget is uncertified; robust control uses the whole retained speed set. These are matched closed-loop scenarios with common exogenous streams, not identical-state estimator queries.

## Contract and limits

Initial gap/lead honesty, bounded uncompromised ego-source subset and execution envelope are premises. Source authentication alone does not establish them. The joint method never narrows safety context from later unqualified source agreement. It conservatively propagates context, uses E=2.5m/s2 worst execution error, and binds each100ms command to an independently checked certificate. Every10ms plant command goes through the actuator watchdog. Missing frames trigger backup. Exact-time delivery and periodic callbacks are modeled; broken controller/actor cadence requires a new qualified session. No network latency tolerance or hard real-time CPU guarantee is claimed.

False bootstrap, authority_exceed and public_motion_innovation are predeclared negative controls. Their contacts, errors and failed state/certificate assumptions remain in the overall and per-condition tables. nominal_authority deliberately uses E=1 and is an assumption ablation. Zero premise-qualified violations is a conditional finite-model result, not immunity to compromised bootstrap, unbounded attacks or a continuous-time vehicle theorem.

## Comparators and ablations

robust_current uses a scalar S3E/current-context estimate with the same frozen learned nominal proposal, E=2.5stopping filter and certificate gate. This is an enhanced common-task adaptation, not a complete native S3E reproduction. no_learning changes bounded point selection while retaining the same history/authority mechanism. always_brake exposes trivial stopping. The preserved1512-trial study separately compares the secure-safety-filter adaptation. Performance must not be presented as native full-framework superiority.

## Independent audit

{
  "trials": 1080,
  "decisions": 216000,
  "vehicle_transitions": 4317840,
  "packet_macs": 1080000,
  "forest_predictions": 133975,
  "certificates": 216000,
  "verified_certificates": 206435,
  "qualified_certified_queries": 102671,
  "qualified_gap_failures": 0,
  "state_containment_failures": 29681,
  "actual_state_certificate_counterexamples": 6789,
  "point_certified_errors": 0,
  "changed_frames_accepted": 0,
  "qualified_containment_failures": 0,
  "qualified_certificate_counterexamples": 0
}

The audit covers all grid/checkpoints, source/command MACs and attack generation, observer and history updates, raw physical transitions, maximum command/hold boundaries, full-stop margins, certificate/expiry/observation binding, executed watchdog commands, metrics and recorded explanation cards. It fails on any premise-qualified containment or robust-certificate counterexample, including cases whose realized trajectory happens not to collide. Manual frozen forest predictions are checked; the upstream frozen scene adapter retains its separate component evidence. Six-seed intervals are descriptive and not query-independent inference.

## Contribution assessment

Secure estimation, safety filters, provenance, self-triggered control and certificate checking exist in prior work. The bounded candidate contribution is their specific joint authorization contract: unqualified context cannot refresh history, actuator error reduces recoverable authority, command lifetime is derived and bound at the gate, and explanation/audit retains the exact assumptions. PROPERTIES.md gives the discrete derivation and information/authority limits. NOVELTY_SCOPE.md records the primary-source comparison. This is evidence for reviewers to assess; no first-ever originality or journal acceptance certification is made.

The operator viewer replays observation-derived cards and discloses the conditional contract. Technical explanation fidelity is tested; human effectiveness is not evaluated. The simulation scope excludes physical hardware validation by user choice. No manuscript has been written.
