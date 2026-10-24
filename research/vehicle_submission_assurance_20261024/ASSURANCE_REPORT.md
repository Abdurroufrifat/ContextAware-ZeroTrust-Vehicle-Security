# CO-ACT-CPS submission assurance supplement

Secondary analysis of the frozen 1080-trial release. No experiments repeated, policy changed or manuscript written.

| Method | Contacts /216 | Mean progress m | Certified point coverage | Observed loops >100ms |
|---|---:|---:|---:|---:|
| joint | 6 | 135.654 | 95.56% | 5.77% |
| robust_current | 42 | 224.062 | 95.25% | 6.45% |
| nominal_authority | 12 | 154.947 | 93.44% | 6.11% |
| no_learning | 6 | 135.655 | 95.56% | 5.91% |
| always_brake | 0 | 29.032 | 99.42% | 5.14% |

A numeric estimate is not necessarily within its stated point budget. Command authorization uses the retained state set and is separate from point certification. The assurance viewer makes both statuses explicit and always discloses source truth and bootstrap limitations.

The measured contact/progress tradeoff is not universal dominance. Always-brake has no contacts and little motion. Current-context control moves farther and has more contacts. Learning improves speed error but does not reduce the contact count relative to no-learning. These are closed-loop policy comparisons with common exogenous scenarios, not identical-state estimation experiments.

Pareto methods for the two aggregate empirical objectives (fewer contacts, greater mean progress): robust_current, nominal_authority, no_learning, always_brake. This is descriptive over the fixed evaluated policies; it is not an equal-progress comparison or an optimized Pareto frontier. No ratio of contacts to distance is used because it could reward unsafe progress.

The exact two-objective Pareto list omits speed-error quality. Joint and no-learning contact counts are equal; mean progress differs by only0.000927m. This sub-millimetre aggregate difference is not evidence of practical control superiority. The measured learning benefit is speed-estimation error.

## Timing evidence

Recorded controller processing, command construction and actuator checking on a shared three-worker Windows run. Excludes subsequent trace writing and physics advancement. Offline simulated timestamps did not advance by wall time; no end-to-end deadline or real-time safety claim.

| Method | p50 ms | p95 ms | p99 ms | Maximum ms |
|---|---:|---:|---:|---:|
| joint | 49.624 | 111.677 | 539.622 | 1117.601 |
| robust_current | 50.906 | 117.438 | 541.024 | 1177.030 |
| nominal_authority | 49.385 | 114.627 | 540.480 | 1151.263 |
| no_learning | 49.699 | 112.361 | 547.466 | 1131.602 |
| always_brake | 47.289 | 101.983 | 504.778 | 1117.565 |

Runtime overruns are retained. These recordings do not prove a 100ms real-time deadline. A deployment claim would require an execution platform and latency-aware closed-loop qualification.

## Fit with the requested call

The publisher's indexed title confirms Human-Centered and Trustworthy AI for Cybersecurity in Cyber-Physical Systems; the full publisher page returned403 on2026-10-24. This supplement supports the user's supplied themes through attack-resilient closed-loop simulation, explicit uncertainty/assumptions and observation-derived operator explanations. Full call compliance is not certified. The operator component has technical fidelity and navigation checks; no participant study, trust improvement, workload reduction or human-effectiveness result is claimed.

Primary framing source: He et al.,2021, The Challenges and Opportunities of Human-Centered AI for Trustworthy Robots and Autonomous Systems, https://arxiv.org/abs/2105.04408 . Its abstract discusses safety, security and usable interaction; it does not validate our viewer. Call: https://www.sciencedirect.com/special-issue/335890/human-centered-and-trustworthy-ai-for-cybersecurity-in-cyber-physical-systems .

The candidate novelty remains the specific context-history/execution-envelope/command-lifetime contract and its failure analysis. This supplement strengthens inspectability and comparison, and introduces no first-ever or native-framework superiority claim.
