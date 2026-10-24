# Published-method adaptation contract
Primary sources:
- Niazi et al., S3E, arXiv2309.05075v4 (31 July2026), Algorithm1 and equations4,6,7,8: https://arxiv.org/html/2309.05075v4
- Tan et al., CDC2025, arXiv2505.06842v1, plausible-state reconstruction and secure safety filter equations11-13 with relaxed-observability section: https://arxiv.org/html/2505.06842v1

These are explicit one-dimensional common-task adaptations. They are not native reproductions of the papers' complete zonotope/unicycle stacks, and their general theorems are not transferred automatically.

## Shared input and permissions
Every method receives exactly the same signed packet schema: candidate ego speed, delayed anchor ego speed, accepted-command history, current scene gap/absolute lead speed, and separately keyed radar gap/relative speed. Cryptographic/time/context admission uses the frozen gateway. Radar and scene are different simulation source domains; synthetic channels share the numerical plant.
The comparator measurement adapter consumes candidate and anchor only when authenticated. Its third ego-speed interval is the frozen scene-pair witness calculated from consecutive signed scene packets. That witness does not use candidate or anchor readings. Radar is used for context intersection, not counted as another independent ego observation.
Common declared errors: candidate0.101m/s (including rounding), anchor0.25m/s plus motion propagation and clock allowance, scene gap0.0015m/lead0.075m/s, radar gap0.05m/relative0.1m/s. The original estimator's0.5m/s certificate uses its own additional scene/command premises. These are not statistically calibrated hard sensor bounds.

## S3E specialization
State x is ego speed. Initialize with the public operating range[0,30]m/s, which contains all configured initial ego speeds; no simulator initial state is passed to the estimator.
Prediction maps each retained interval through accepted command u and bounded execution error?1m/s^2 with acceleration saturation[-4,+2] and zero-speed clipping. This clipping is a declared monotone-plant adaptation of the paper's linear time update.
Current candidate interval is speed?0.101m/s; anchor interval is its measured speed?0.25m/s, propagated with[-4,+2]m/s^2 over its age and widened for5ms clock uncertainty. Scene-pair interval comes from the frozen kinematic identity.
For p available scalar domains and q=1, form all p-q observation intersections and intersect their union with the predicted union. This equals per-predicted-set measurement update, agreement and union in1D. Preserve disconnected components; do not intersect mutually inconsistent hypotheses or collapse before the next prediction. If p<=q, no available-honest-source guarantee is possible and the set is empty.
An empty state is not reset from ground truth; subsequent temporal sets remain empty. This cost outside the premise is retained. The current-set SSF variant may recover next query by its different estimator design.
The reported point is the midpoint of the set's global hull. It may lie between hypotheses. Its separate0.5m/s point certificate is true only when it is within0.5m/s of every interval endpoint. Safety filtering uses the entire plausible-state envelope even if no point-discrepancy certificate exists.
The source-budget premise applies to these three ego observation domains only; it does not imply scene/radar context truth. Shared semantic context corruption is scored as a context-premise failure.

## Relaxed reconstruction and ZO-CBF adaptation
Use the same current ego intervals and two-of-three agreement union, without the temporal prior. This is an interval-valued instantaneous scalar reconstruction (C_i=1), not a general nonlinear observability-map implementation.
Use the common scene/radar context intersection with the ego-set envelope to derive conservative lead/gap bounds.
Nominal input is the same IDM-form controller. Set h=g-2-v_ego (1s headway). At each100ms lease choose the integer command u?[-4000,u_nom] minimizing squared deviation, subject to h_next>=0.8*h_current under worst ego execution u+1, lead braking-4, and right-endpoint10ms dynamics. Also check current and every microstep gap>=2.
Monotonicity makes the largest feasible integer command the least-change solution; use binary search and compare against exhaustive search in tests. If infeasible, return-4m/s^2 and a negative feasibility margin.
Initial h>=0, correct source envelopes, bounded execution and online feasibility are necessary for a conditional barrier interpretation. Finite-grid checks and SUMO parity are reported, not a continuous-time safety theorem.

## Attribution and fairness
s3e_brake shares the proposed braking-reserve filter and actuator gate; differences from full concern estimation under the explicitly different source-budget assumptions.
ssf_zocbf changes both reconstruction and filter; its comparison is a complete adapted method comparison, not an isolated filter ablation.
no_learning versus no_refinement isolates command-scene refinement with midpoint selection. full versus no_learning isolates the frozen learned point selector.
full versus no_fusion/no_brake/no_actuator_auth removes exactly the named downstream component.
No native RAIL, GUARDIAN or AVMON numerical-superiority claim is supported by this study. All-case outcomes and assumption-qualified counts remain separate.

Refinement-ablation clarification: no_refinement uses the broad scene interval midpoint only when the frozen refined-midpoint arm reports successful refinement. All unavailable/conflict fallback selections remain identical to no_learning. This isolates successful interval refinement, not every behavior of deleting the whole module; report its eligible-row count. It computes refinement eligibility but does not use refined bounds for the ablated point. Full versus no_learning remains the untouched frozen learning-selector comparison.
The ZO-CBF adapter checks current/every10ms headway h>=0, terminal contraction and gap, including the zero-speed crossing; no intersample continuous-time theorem transfer.
