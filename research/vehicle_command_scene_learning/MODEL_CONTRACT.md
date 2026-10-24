# Discrete command-scene estimation contract
This is implementation documentation and a conditional proof, not a manuscript or a physical test certificate.

Let delta=0.01 seconds, h=N*delta<=0.1 seconds. The simulator updates positions using the new velocity at each discrete step. Denote ego/lead speeds by e_j and l_j. Signed scene observations describe the same fixed lead vehicle at both timestamps. Both scene timestamps equal the receiving query clock.

## Assumptions
1. Each signed gap has absolute error <=0.0015m. Both endpoint lead speeds have error <=0.075m/s. Authentication establishes origin/integrity; these error bounds remain external truth assumptions.
2. Lead discrete increments satisfy -4*delta <= l_j-l_(j-1) <= 2*delta, including the nonnegative-speed floor.
3. Complete authenticated command records uniquely cover each 10ms cell. When ego speed stays positive, actual increments lie within delta*[max(-4,u_j-1),min(2,u_j+1)].
4. The original scene interval [L_b,U_b] contains current ego speed. Refinement runs only when L_b>2h. Since no forward increment exceeds 2*delta, every earlier ego speed in the window is then strictly positive. This establishes the no-zero-contact condition used in assumption3.
5. Scene target identity, clock, geometry, command execution and trusted host are valid. The fixed SUMO vehicle is named lead; general target association is not authenticated by this packet schema.

## Identity
The gap update gives mean_right(ego)=mean_right(lead)-(gap_N-gap_0)/h.
Telescoping ego increments gives
e_N = mean_right(lead) -(gap_N-gap_0)/h + sum_(j=1..N) ((j-1)/N)*(e_j-e_(j-1)).
The first step has weight zero. Replacing these weights by j/N would be incorrect for the right-endpoint update.

## Lead envelope
At time j*delta, the lower feasible speed is the maximum of zero, the old lead lower endpoint minus4*j*delta, and the new lead lower endpoint minus2*(h-j*delta).
The upper feasible speed is the minimum of the old upper endpoint plus2*j*delta and the new upper endpoint plus4*(h-j*delta).
Average the bounds for j=1..N. The production code computes endpoint cones using Fraction. The independent verifier derives the same bounds by forward/backward difference-constraint propagation.

## Current ego interval
Subtract the observed gap rate, widen by2*0.0015/h, and add the nonnegative weighted sums of the command increment bounds. Intersect with [L_b,U_b].
Every term brackets the corresponding term in the identity, so a true current ego speed satisfying all assumptions lies in the resulting [L_r,U_r]. Empty intersections reject admission. Unavailable coverage/grid or possible zero contact preserve the original scene result.

The intersection is a subset of the original scene interval, under the additional command-execution premise. Thus its admissible point set [U_r-0.5,L_r+0.5] contains the original admissible point set. Integer mm/s outputs use inward ceiling/floor endpoints. Any model point projected into that set differs by <=0.5m/s from every feasible true speed. This is an estimation-discrepancy statement, not a collision-avoidance or stability theorem.

## Certificate separation
The modified estimate sets scene_budget_satisfied=False and refined_budget_satisfied=True. Its guarantee needs BOTH valid scene observations and valid command execution/log coverage. The old scene-only decision is retained inside scene_decision.
When refinement is unavailable, the original scene-only result and its guarantee are preserved. Scoring checks these two certificate types separately against raw trajectories.

## Identifiability limit
If every measurement domain may lie, two different constant-speed worlds can emit identical signed packets while both obeying zero acceleration. Choose speeds10 and12m/s and shift lead speed accordingly while preserving true gap; compromised sources can report the same fabricated speeds and scene stream in both worlds. Any deterministic estimate sees identical inputs, so it cannot be within0.5m/s of both states. Randomization cannot provide a universal guarantee either.
This is a standard indistinguishability argument. More neural layers, HMAC tags and software tests cannot establish truthful physical measurements without an uncompromised information source or a valid external bound.

## Physical evidence
The earlier Boreas study compares96 actual Aeva frames with a separately instrumented GNSS/IMU/wheel reference. It is recorded physical-data evidence. It does not provide this scene/lead/command interface, verified hard bounds, physical source attestation or controlled real-vehicle response. Its positive and negative results remain unchanged.
The discrete theorem is tested in SUMO. Applying it to continuous physical vehicles requires qualified timing, sensor errors, target association, execution bounds and a justified integration model.

Controller scope: the IDM-form longitudinal controller receives uncorrupted simulator lead speed and gap, as in the frozen predecessor experiment. Attacks alter the admission input streams. In the common-scene-compromise case, the signed scene stream presented to the estimator is corrupted; the separate controller context is still ideal. Zero collisions in this grid are not evidence of whole-perception-stack resilience.
