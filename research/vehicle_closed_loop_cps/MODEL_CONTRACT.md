# Numerical model and control contract
The controller observes only the strict input object in cps_session.py. The harness uses simulator state to generate synthetic sensor channels and to score outcomes. Controller code has no simulator handle or evaluator-truth argument.

The plant uses a 10 ms right-endpoint update with zero-speed contact. Acceleration is the accepted command plus a held bounded disturbance, saturated to [-4,+2] m/s^2. TraCI acceleration commands are refreshed every 10 ms; a longer-duration command produces different near-zero behavior and is not used. Query/control leases are 100 ms. Lead acceleration is bounded by [-4,+2] m/s^2.

Scene gap and absolute lead speed form one authenticated domain. A separately keyed synthetic radar domain supplies gap and relative speed. These channels have distinct injected noise, but share the SUMO plant. This is an explicit simulation construction, not measured statistical independence.
Assumed scene errors: gap +/-1.5 mm and lead speed +/-0.075 m/s. Radar errors: gap +/-0.05 m and relative speed +/-0.1 m/s. The frozen ego estimator's conditional speed-error budget is 0.5 m/s. HMAC authenticates origin and integrity; it does not make a compromised source truthful.

For admitted observations, the policy intersects the two gap intervals and the two lead-speed intervals, accounting for the ego-speed interval when converting relative speed. An empty intersection causes braking. No attack label is passed to this logic.

The nominal IDM-form command is restricted by a braking-reserve calculation. For each candidate integer command in [-4000,2000] mm/s^2, propagate the ego upper speed under command+1 m/s^2 (capped at +2), the lead lower speed under -4 m/s^2, and gap through ten 10 ms steps. Check the current gap and every step against 2 m. Then compare the resulting gap with right-endpoint stopping distances for ego braking at 3 m/s^2 and lead braking at 4 m/s^2. Under these braking rates the future gap minimum is at an endpoint; tests compare the closed-form calculation against explicit full braking paths.
The largest feasible command no greater than the nominal command is selected. If even maximum braking fails the reserve test, return -4 m/s^2 with reserve_unrecoverable, not a safety certificate.

The actuator gate checks an independently keyed command, context, sequence, current time, duration and numeric range. Invalid messages trigger configured braking. The simulator can separately violate actual execution bounds; these cases are retained and qualified separately.

Verification checks raw SUMO speed/position evolution, packet-derived controller inputs, manual frozen-forest predictions and integer projection, source-bound containment, actuator mutations, explanation fidelity, and the following 100 ms of measured gap where the numerical envelope applies. Rounded FCD precision is accounted for. Exact discrepancy-count thresholds use the logged full-precision errors after independent FCD cross-check.

This is conditional numerical evidence. Shared semantic corruption, actual execution outside bounds, unmodelled association changes, lateral traffic and rendered perception are outside the stated guarantee. The experiment still reports every tested failure. Operator cards expose reasons and assumptions; no human-participant outcomes are claimed.

Replay counters in the new radar/actuator gates are in memory. These experiments cover uninterrupted sessions; crash recovery, durable rollback protection and production key management are not claimed.
