# Conditional properties of the lifecycle contract

These properties concern the specified 10ms right-endpoint numerical model. They are not a new general safety theorem, a continuous-time vehicle proof, or a priority claim.

## Model and premises

At microstep h=0.01s, vE+=max(0,vE+h*aE), vL+=max(0,vL+h*aL), and g+=h*(vL+-vE+). The requested integer u is mm/s^2; aE=clip(u/1000+d,-4,2), |d|<=E<4. Lead acceleration is bounded below by-4. The executed backup command is-4000, with worst guaranteed deceleration b=4-E>0.

Bootstrap gap/lead intervals contain the actual state. Later context observations are unqualified: their agreement or valid signatures never narrow those intervals. Secure ego-state updates retain true speed if their initialization, process and available-domain attack-budget premises hold. Actual command execution respects the declared E. Loss of one of these premises removes the guarantee; the implementation does not infer source truth from a signature.

## Stopping distance

For speed v and deceleration b, positive right-endpoint velocities occur at indices1..n, n=max(0,ceil(v/(b*h))-1). The remaining distance is S(v,b)=h*(n*v-b*h*n*(n+1)/2). This includes the nonnegative-speed saturation exactly in real arithmetic.

## Property1: interval propagation

The speed update is monotone in initial speed and acceleration. Therefore propagating lower/upper ego speeds with clip(u-E) and clip(u+E), and lead speeds with-4 and+2, contains their next values. Gap lower/upper updates use h*(lead_low-ego_high) and h*(lead_high-ego_low). Induction preserves containment. The code rounds gap bounds outward. A secure ego intersection can tighten speeds because true speed is in both sets under its premises.

An unqualified context observation is used only for diagnostics/nominal proposals. It cannot relax the propagated lower gap or replace the lead lower bound. Consequently an arbitrary later shared range jump cannot enlarge the safety-authorized region. No automatic recovery or new bootstrap is inferred from subsequent source agreement.

## Property2: command safety with backup

For n held microsteps, simulate the maximum ego acceleration clip(u+E) and minimum lead acceleration-4 from worst interval endpoints. Record each gap. After the hold, switch to backup deceleration b. Since lead braking4>=b, the backup gap minimum occurs at its starting or terminal endpoint: if ego stops first the gap cannot develop a smaller interior value; otherwise relative speed decreases while both move and the lead stops first, after which gap decreases until ego stops. The terminal gap is g_n+S(vL_n,4)-S(vE_n,b).

Thus min(all held gaps-2, terminal gap-2)>=0 implies numerical gap>=2 over the held command and complete backup trajectory. Production arithmetic subtracts a small conservative numerical slack; the independent checker explicitly simulates the full stopping trajectory.

Safety of repeated authorizations additionally needs state inclusion and nonnegative recoverability at each step. The algorithm emits a conditional certificate only when these tests pass. A negative margin causes requested backup braking and an explicit absence of certification, not a false positive safety claim.

## Property3: maximal command and lifetime

The plant is monotone in ego acceleration. For a fixed hold, increasing u cannot improve the worst gap, so bisection finds the maximal safe integer command <= nominal proposal. Extending the held command replaces backup acceleration with an acceleration no smaller than backup; it cannot improve the worst gap. Consequently admissible hold lengths form a prefix of the discrete horizon and bisection finds its maximum. Lifetimes are capped at1s; issued commands use100ms and are bound to their observation digest, bounds, context, sequence, execution envelope and exact wire command.

The checker rejects expired, replayed, cross-context, cross-observation, mismatched-bound, mismatched-command and unsafe certificates even when a sidecar has a valid signature. Its state ledger and code are trusted components; arbitrary compromise of that checker is outside this threat model.

## Information and authority limits

If every range source and the bootstrap can be coherently forged, two different true gaps can produce identical signed packet histories. A deterministic packet-only policy must make identical decisions in those worlds. It cannot establish which gap is true from those packets alone. An extra trusted reference or bootstrap assumption is necessary for a range-containment claim.

If E>=4, backup braking no longer guarantees a positive deceleration. Likewise, an initially insufficient gap may be outside the recoverable set. No cryptographic check or learned estimator creates missing braking authority. These conditions are retained negative controls, not silently removed trials.

## What could be a contribution

The specific joint contract, failure attribution, finite-grid derivation, matched authority/history/learning ablations and audited engineering evidence may support a contribution after comparison. Secure estimation, stopping-distance safety filters, provenance, self-triggered control and certificate checking are existing ideas. Broad originality and journal acceptance are not established by this derivation.
