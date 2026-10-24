# Closed-loop CPS implementation plan
> For agentic workers: execute inline using superpowers:executing-plans; one fresh final review.
Goal: finish the user-approved simulation scope with attacked observations driving control and reproducible evaluation.
Architecture: strict packet-only policy session, generic signed radar/actuator messages, numerical stopping-reserve filter, SUMO experiment runner, independent trace/trajectory verifier and operator report.
Tech stack: existing Python 3.12 environment, SUMO/TraCI, frozen scikit-learn forest; no new dependencies.
Spec: DESIGN.md and PROTOCOL.json.
Global constraints: work in this new research folder and a new root completion entry; preserve all old frozen files and protected data. No manuscript, vehicle operation, git commit or publication.
Review focus: truth leaking into control; right-endpoint dynamics and zero-speed contact; forged/replayed/context-mismatched messages; shared-source compromise and execution-bound failures; explanations overstating guarantees or human validation.
- [x] Task 1: write tests for message trust, causal control inputs, braking projection, explanation fields and adversarial advisories; watch RED; implement and run GREEN.
- [x] Task 2: integrate frozen estimator, new trust domains and controller in SUMO; run development mechanics checks; freeze code/model/protocol; execute 540 fresh trials and retain every result.
- [x] Task 3: independently verify serialized control decisions, raw FCD states, command execution and summaries; run project and new tests; obtain fresh review and fix important findings.
- [x] Task 4: create the scope-specific completion report, operator explanation viewer, reproducibility manifest and a single verification command. Hardware is out of scope; no acceptance guarantee or human-study result.
