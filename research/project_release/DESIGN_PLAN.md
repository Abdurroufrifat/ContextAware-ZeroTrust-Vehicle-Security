# CO-ACT-CPS consolidated PC research release
23 October 2026. This release closes application integration and evidence navigation; it does not rename unmet research requirements as completed.

Goal: one reusable signed-report research session, one root command and one current result/claim index. Prior frozen studies remain authoritative for their own cohorts.

Architecture: ResearchSession owns the original CommandGateway and one selected temporal or scene-witness adapter. Callers provide signed packets, receiver time and risk context; they cannot provide an already-approved base decision. Four distinct signing keys, fixed study source profiles and explicit context are required. Session state uses a fresh directory inside this release runtime. Query time is monotone; existing state directories are not overwritten. This is offline research software, with no physical operation interface.

CLI: COACT_CPS_RESEARCH.py status shows the saved completion scope; verify validates the release and latest full evidence chain; demo replays the stored gradual-drift scene-witness trajectory using only signed packets/time/context. No dataset extraction or new training is required.

Integration validation: replay 34 existing trajectories through the new application API: both modes for all 12 cases at 90 ms/seed17, plus the other nominal age/seed/mode combinations. This is 6,800 exact decision comparisons with the existing frozen results, not new driving evidence. The complete 144-trial independent mathematical replay remains authoritative for the latest simulation study. Execute the focused and configured core test suites after application changes.

Evidence index: core software, recorded-data studies, classical/neural comparators, signed command gateway/tube, SUMO component study, S3E scalar comparison, temporal study and scene witness. Preserve reference failures, unavailable native inputs, coordinated compromise, rollback/restart and scene-independence limits. Never combine different dataset/error-reference/threat-model denominators.

Completion fields distinguish: research application integration, fixed PC evaluation scope, full original research objectives, physical validation, full published-framework reproduction, novelty and journal readiness. Passing software checks can establish only their documented scope. No learned-model result is assigned to the SUMO scene witness, and no human effectiveness is claimed.

Plan:
1. Behavioral tests and implementation of ResearchSession with strict input/context/key/state handling.
2. Exact signed-input replay through the application on all 34 chosen trajectories; one runnable demo.
3. Current evidence and claim index with source hashes and retained failures; readable supervisor summary.
4. Fresh combined tests, package integrity checks and one verification command; freeze final artifacts.
