# CO-ACT-CPS closed-loop simulation release

This release completes the simulation scope selected by the user. It supplements the earlier frozen research evidence without changing its historical conclusions. It does not include a manuscript.

## Read the evidence
- RESULTS.md: complete confirmation results and qualified claim boundaries.
- aggregate_results.csv: all 60 scenario/policy cells and five overall summaries.
- paired_results.json and seed_sensitivity.json: matched scenario comparisons and descriptive seed sensitivity.
- operator_view.html: offline explanation viewer; open directly in a browser.
- MODEL_CONTRACT.md: sensor, clock, command and numerical plant assumptions.
- EAAI_ALIGNMENT.md: special-issue topic mapping and published-method distinctions.
- verification.json and report_verification.json: independent computational checks.
- completion_status.json: final scope-specific status, created only after verification.
- environment.json: versions used on this computer.

## Verify without retraining or new simulations

From PowerShell:

    Set-Location 'D:\CO-ACT-CPS\phase2_joint_history'
    & 'D:\CO-ACT-CPS\.venv\Scripts\python.exe' .\COACT_CPS_SIMULATION.py verify

This checks release hashes, all 540 raw confirmation traces and trajectories, report aggregation, new tests, original project tests and the previous frozen learning package. Logs go to runtime/check_N.log. A failing check returns a nonzero exit code. This is a numerical replay, not another confirmation study.

To display status or one operator explanation, use the same Python executable with COACT_CPS_SIMULATION.py status or COACT_CPS_SIMULATION.py operator.

## Experiment provenance
PROTOCOL.json, frozen model dependencies and all simulation behavior files were hashed into experiment_lock.json before the first confirmation trial. Three independent process workers execute SUMO; no GPU is required. Every scheduled trial is retained, including assumption-breaking conditions. No model was retrained using confirmation results.

The synthetic sensor channels are separate keyed observations of one numerical plant. This is not rendered perception or an empirical independence claim. Cases include authenticated semantic corruption, shared-source compromise, command-message tampering, false-authority advisories and execution failures.

Five policies separate the effects of sensing, learned estimation, source admission, control filtering and actuator authentication. These are component baselines; they do not reproduce every published method natively. The earlier negative learning results remain preserved.

runtime/pilot*, archived runner snapshots and RED/GREEN logs are development evidence, not confirmation results. package_manifest.json covers the final release and its data; experiment_lock.json covers pre-confirmation inputs. Excluded cache and verification-run logs are mutable.

Human-centered support consists of faithful evidence cards and a viewer. No participant study, measured human trust or general persuasion defense is claimed. Editorial assessment remains a separate decision.
