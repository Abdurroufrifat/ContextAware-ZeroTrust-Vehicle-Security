# Context Aware Zero Trust for Vehicle Security

Multi-source context aware authorization for longitudinal vehicle control under corrupted sensing and actuation. Internal Python package and frozen module names retain CO-ACT-CPS identifiers for compatibility.

## Implemented mechanisms

* Authenticated source admission with freshness, replay and session checks.
* Plausible ego speed sets and a frozen residual forest.
* Retained lead and gap history, bounded execution and a discrete stopping reserve.
* Certificates tied to the exact acceleration, evidence and expiry.
* An actuator watchdog and a read-only operator assurance viewer.

This is a cyber physical simulation project. Physical vehicle safety, human effectiveness and hard real time behavior are not established.

## Saved experimental status

| Study | Saved trials | Interpretation |
|---|---:|---|
| Original closed loop study | 540 | Earlier development evidence |
| Common task comparison | 1,512 | Earlier component ablations and scalar published method adaptations |
| Final lifecycle confirmation | 1,080 | Frozen three profile, twelve condition, six seed, five policy grid |

The studies have different protocols and are not pooled. Final joint policy contacts are 6/216 versus 42/216 for current context. Mean progress is 135.654 m versus 224.062 m. Learning reduces speed MAE from 0.36183 to 0.27791 m/s compared with midpoint selection, while contact counts are equal. All six joint contacts occur in the declared authority exceedance condition.

See [final results](research/vehicle_evidence_lifecycle/confirmation_v2/RESULTS.md), [contract](research/vehicle_evidence_lifecycle/V2_CONTRACT.md), [properties](research/vehicle_evidence_lifecycle/PROPERTIES.md), and [export scope](docs/EXPORT_SCOPE.md).

## Operator viewer

Open research/vehicle_submission_assurance_20261024/operator_assurance_viewer.html in a browser. It contains saved replay cards and distinguishes point estimation assurance from command permission. No participant study is claimed.

## Install and inspect

Use Python 3.12 in an isolated environment:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

The original package dependencies and source structure are retained. The final benchmark used scikit-learn 1.9.0; loading frozen models with another version is not qualified here. Historical SUMO scripts also require SUMO/TraCI and some retain the recorded Windows installation path.

Final authorization code:

* research/vehicle_evidence_lifecycle/lifecycle_core.py
* research/vehicle_evidence_lifecycle/packet_policy_v2.py
* research/vehicle_evidence_lifecycle/certificate_v2.py
* research/vehicle_comparative_validation/policy.py
* research/vehicle_command_scene_learning/models/forest.joblib

Focused tests are beside the research modules. Historical source remains for traceability; external data workflows still require their inputs.

## Export contents and limits

Canonical source code, tests, configurations, frozen command-scene models, result summaries, audit summaries and the standalone viewer are included. Source files are copied byte for byte. Newline conversion is disabled to preserve frozen hashes.

Raw datasets, protected evaluation payloads, runtime databases, per-trial archives and checkpoints (except one synthetic unit test trace), environment folders, manuscripts and downloaded papers are excluded.

Original status/resume/full audit entry points expect the complete local evidence archive. They should not be used to infer missing trials from this compact export. Copied result files record completed studies; omitted checkpoint folders do not mean those trials failed or were unfinished.

Full independent replay requires separately preserved traces, lock inputs and data prerequisites. Some benchmark harnesses require derived public sequence input that is not redistributed here. Historical public motion source: Driving Data of a Real F1tenth Car, DOI 10.5281/zenodo.12536536. Dataset licenses are separate from the source license.

## License and provenance

The existing MIT license is retained. PUBLICATION_PROVENANCE.json records copied file hashes and original Git identity. Public deterministic fixture keys are test values, not production credentials. No experiments were rerun for publication. The original repository and working checkout are preserved.

## Focused checks without the full archive

The exported contract and packet policy tests use one saved synthetic nominal trace. Run from the repository root on Windows:

    New-Item -ItemType Directory -Force research/project_release/runtime
    $env:PYTHONPATH="src;scripts;research/vehicle_evidence_lifecycle"
    .\.venv\Scripts\python.exe -m pytest -q research/vehicle_evidence_lifecycle/test_lifecycle_core.py research/vehicle_evidence_lifecycle/test_certificate.py research/vehicle_evidence_lifecycle/test_certificate_v2.py research/vehicle_evidence_lifecycle/test_packet_policy.py research/vehicle_evidence_lifecycle/test_packet_policy_v2.py --basetemp research/project_release/runtime/publication_unit_tests

The full harness import test additionally needs the external derived public_sequences.json input and is not part of this compact export check. These are software checks, not new vehicle experiments.
