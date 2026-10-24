# Evidence lifecycle implementation plan
> Agentic execution: inline; user opted out of repeated questions.
Goal: implement and assess the joint authorization hypothesis from DESIGN.md.
Tech stack: Python3.12, existing frozen forest/HMAC adapters,10ms plant. No new paid services.
Constraints: preserve existing sources/results; simulation scope; no paper/commit.
Review focus: false bootstrap, unqualified shared refresh, envelope exceedance, stale/replayed certificates, trivial stopping.
1. lifecycle_core.py/test_lifecycle_core.py: RED then margin/lease/tube invariants GREEN, including loss of recoverability atE=2.5.
2. packet_policy.py/certificate.py tests: strictly signed inputs; HMAC/capsule binding and independent checker; no truth inputs. Source speed domains and all actor assumptions explicit.
3. harness.py/PROTOCOL: common fresh task with robust S3E/SSF adaptations, history and authority ablations, no-learning and always-brake controls. Development first, lock fresh confirmation before scoring.
4. verify.py/PROPERTIES.md/report.py: independent dynamics, interval/certificate checks, packet replay, full grid, source hashes and honest negative controls. Project tests and final manifests.
