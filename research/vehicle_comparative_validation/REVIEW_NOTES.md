# Review and evidence boundary

One independent read-only reviewer inspected the new method adaptations, ablation isolation, finite-grid barrier, independent arithmetic and release gates. Important findings were repaired before final evidence sealing; failing regression logs and pilot attempts remain in runtime.

The refinement ablation changes only successful interval refinement and matches fallback logic. The barrier checks current/all10ms headway plus terminal contraction. Scalar plant parity covers all20s and all5 lead segments for each new profile.

Independent mathematics and same-algorithm packet replay are separate evidence. The verifier authenticates accepted-command history, checks declared heldout sources and prescribed attacks, and recomputes every transition and metric. Replay uses fresh state and full packet-only input; its cache is bound to exact trace, behavior lock and replay-engine digest.

Final completion is final_seal.json, published only after final manifest checks. A failure-injection test proves a failed final seal does not publish completion. Active pipeline logs are excluded from immutable manifests. The pipeline checks runner liveness and resumes only unfinished trial IDs; root resume uses atomic single-pipeline ownership and preserves stale locks after checking command identity.

Limitations are part of the report: finite scalar numerical simulation; common-task published-method adaptations; public receipt/motion stress proxies; six-seed exploratory uncertainty; no human study or journal acceptance certification. No physical hardware gate is imposed on the user-authorized simulation scope.
