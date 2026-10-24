import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
s=json.loads((R/"quality_summary.json").read_text());run=R/s["runtime_relative_path"]
rows=json.loads((run/"predictions.json").read_text());check=[r for r in rows if r["quality_role"]=="check"]
bad=[r for r in check if r["error_mps"] is not None and r["error_mps"]>2]
unknown=[r for r in check if r["error_mps"] is None]
# Grant every unknown/missing/invalid case the benefit of the doubt: still an upper ceiling.
ceiling=len(check)-len(bad)
result=dict(analysis="post-result arithmetic diagnosis, not a second candidate or parameter change",
 primary_summary_sha256=hashlib.sha256((R/"quality_summary.json").read_bytes()).hexdigest(),
 check_requests=len(check),known_large_errors=len(bad),unscorable_requests=len(unknown),
 optimistic_oracle_rejection_only_max_accepts=ceiling,
 optimistic_oracle_rejection_only_max_coverage_pct=100*ceiling/len(check),
 availability_target_pct=95,
 interpretation="Any filter that only accepts/rejects these unchanged speed estimates must reject the 72 known errors above2m/s. Even granting every remaining case safe, at most45/117 can be accepted. This is an upper ceiling, not achieved safe coverage.",
 conclusion="additional calibration blocks alone cannot repair the underlying estimator or meet95% availability; a materially better independent motion estimate and adequate independent calibration are both required",
 policy_changed=False,promotion_allowed=False)
(R/"quality_limit_diagnosis.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
