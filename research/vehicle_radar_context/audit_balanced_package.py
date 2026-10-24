import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/"balanced_file_manifest.json").read_text())
for name,want in manifest.items():
    p=ROOT/name
    assert p.parent==ROOT and hashlib.sha256(p.read_bytes()).hexdigest()==want,name
report=json.loads((ROOT/"balanced_experiment_summary.json").read_text())
verification=json.loads((ROOT/"balanced_independent_verification.json").read_text())
completion=json.loads((ROOT/"balanced_completion.json").read_text())
assert verification["all_passed"] and verification["policy_decisions"]==report["policy_decisions"]==27168
q=report["results"]["sensor_balanced_v2"]["raw_header_association"]
nominal=q["nominal"]["context_no_ood_plus_radar"]
assert nominal["coverage_pct"]==completion["nominal_coverage_pct"]
assert nominal["accepted"]==224 and nominal["unscorable_accepted"]==0
assert all(cell["context_no_ood_plus_radar"]["accepted_discrepancy_above_2_mps"]==0 for cell in q.values())
assert not completion["full_project_complete"] and not completion["promotion_allowed"]
assert report["quality"]["sensor_balanced_v2"]["processed_corrected_time"]["accepted"]==0
print(json.dumps(dict(package_hashes_verified=len(manifest),decisions_verified=27168,full_project_complete=False,
    nominal_accepted=224,nominal_requests=283,development_only=True),indent=2))
