"""Offline artifact integrity and claim checks; no physical promotion."""
from common import *
install_guard()
manifest=json.loads((R/"file_manifest.json").read_text())
paths=manifest["files"]
assert paths and len({x["path"] for x in paths})==len(paths)
for row in paths:
    path=checked(R/row["path"])
    assert path.resolve().is_relative_to(R.resolve()),"outside module"
    assert path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],row["path"]
tests=json.loads((R/"test_validation.json").read_text())
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
assert tests["focused"]["counts"]["passed"]==422
assert tests["core"]["counts"]["passed"]==731 and tests["core"]["counts"]["skipped"]==4
v=json.loads((R/"verification.json").read_text())
assert v["passed"] is True and v["physical_frames"]==96 and v["comparisons_verified"]==864
assert v["fits_recomputed_by_independent_QR"]==288 and v["prediction_rows_changed_after_review"]==0
r=json.loads((R/"results.json").read_text())
assert r["scored_comparisons"]==r["reference_supported_comparisons"]==864
assert r["independent_physical_reference_comparison_completed"] is True
q=json.loads((R/"reference_qualification.json").read_text())
assert q["documented_separate_reference_instrument"] is True and q["independent_recorded_reference_comparison_completed"] is True
for name in ("hard_reference_error_bound_established","statistical_error_independence_established","physical_control_validation_established","uniform_05mps_true_error_certificate_established","promotion_allowed"):
    assert q[name] is False,name
a=json.loads((R/"contribution_assessment.json").read_text())
assert a["improved_cells"]==0 and a["worse_cells"]==a["paired_cells"]==9
assert a["hypothesis_supported"] is a["local_variant_promoted"] is a["specific_novelty_established"] is False
assert all(x["huber6_minus_huber3_mean_abs_error_mps"]>0 for x in r["paired"])
print(json.dumps({"package_integrity_passed":True,"manifest_files":len(paths),"recorded_comparisons":864,"focused_tests":422,"core_tests":731,"core_skipped":4,"novelty_established":False,"physical_control_validated":False,"promotion_allowed":False},indent=2))
