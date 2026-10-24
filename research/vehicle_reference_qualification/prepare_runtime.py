from pathlib import Path
R=Path(__file__).resolve().parent;I=R.parent/"vehicle_inertial_consistency"
(R/"shared.py").write_text((I/"shared.py").read_text(encoding="utf-8"),encoding="utf-8")
source=(I/"run_checks.py").read_text(encoding="utf-8")
old='targets=[R/"test_inertial_core.py",R/"test_diagnostic_details.py",'
new='targets=[R/"test_reference_core.py",R.parent/"vehicle_inertial_consistency"/"test_inertial_core.py",R.parent/"vehicle_inertial_consistency"/"test_diagnostic_details.py",'
assert source.count(old)==1
(R/"run_checks.py").write_text(source.replace(old,new),encoding="utf-8")
print("Reference-audit runtime and full focused/core checks prepared.")
