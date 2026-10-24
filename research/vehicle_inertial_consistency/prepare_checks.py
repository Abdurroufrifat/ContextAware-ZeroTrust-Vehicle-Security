from pathlib import Path
R=Path(__file__).resolve().parent
source=(R.parent/"vehicle_neural_adaptation/run_checks.py").read_text(encoding="utf-8")
old='targets=[R/"test_neural_core.py",'
new='targets=[R/"test_inertial_core.py",R/"test_diagnostic_details.py",R.parent/"vehicle_neural_adaptation"/"test_neural_core.py",'
assert source.count(old)==1
(R/"run_checks.py").write_text(source.replace(old,new),encoding="utf-8")
print("Created focused/core checks with new inertial tests and previous110 focused tests.")
