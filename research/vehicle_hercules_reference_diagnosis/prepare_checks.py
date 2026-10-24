from pathlib import Path
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development"
text=(H/"run_checks.py").read_text()
target='[str(R/"test_hercules_core.py"),str(R.parent/"vehicle_independent_source_qualification"/"test_pilot_core.py"),str(R.parent/"vehicle_independent_source_qualification"/"test_native_compat.py")]'
replacement='[str(R/"test_source_stress.py"),str(R.parent/"vehicle_hercules_development"/"test_hercules_core.py"),str(R.parent/"vehicle_independent_source_qualification"/"test_pilot_core.py"),str(R.parent/"vehicle_independent_source_qualification"/"test_native_compat.py")]'
assert target in text
(R/"run_checks.py").write_text(text.replace(target,replacement),encoding="utf-8")
print("Prepared offline core and35focused checks.")
