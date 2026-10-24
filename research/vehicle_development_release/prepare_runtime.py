from pathlib import Path
R=Path(__file__).resolve().parent
source=(R.parent/"vehicle_reference_qualification/shared.py").read_text(encoding="utf-8")
(R/"shared.py").write_text(source,encoding="utf-8")
source=(R.parent/"vehicle_reference_qualification/run_checks.py").read_text(encoding="utf-8")
old='targets=[R/"test_reference_core.py",'
new='targets=[R/"test_release_core.py",R.parent/"vehicle_reference_uncertainty"/"test_reference.py",R.parent/"vehicle_reference_qualification"/"test_reference_core.py",'
assert source.count(old)==1
source=source.replace(old,new).replace('import pytest','sys.path.insert(0,str(R.parent/"vehicle_reference_uncertainty"))\nimport pytest')
(R/"run_checks.py").write_text(source,encoding="utf-8")
print("Prepared exact-bound and previous regression checks.")
