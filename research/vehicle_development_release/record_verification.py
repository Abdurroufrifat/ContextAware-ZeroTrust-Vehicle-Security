from shared import *
import runpy
result=runpy.run_path(str(R/"verify_release.py"),run_name="release_independent_verification")
write("independent_verification.json",result["out"])
print(json.dumps(result["out"],indent=2))
