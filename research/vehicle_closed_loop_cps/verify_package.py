from cps_common import R,P,sha
import json
manifest=json.loads((R/"package_manifest.json").read_text())
def verify(files):
    for n,digest in files.items():
        target=P/n
        assert target.resolve().is_relative_to(P.resolve()),"package scope"
        assert target.is_file() and sha(target)==digest,n
verify(manifest["files"])
development=json.loads((R/"development_manifest.json").read_text())
verify(development["files"])
lock=json.loads((R/"experiment_lock.json").read_text())
verify(lock)
print("CPS_PACKAGE_INTEGRITY_PASS",len(manifest["files"]),"release files;",len(development["files"]),"development files;",len(lock),"frozen experiment inputs")
