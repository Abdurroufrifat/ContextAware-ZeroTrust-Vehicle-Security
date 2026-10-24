from shared import *
from pathlib import Path
core=R/"release_core.py";source=core.read_text(encoding="utf-8")
needle='    return target\ndef verify_manifest'
patch='''    resolved_low=str(target).replace(chr(92),"/").lower()
    if "library03" in resolved_low or "rav4" in resolved_low or "vehicle_final_evaluation" in resolved_low or ("urban02" in resolved_low and "night" in resolved_low):raise ValueError("resolved reserved artifact")
    return target
def verify_manifest'''
assert source.count(needle)==1;source=source.replace(needle,patch)
source=source.replace('manifest=json.loads((root/"sha256.json").read_text(encoding="utf-8"));path=root/"reference_bounds.py"',
 'manifest=json.loads(artifact_path(root,"sha256.json").read_text(encoding="utf-8"));path=artifact_path(root,"reference_bounds.py")')
source+='''
def validate_scenario_identity(row,historical,index):
    expected={"row":index,**{key:historical[key] for key in ["sequence","group","scope","epsilon_assumption_mps"]}}
    for key,value in expected.items():
        if key not in row or type(row[key]) is not type(value) or row[key]!=value:
            raise ValueError("historical scenario identity mismatch: "+key)
'''
core.write_text(source,encoding="utf-8")
verify=R/"verify_release.py";source=verify.read_text(encoding="utf-8")
source=source.replace('from fractions import Fraction','from fractions import Fraction\nfrom release_core import validate_scenario_identity')
needle='for row,prior in zip(rows,historical):'
assert source.count(needle)==1
source=source.replace(needle,'for index,(row,prior) in enumerate(zip(rows,historical)):\n    validate_scenario_identity(row,prior,index)')
verify.write_text(source,encoding="utf-8")
previous=read("pre_review_fix/experiment_lock.json")
current={name:sha(R/name) for name in previous};current["verify_release.py"]=sha(verify)
write("experiment_lock.json",current)
write("review_fix_record.json",{"reason":"Two static-review verification defects; numerical formulas and recorded outcomes unchanged",
 "previous_lock_sha256":sha(R/"pre_review_fix/experiment_lock.json"),
 "previous_source_sha256":{name:sha(R/"pre_review_fix"/name) for name in ["release_core.py","verify_release.py","test_release_core.py","independent_verification.json"]},
 "current_lock":current,"changes":["reject reserved paths after filesystem resolution, including exact-helper source paths","compare every sidecar scenario identity field with its historical source row"]})
for name in ["finish_release.py","audit_package.py"]:
    path=R/name;path.write_text(path.read_text(encoding="utf-8").replace("177","184"),encoding="utf-8")
(R/"REVIEW_FIXES.md").write_text("Two review defects were reproduced by7regression cases, then corrected. Artifact validation now checks resolved filesystem targets, including junction aliases and exact-helper sources, before opening. The verifier matches row index/sequence/group/scope/epsilon and types to historical rows before recomputation. Pre-fix source/test/lock/verification snapshots remain under pre_review_fix; review_fix_record.json records old/new identities. The current source lock is a documented post-review correction, not a retuning of models, references or outcomes. Fresh regression and independent verification are required before release.\n",encoding="utf-8")
print("Review fixes applied; pre-fix snapshot preserved. Current verification source lock updated explicitly.")
