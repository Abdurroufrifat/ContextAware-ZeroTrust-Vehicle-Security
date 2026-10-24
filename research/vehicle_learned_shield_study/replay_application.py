"""Actual signed-input API replay of all 54 protected-model trajectories."""
from learned_common import *
install_guard()
import gzip,uuid,time
from learned_session import LearnedResearchSession
from research_session import study_keys,strip_record
trials=[x for x in json.loads((R/"results.json").read_text())["trials"] if x["policy"]=="learned_shielded"]
assert len(trials)==54
root=R.parent/"project_release"/"runtime"/"learned_shield_application"/uuid.uuid4().hex
start=time.perf_counter();checks=[];count=0
for t in trials:
 with gzip.open(checked(R/t["trace"]),"rt",encoding="utf-8") as f:rows=[json.loads(line) for line in f]
 session=LearnedResearchSession(root/f"trial_{t['trial_id']:03d}",rows[0]["context"],study_keys())
 digest=hashlib.sha256()
 for row in rows:
  output=session.process(strip_record(row));assert output==row["decision"],(t["trial_id"],row["step"])
  digest.update(json.dumps(output,sort_keys=True,separators=(",",":")).encode());count+=1
 checks.append(dict(trial_id=t["trial_id"],queries=len(rows),output_sha256=digest.hexdigest()))
 if len(checks)%9==0:print("Application replay",len(checks),"of54",flush=True)
assert count==10800
write("application_replay.json",dict(passed=True,trials=len(checks),decisions=count,checks=checks,
 seconds=time.perf_counter()-start,application_source_sha256={str(q.relative_to(P)):sha(q) for q in (R/"learned_session.py",R/"learned_core.py",R/"replay_application.py")},truth_fields_passed_to_api=False,new_driving_trials=0,physical_vehicle_operation=False))
print("LEARNED_SIGNED_APPLICATION_REPLAY_PASSED",count,flush=True)
