from cs_common import *
install_guard()
import gzip,uuid,time
from command_scene_session import CommandSceneSession
from research_session import strip_record,study_keys
cfg=json.loads((R/"PROTOCOL.json").read_text());trials=[t for t in json.loads((R/"results.json").read_text())["trials"] if t["seed"]==cfg["seeds"][0] and t["policy"] in ("refined_midpoint","refined_learned")]
assert len(trials)==54
root=R.parent/"project_release"/"runtime"/"command_scene_application"/uuid.uuid4().hex
start=time.perf_counter();checks=[];count=0
for trial in trials:
 with gzip.open(checked(R/trial["trace"]),"rt",encoding="utf-8") as f:rows=[json.loads(line) for line in f]
 session=CommandSceneSession(root/f"trial_{trial['trial_id']:03d}",rows[0]["context"],study_keys(),mode=trial["policy"])
 digest=hashlib.sha256()
 for row in rows:
  output=session.process(strip_record(row));assert output==row["decision"],(trial["trial_id"],row["step"])
  digest.update(json.dumps(output,sort_keys=True,separators=(",",":")).encode());count+=1
 checks.append(dict(trial_id=trial["trial_id"],decisions=len(rows),sha256=digest.hexdigest()))
 if len(checks)%9==0:print("Signed application",len(checks),"of54",flush=True)
assert count==10800
write("application_replay.json",dict(passed=True,trials=len(checks),decisions=count,checks=checks,seconds=time.perf_counter()-start,
 source_bindings={str(q.relative_to(P)):sha(q) for q in (R/"command_scene_session.py",R/"command_scene_core.py",R/"command_scene_guard.py",R/"replay_application.py")},
 truth_fields_passed_to_model=False,new_driving_evidence=False,physical_vehicle_operation=False))
print("SIGNED_COMMAND_SCENE_APPLICATION_PASSED",count,flush=True)
