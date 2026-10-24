from release_common import *
install_guard()
import gzip,time,uuid
from research_session import ResearchSession,study_keys,strip_record
trials=json.loads((KW/"results.json").read_text())["trials"]
chosen=[t for t in trials if (t["age_us"]==90000 and t["seed"]==17) or t["case"]=="nominal"]
assert len(chosen)==34
run=R/"runtime"/("application_replay_"+uuid.uuid4().hex);run.mkdir()
summary=[];count=0;start=time.perf_counter()
for i,t in enumerate(chosen):
 with gzip.open(KW/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 session=ResearchSession(run/f"trial_{t['trial_id']:03d}",rows[0]["context"],study_keys(),mode=t["policy"])
 digester=hashlib.sha256()
 for row in rows:
  observed=session.process(strip_record(row))
  assert observed==row["decision"],(t["trial_id"],row["step"],"application output mismatch")
  assert all(observed[k] is False for k in ("physical_speed_authorized","physical_validation_established","promotion_allowed"))
  digester.update(json.dumps(observed,sort_keys=True,separators=(",",":")).encode())
  count+=1
 summary.append(dict(trial_id=t["trial_id"],case=t["case"],age_us=t["age_us"],seed=t["seed"],mode=t["policy"],decisions=len(rows),output_sha256=digester.hexdigest()))
 if (i+1)%6==0:print("Application replay",i+1,"of34 trajectories",flush=True)
assert count==6800
dump(R/"application_replay.json",dict(passed=True,trajectories=34,decisions=6800,
 scope="All12 cases at90ms/seed17/both modes plus every nominal age/seed/mode combination",
 elapsed_seconds=time.perf_counter()-start,comparisons=summary,new_driving_evidence=False,physical_authorization=False))
print("UNIFIED_APPLICATION_REPLAY_PASS",count,flush=True)
