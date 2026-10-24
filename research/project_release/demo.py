from release_common import *
install_guard()
import gzip,uuid
from research_session import ResearchSession,study_keys,strip_record
def run_demo():
 t=next(x for x in json.loads((KW/"results.json").read_text())["trials"] if (x["case"],x["age_us"],x["seed"],x["policy"])==("joint_slow_drift",90000,17,"kinematic"))
 with gzip.open(KW/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 session=ResearchSession(R/"runtime"/("demo_"+uuid.uuid4().hex),rows[0]["context"],study_keys(),mode="kinematic")
 selected=bad=corrected=0;examples=[]
 for row in rows:
  # The application gets only signed reports/time/risk. Reference fields are scored afterwards.
  out=session.process(strip_record(row));assert out==row["decision"]
  value=out["research_speed_mmps"]
  if value is not None:selected+=1;bad+=abs(value-row["ego_speed_mps"]*1000)>500
  corrected+=out["selection"]=="scene_witness"
  if row["step"] in (0,79,109,139):examples.append(dict(time_s=row["time_s"],reason=out["reason"],selected_speed_mps=out["research_speed_mps"],
   original_speed_mps=out["base_decision"]["research_speed_mps"],scene_status=out["scene_status"]))
 report=dict(case="stored gradual-drift simulation replay",mode="kinematic",queries=len(rows),selected=selected,
 selected_reference_error_gt05=bad,witness_replacements=corrected,examples=examples,
 source="Existing frozen simulation; public deterministic test keys",new_driving_evidence=False,physical_operation=False)
 dump(R/"runtime"/"last_demo.json",report);return report
if __name__=="__main__":print(json.dumps(run_demo(),indent=2))
