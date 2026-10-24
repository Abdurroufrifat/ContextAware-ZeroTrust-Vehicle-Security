"""Preserve the first run and fix explanation labels without changing any decisions."""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    snapshot=ROOT/"initial_v1_snapshot"
    snapshot.mkdir(exist_ok=False)
    for name in ("run_radar_experiment.py","radar_experiment_summary.json"):
        shutil.copyfile(ROOT/name,snapshot/name)
    path=ROOT/"run_radar_experiment.py"
    text=path.read_text()
    text=text.replace("from radar_motion import mounting,design,fit_velocity,select_frames",
        "from radar_motion import mounting,design,fit_velocity,select_frames\nfrom fusion_rules import fusion_reason")
    old='''                    reason="existing_context_policy" if policy.startswith("existing") else (
                        "candidate_"+e["checked"]["candidate_status"] if not e["checked"]["candidate_ok"] else
                        "radar_"+status if not radar_ok else "radar_agreement" if agree else "radar_disagreement")'''
    new='''                    reason=fusion_reason(policy,e["checked"]["candidate_ok"],e["checked"]["candidate_status"],
                        radar_ok,status,agree,old_flags[(e["event_id"],"context_no_ood")])'''
    assert text.count(old)==1
    path.write_text(text.replace(old,new))
    print("Initial run preserved; explanation labels corrected; acceptance rules unchanged")
if __name__=="__main__": main()
