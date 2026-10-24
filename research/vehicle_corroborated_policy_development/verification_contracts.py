"""Independent artifact identity and prescribed-exposure checks."""
from pathlib import Path
import json,hashlib,copy

def expected_public_run(config,seed,pilot):
    if seed in config["seeds"]:
        return config["public_input_policy"]["stress_runs"][config["seeds"].index(seed)%2]
    if pilot:return "heldout_medium"
    raise ValueError("unregistered confirmation seed")

def expected_delivery(issued,case,active):
    result=copy.deepcopy(issued)
    if active and case=="actuator_tamper":
        obj=json.loads(result["wire"]);obj["body"]["accel_mmps2"]=2000
        result["wire"]=json.dumps(obj,separators=(",",":"))
    return result

def verify_checkpoint(base,summary,digest):
    base=Path(base).resolve()
    trace=(base/summary["trace"]).resolve();trajectory=(base/summary["trajectory"]).resolve()
    root=trace.parent
    assert root.is_relative_to(base) and trajectory.parent==root
    assert trace.name=="trace.jsonl.gz" and trajectory.name=="trajectory.jsonl.gz"
    marker=json.loads((root/"complete.json").read_text())
    assert marker["trial_id"]==summary["trial_id"] and marker["lock_digest"]==digest
    assert set(marker["files"])=={"summary.json","trace.jsonl.gz","trajectory.jsonl.gz"}
    for n,h in marker["files"].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
    assert json.loads((root/"summary.json").read_text())==summary
    completed=list(root.parent.glob("attempt_*/complete.json"))
    assert len(completed)==1 and completed[0].parent==root

def verify_refinement_metadata(out,pilot):
    assert type(out["ablation_refinement_available"]) is bool
    eligible=out["ablation_refinement_available"]
    if eligible:
        if not pilot:assert "refinement_ablation_source_decision" in out
        source=out.get("refinement_ablation_source_decision")
        if source is not None:
            assert source["refinement_status"]=="available"
            assert source["scene_decision"]==out["unrefined_original_decision"]
        assert out["estimator_decision"]["selection"]=="scene_midpoint_ablation"
        return source or out["estimator_decision"]
    assert "refinement_ablation_source_decision" not in out and "unrefined_original_decision" not in out
    assert out["estimator_decision"].get("refinement_status")!="available"
    return out["estimator_decision"]

def expected_reason(method,speed,scene,radar,radar_status,context,margin,command,desired,eligible=False):
    baseline=method in ("s3e_brake","ssf_zocbf")
    wrapper=baseline or method=="no_fusion" or (method=="no_refinement" and eligible)
    if baseline and (speed is None or scene is None or radar is None):return "plausible_state_unavailable"
    if speed is None or scene is None:return "observation_unavailable"
    if radar is None and method!="no_fusion":
        return "observation_unavailable" if wrapper else "radar_"+radar_status
    if context is None:return "context_disagreement"
    if method=="no_brake":return "control_filter_removed"
    if margin<0:return "reserve_unrecoverable"
    if command!=desired:return "filter_limited" if wrapper else "reserve_limited"
    return "observed_context_accepted"

def accepted_history(ledger,lower,now):
    if lower is None:return []
    rows=[list(r)+[1] for r in ledger if r[1]>lower and r[0]<now]
    if not rows or len(rows)>16 or rows[0][0]>lower or rows[-1][1]!=now:return []
    if any(a[1]!=b[0] for a,b in zip(rows,rows[1:])):return []
    return rows

def verify_command_history(packet,key,ctx,seq,now,commands):
    import hmac
    fields=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","issued_us","commands")
    obj=json.loads(packet["wire"]);body=obj["body"]
    assert set(body)==set(fields)
    data=b"COACTCPS-COMMAND-HMAC-SHA256-v1\0"+json.dumps([body[k] for k in fields],ensure_ascii=True,separators=(",",":")).encode("ascii")
    assert hmac.compare_digest(obj["tag"],hmac.new(key,data,hashlib.sha256).hexdigest())
    assert body=={"schema":1,"source_id":"command_logger","key_id":"h-key",**ctx,"channel":"command_history","seq":seq,"issued_us":now,"commands":commands}
    assert packet["received_us"]==now
