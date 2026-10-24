"""Frozen v2 development ablation and signed coordinated attack stress test."""
import hashlib,json,os,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from radar_motion import design,mounting,select_frames
from radar_balanced import fit_velocity
from attack_values import craft
ROOT=Path(__file__).resolve().parent
CAN=ROOT.parent/"vehicle_context_policy";sys.path.insert(0,str(CAN))
import context_core as core
from run_experiment import make_body,metrics
def rows(path):
    with path.open() as f: return [json.loads(l) for l in f]
def main():
    protocol=json.loads((ROOT/"protocol_v2.json").read_text())
    ph=hashlib.sha256((ROOT/"protocol_v2.json").read_bytes()).hexdigest()
    assert ph==json.loads((ROOT/"protocol_v2_freeze.json").read_text())["sha256"]
    old=json.loads((ROOT/"radar_experiment_summary.json").read_text());oldrun=ROOT/old["runtime_relative_path"]
    can=json.loads((CAN/"experiment_summary.json").read_text());canrun=CAN/can["runtime_relative_path"]
    nominal=[e for e in rows(canrun/"packet_events.jsonl") if e["case"]=="nominal"]
    evaluation=[r for r in json.loads((canrun/"online_inputs.json").read_text()) if r["role"]=="evaluation"]
    models=json.loads((canrun/"fitted_model.json").read_text())
    lookup={(e["recording"],e["index"]):e for e in nominal}
    runtime=ROOT/"runtime"/("balanced_"+os.urandom(8).hex());runtime.mkdir()
    keys=[os.urandom(32) for _ in range(3)]
    (runtime/"research_keys.json").write_text(json.dumps([k.hex() for k in keys]))
    cache={("point_count_v1",r["recording"],r["index"],r["track"]):r for r in rows(oldrun/"radar_estimates.jsonl")}
    for name,expected in old["source_frame_hashes"].items():
        folder=ROOT/"runtime"/name;source=folder/"frames.jsonl"
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        frames=rows(source);cal=mounting(json.loads((folder/"transforms.json").read_text()))
        for row in [r for r in evaluation if r["recording"]==name]:
            for track in protocol["timing_tracks"]:
                chosen=select_frames(frames,row["now_us"]*1000,track)
                if chosen:
                    a,b,s=design(chosen,cal);fit=fit_velocity(a,b,s)
                else: fit=dict(accepted=False,reason="no_fresh_complete_radar_scans",state=None,inlier_indices=[],detections=0)
                field="raw_header_ns" if track=="raw_header_association" else "corrected_header_ns"
                cache["sensor_balanced_v2",name,row["index"],track]=dict(recording=name,index=row["index"],
                    track=track,now_us=row["now_us"],estimate=fit,
                    measured_us=min(f[field] for f in chosen)//1000 if chosen else None,
                    frame_keys=[[f["sensor"],f["raw_header_ns"],f["payload_sha256"]] for f in chosen])
        print(name+": frozen balanced estimates complete",flush=True)
    with (runtime/"estimates.jsonl").open("w") as out:
        for (estimator,_,_,_),item in cache.items(): out.write(json.dumps(dict(estimator=estimator,**item),allow_nan=False)+"\n")
    cells=defaultdict(list);count=0;crafted=Counter();quality={}
    with (runtime/"events.jsonl").open("w") as audit,(runtime/"decisions.jsonl").open("w") as decisions:
        for estimator in ("point_count_v1","sensor_balanced_v2"):
            quality[estimator]={}
            for track in protocol["timing_tracks"]:
                vals=[v for (est,n,i,t),v in cache.items() if est==estimator and t==track]
                errors=[abs(v["estimate"]["speed_mps"]-lookup[v["recording"],v["index"]]["reference"])
                    for v in vals if v["estimate"]["accepted"] and lookup[v["recording"],v["index"]]["reference"] is not None]
                quality[estimator][track]=dict(requests=len(vals),accepted=sum(v["estimate"]["accepted"] for v in vals),
                    reasons=dict(Counter(v["estimate"]["reason"] for v in vals)),scored=len(errors),
                    p95_discrepancy_mps=float(np.quantile(errors,.95)) if errors else None,
                    maximum_discrepancy_mps=max(errors) if errors else None,above_2_mps=sum(e>2 for e in errors))
                for name in old["source_frame_hashes"]:
                    group=[r for r in evaluation if r["recording"]==name];onset=group[0]["now_us"]+1000000
                    for case in protocol["attacks"]:
                        capture=hashlib.sha256((estimator+"|"+track+"|"+case).encode("ascii")).hexdigest()[:32]
                        ctx=core.auth.Context("dev_vehicle",name,capture)
                        state=runtime/(name+"_"+capture+".sqlite");core.auth.create_state(state,ctx)
                        guard=core.auth.Guard(state,ctx,[core.auth.KeyRule("candidate","kc","speed",keys[0]),
                            core.auth.KeyRule("peer","ka","motion_context",keys[1]),core.auth.KeyRule("radar","kr","radar_speed",keys[2])])
                        for row in group:
                            active=case!="nominal" and row["now_us"]>=onset
                            item=cache[estimator,name,row["index"],track];fit=item["estimate"]
                            rv=round(fit["speed_mps"]*1000)/1000 if fit["accepted"] else None
                            c,f=craft(case if active else "nominal",row["candidate"],row["features"],rv)
                            cw=aw=rw=None;radar_ok=False
                            if c is not None:
                                cw=json.dumps(core.auth.sign(make_body(ctx,"candidate","kc","speed",row["index"],row["measured_us"],c),keys[0]))
                            if f is not None:
                                aw=json.dumps(core.bind_context(make_body(ctx,"peer","ka","motion_context",row["index"],row["measured_us"],f["peer"]),f,keys[1]))
                            checked=core.authenticate(guard,cw,aw,row["now_us"])
                            if rv is not None:
                                envelope=core.auth.sign(make_body(ctx,"radar","kr","radar_speed",row["index"],item["measured_us"],rv),keys[2])
                                rw=json.dumps(envelope);rcheck=guard.verify(envelope,row["now_us"],row["now_us"])
                                radar_ok=rcheck["authenticated_report"]
                            available=bool(checked["candidate_ok"] and checked["context_ok"])
                            learned=core.assess(models,checked["candidate"],checked["features"],False) if available else dict(accept=False,reason="CAN_authentication_or_availability")
                            full=core.assess(models,checked["candidate"],checked["features"],True) if available else dict(accept=False)
                            agree=bool(checked["candidate_ok"] and radar_ok and abs(round(checked["candidate"]*1000)-round(rv*1000))<=1000)
                            flags=dict(context_learned=full["accept"],context_no_ood=learned["accept"],
                                radar_agreement=agree,context_no_ood_plus_radar=agree and learned["accept"])
                            event_id=f"{estimator}:{track}:{name}:{case}:{row['index']}"
                            crafted_flag=c is not None and f is not None
                            if active and crafted_flag: crafted[estimator+"|"+track+"|"+case]+=1
                            audit.write(json.dumps(dict(event_id=event_id,estimator=estimator,track=track,recording=name,
                                case=case,index=row["index"],now_us=row["now_us"],active=active,crafted=crafted_flag,
                                candidate_wire=cw,context_wire=aw,radar_wire=rw,checked=checked,
                                radar_ok=radar_ok,radar_speed=rv,learned=learned,
                                reference=lookup[name,row["index"]]["reference"]),allow_nan=False)+"\n")
                            for policy,accept in flags.items():
                                reason=("accepted" if accept else "candidate_authentication" if not checked["candidate_ok"]
                                    else "radar_unavailable" if policy in ("radar_agreement","context_no_ood_plus_radar") and not radar_ok
                                    else "radar_disagreement" if policy in ("radar_agreement","context_no_ood_plus_radar") and not agree
                                    else "context_rejection")
                                out=dict(event_id=event_id,estimator=estimator,track=track,recording=name,case=case,policy=policy,
                                    active=active,crafted=crafted_flag,accept=bool(accept),value=checked["candidate"] if accept else None,
                                    reference=lookup[name,row["index"]]["reference"],reason=reason,physical_speed_authorized=False)
                                decisions.write(json.dumps(out,allow_nan=False)+"\n");cells[estimator,track,case,policy].append(out);count+=1
        print("Signed attack evaluation complete",flush=True)
    policies=("context_learned","context_no_ood","radar_agreement","context_no_ood_plus_radar")
    results={est:{track:{case:{p:metrics(cells[est,track,case,p]) for p in policies} for case in protocol["attacks"]}
        for track in protocol["timing_tracks"]} for est in quality}
    report=dict(protocol_sha256=ph,runtime_relative_path=str(runtime.relative_to(ROOT)),source_frame_hashes=old["source_frame_hashes"],
        baseline_runtime=old["runtime_relative_path"],can_runtime=can["runtime_relative_path"],quality=quality,
        results=results,crafted_active_counts=dict(crafted),policy_decisions=count,development_only=True,
        physical_witness_enabled=False,promotion_allowed=False,production_policy_changed=False,
        acquisition_latency_certified=False,thresholds_retuned_after_evaluation=False)
    (ROOT/"balanced_experiment_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (runtime/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(quality=quality,raw_combined={est:{case:results[est]["raw_header_association"][case]["context_no_ood_plus_radar"] for case in protocol["attacks"]} for est in quality}),indent=2))
if __name__=="__main__": main()
