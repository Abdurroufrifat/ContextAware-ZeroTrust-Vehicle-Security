"""Independent radar cross-check, with separately reported transport/corrected timing tracks."""
import hashlib,json,os,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from radar_motion import mounting,design,fit_velocity,select_frames
from fusion_rules import fusion_reason
ROOT=Path(__file__).resolve().parent
CAN=ROOT.parent/"vehicle_context_policy"
sys.path.insert(0,str(CAN))
import context_core as context
from run_experiment import metrics
def main():
    protocol=json.loads((ROOT/"protocol_v1.json").read_text())
    sha=hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()
    assert sha==json.loads((ROOT/"protocol_freeze.json").read_text())["sha256"]
    previous=json.loads((CAN/"experiment_summary.json").read_text())
    old_runtime=CAN/previous["runtime_relative_path"]
    online=json.loads((old_runtime/"online_inputs.json").read_text())
    evaluation=[r for r in online if r["role"]=="evaluation"]
    with (old_runtime/"packet_events.jsonl").open() as inp: events=[json.loads(line) for line in inp]
    old_flags={}
    with (old_runtime/"decisions.jsonl").open() as inp:
        for line in inp:
            r=json.loads(line)
            if r["policy"] in ("context_learned","context_no_ood"): old_flags[(r["event_id"],r["policy"])]=r["accept"]
    runtime=ROOT/"runtime"/("run_"+os.urandom(8).hex());runtime.mkdir()
    key=os.urandom(32)
    (runtime/"research_radar_key.json").write_text(json.dumps(key.hex()))
    cache={};source_hashes={};extract_summaries={}
    for name in protocol["records"]:
        folder=ROOT/"runtime"/name
        frames_path=folder/"frames.jsonl"
        source_hashes[name]=hashlib.sha256(frames_path.read_bytes()).hexdigest()
        with frames_path.open() as inp: frames=[json.loads(line) for line in inp]
        calibration=mounting(json.loads((folder/"transforms.json").read_text()))
        extract_summaries[name]=json.loads((folder/"extraction_summary.json").read_text())
        assert extract_summaries[name]["raw_frames"]==extract_summaries[name]["exact_value_matches"]
        for row in [r for r in evaluation if r["recording"]==name]:
            for track in protocol["timing_tracks"]:
                selected=select_frames(frames,row["now_us"]*1000,track)
                if selected:
                    a,b,sensors=design(selected,calibration)
                    fitted=fit_velocity(a,b,sensors)
                else: fitted=dict(accepted=False,reason="no_fresh_complete_radar_scans",detections=0,state=None,inlier_indices=[])
                field="raw_header_ns" if track=="raw_header_association" else "corrected_header_ns"
                estimate=dict(recording=name,index=row["index"],track=track,now_us=row["now_us"],
                    estimate=fitted,frame_keys=[[f["sensor"],f["raw_header_ns"],f["payload_sha256"]] for f in selected],
                    measured_us=min(f[field] for f in selected)//1000 if selected else None)
                cache[(name,row["index"],track)]=estimate
        print(name+": motion estimates complete",flush=True)
    with (runtime/"radar_estimates.jsonl").open("w") as out:
        for v in cache.values(): out.write(json.dumps(v,allow_nan=False)+"\n")
    guards={};cells=defaultdict(list);record_cells=defaultdict(list);decision_count=0
    with (runtime/"radar_packets.jsonl").open("w") as packets,(runtime/"fusion_decisions.jsonl").open("w") as audit:
        for e in events:
            for track in protocol["timing_tracks"]:
                item=cache[(e["recording"],e["index"],track)]
                fitted=item["estimate"];radar_ok=False;radar_speed=None;wire=None
                status=fitted["reason"]
                if fitted["accepted"]:
                    scope=(track,e["recording"],e["case"])
                    if scope not in guards:
                        ctx=context.auth.Context("dev_vehicle",e["recording"],e["case"])
                        state=runtime/("_".join(scope)+".sqlite")
                        context.auth.create_state(state,ctx)
                        guards[scope]=context.auth.Guard(state,ctx,[context.auth.KeyRule("radar","kr","radar_speed",key)])
                    ctx=guards[scope].context
                    body=dict(schema=1,source_id="radar",key_id="kr",vehicle_id=ctx.vehicle_id,
                        session_id=ctx.session_id,capture_id=ctx.capture_id,channel="radar_speed",
                        seq=e["index"],measured_us=item["measured_us"],speed_mmps=round(fitted["speed_mps"]*1000))
                    envelope=context.auth.sign(body,key)
                    wire=json.dumps(envelope,separators=(",",":"))
                    checked=guards[scope].verify(envelope,e["now_us"],e["now_us"])
                    status=checked["status"];radar_ok=checked["authenticated_report"]
                    if radar_ok: radar_speed=body["speed_mmps"]/1000
                c=e["checked"]["candidate"]
                agree=bool(e["checked"]["candidate_ok"] and radar_ok and abs(round(c*1000)-round(radar_speed*1000))<=1000)
                flags={"existing_context_learned":old_flags[(e["event_id"],"context_learned")],
                    "existing_context_no_ood":old_flags[(e["event_id"],"context_no_ood")],
                    "radar_agreement":agree,
                    "context_no_ood_plus_radar":agree and old_flags[(e["event_id"],"context_no_ood")]}
                packets.write(json.dumps(dict(event_id=e["event_id"],track=track,wire=wire,
                    radar_status=status,radar_ok=radar_ok,radar_speed=radar_speed),allow_nan=False)+"\n")
                for policy,accept in flags.items():
                    reason=fusion_reason(policy,e["checked"]["candidate_ok"],e["checked"]["candidate_status"],
                        radar_ok,status,agree,old_flags[(e["event_id"],"context_no_ood")])
                    row=dict(event_id=e["event_id"],recording=e["recording"],case=e["case"],track=track,
                        policy=policy,active=e["active"],accept=accept,value=c if accept else None,
                        reference=e["reference"],reason=reason,physical_speed_authorized=False)
                    cells[(track,e["case"],policy)].append(row)
                    record_cells[(e["recording"],track,e["case"],policy)].append(row)
                    audit.write(json.dumps(row,allow_nan=False)+"\n");decision_count+=1
    results={t:{case:{p:metrics(cells[(t,case,p)]) for p in protocol["policy_variants"]} for case in previous["results"]} for t in protocol["timing_tracks"]}
    per_record={name:{t:{case:{p:metrics(record_cells[(name,t,case,p)]) for p in protocol["policy_variants"]} for case in previous["results"]} for t in protocol["timing_tracks"]} for name in protocol["records"]}
    quality={}
    for track in protocol["timing_tracks"]:
        values=[r for r in cache.values() if r["track"]==track]
        quality[track]=dict(queries=len(values),accepted_motion_estimates=sum(v["estimate"]["accepted"] for v in values),
            reasons=dict(Counter(v["estimate"]["reason"] for v in values)))
    report=dict(protocol_sha256=sha,runtime_relative_path=str(runtime.relative_to(ROOT)),source_frame_hashes=source_hashes,
        decoded_records=extract_summaries,radar_quality=quality,results=results,per_record=per_record,
        evaluation_queries=len(evaluation),policy_decisions=decision_count,source_domain="radar independent of CAN content",
        acquisition_latency_certified=False,raw_track_has_transport_freshness_only=True,
        corrected_timing_must_not_be_silently_discarded=True,physical_witness_enabled=False,
        production_policy_changed=False,promotion_allowed=False,development_only=True,
        assumptions=["Planar rigid-body motion and majority static radar scene",
            "Published extrinsic transforms are used without fitted correction",
            "Decoder/host integrity and unmodified radar are assumed in CAN-compromise cases",
            "Research radar signing does not authenticate original acquisition",
            "All three recordings are already-inspected development data from one day"])
    (ROOT/"radar_experiment_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(quality=quality,decisions=decision_count,
        nominal={t:results[t]["nominal"]["context_no_ood_plus_radar"] for t in results},
        coordinated={t:results[t]["coordinated_bias"]["context_no_ood_plus_radar"] for t in results}),indent=2),flush=True)
if __name__=="__main__": main()
