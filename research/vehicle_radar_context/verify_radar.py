"""Independent geometry, timing, signature and fusion-decision audit."""
import hashlib,hmac,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
ORDER=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","speed_mmps")
def geometry(transforms):
    nodes={t["child_frame_id"].lstrip("/"):t for t in transforms}
    cache={"base_link":(np.eye(3),np.zeros(3))}
    def pose(name):
        if name not in cache:
            t=nodes[name];q=t["transform"]["rotation"]
            local=Rotation.from_quat([q[k] for k in ("x","y","z","w")]).as_matrix()
            xyz=np.array([t["transform"]["translation"][k] for k in ("x","y","z")])
            parent_r,parent_p=pose(t["header"]["frame_id"].lstrip("/"))
            cache[name]=(parent_r@local,parent_p+parent_r@xyz)
        return cache[name]
    rr,rp=pose("vehicle/rear_axis")
    output={}
    for name in nodes:
        if name.startswith("sensor/radar/umrr/") and name.rsplit("/",1)[1].isdigit():
            sr,sp=pose(name)
            output[int(name.rsplit("/",1)[1])]=(rr.T@sr,rr.T@(sp-rp))
    return output
def main():
    report=json.loads((ROOT/"radar_experiment_summary.json").read_text())
    runtime=ROOT/report["runtime_relative_path"]
    spec=json.loads((ROOT/"protocol_v1.json").read_text())
    assert hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()==report["protocol_sha256"]
    source={};calibration={};matched=0
    for name in spec["records"]:
        folder=ROOT/"runtime"/name
        assert hashlib.sha256((folder/"frames.jsonl").read_bytes()).hexdigest()==report["source_frame_hashes"][name]
        with (folder/"frames.jsonl").open() as inp: source[name]=[json.loads(line) for line in inp]
        assert all(f["corrected_header_ns"]==f["raw_header_ns"]-106_000_000 for f in source[name])
        assert len(source[name])==report["decoded_records"][name]["exact_value_matches"]
        matched+=len(source[name])
        calibration[name]=geometry(json.loads((folder/"transforms.json").read_text()))
    estimates={};quality_checks=0;clock_checks=0
    with (runtime/"radar_estimates.jsonl").open() as inp:
        for line in inp:
            e=json.loads(line);now=e["now_us"]*1000
            field="raw_header_ns" if e["track"]=="raw_header_association" else "corrected_header_ns"
            eligible=[f for f in source[e["recording"]] if f["available_ns"]<=now and 0<=now-f[field]<=100_000_000]
            latest={}
            for f in eligible:
                if f["sensor"] not in latest or f["raw_header_ns"]>latest[f["sensor"]]["raw_header_ns"]: latest[f["sensor"]]=f
            frames=[latest[k] for k in sorted(latest)]
            assert e["frame_keys"]==[[f["sensor"],f["raw_header_ns"],f["payload_sha256"]] for f in frames]
            assert e["measured_us"]==(min(f[field] for f in frames)//1000 if frames else None)
            clock_checks+=1
            result=e["estimate"]
            if result["accepted"]:
                rows=[];values=[];sensors=[]
                for f in frames:
                    rotation,position=calibration[e["recording"]][f["sensor"]]
                    angular_basis=np.cross(np.array([0.,0.,1.]),position)
                    for target in f["targets"]:
                        distance,radial,azimuth,elevation=target[:4]
                        if distance<=0: continue
                        local=np.array([math.cos(elevation)*math.cos(azimuth),math.cos(elevation)*math.sin(azimuth),math.sin(elevation)])
                        ray=rotation@local
                        rows.append([ray[0],ray[1],float(ray@angular_basis)])
                        values.append(-radial);sensors.append(f["sensor"])
                matrix=np.array(rows);values=np.array(values);sensors=np.array(sensors)
                state=np.array(result["state"]);residual=np.abs(matrix@state-values)
                mask=residual<=.25
                assert np.flatnonzero(mask).tolist()==result["inlier_indices"]
                assert len(values)>=30 and mask.sum()>=math.ceil(.5*len(values))
                assert len(set(sensors[mask].tolist()))>=3 and np.linalg.cond(matrix[mask])<=1000
                assert math.hypot(*state[:2])<=60 and abs(state[2])<=2
                assert abs(math.hypot(*state[:2])-result["speed_mps"])<1e-12
                quality_checks+=1
            estimates[(e["recording"],e["index"],e["track"])]=e
    can=ROOT.parent/"vehicle_context_policy"
    prior=json.loads((can/"experiment_summary.json").read_text())
    old_runtime=can/prior["runtime_relative_path"]
    with (old_runtime/"packet_events.jsonl").open() as inp: events={e["event_id"]:e for e in map(json.loads,inp)}
    flags={}
    with (old_runtime/"decisions.jsonl").open() as inp:
        for line in inp:
            d=json.loads(line)
            if d["policy"] in ("context_learned","context_no_ood"): flags[(d["event_id"],d["policy"])]=d["accept"]
    key=bytes.fromhex(json.loads((runtime/"research_radar_key.json").read_text()))
    radar={};sequences={};packet_count=0
    with (runtime/"radar_packets.jsonl").open() as inp:
        for line in inp:
            p=json.loads(line);event=events[p["event_id"]]
            item=estimates[(event["recording"],event["index"],p["track"])]
            fitted=item["estimate"]
            if p["wire"] is not None:
                envelope=json.loads(p["wire"]);body=envelope["body"]
                canonical=b"COACTCPS-EVIDENCE-HMAC-SHA256-v1\x00"+json.dumps([body[k] for k in ORDER],ensure_ascii=True,separators=(",",":")).encode("ascii")
                assert hmac.compare_digest(hmac.new(key,canonical,hashlib.sha256).hexdigest(),envelope["tag"])
                assert body["source_id"]=="radar" and body["channel"]=="radar_speed"
                assert body["session_id"]==event["recording"] and body["capture_id"]==event["case"]
                assert body["measured_us"]==item["measured_us"]
                assert 0<=event["now_us"]-body["measured_us"]<=100000
                scope=(event["recording"],event["case"],p["track"])
                assert body["seq"]>sequences.get(scope,-1);sequences[scope]=body["seq"]
                assert body["speed_mmps"]==round(fitted["speed_mps"]*1000)
                assert p["radar_ok"] and fitted["accepted"] and p["radar_status"]=="authenticated"
                assert p["radar_speed"]==body["speed_mmps"]/1000
                packet_count+=1
            else: assert not p["radar_ok"] and not fitted["accepted"]
            radar[(p["event_id"],p["track"])]=p
    groups=defaultdict(list);seen=set();count=0
    with (runtime/"fusion_decisions.jsonl").open() as inp:
        for line in inp:
            d=json.loads(line);unique=(d["event_id"],d["track"],d["policy"])
            assert unique not in seen;seen.add(unique)
            event=events[d["event_id"]];p=radar[(d["event_id"],d["track"])]
            agree=bool(event["checked"]["candidate_ok"] and p["radar_ok"] and
                abs(round(event["checked"]["candidate"]*1000)-round(p["radar_speed"]*1000))<=1000)
            expected={"existing_context_learned":flags[(d["event_id"],"context_learned")],
                "existing_context_no_ood":flags[(d["event_id"],"context_no_ood")],"radar_agreement":agree,
                "context_no_ood_plus_radar":agree and flags[(d["event_id"],"context_no_ood")]}[d["policy"]]
            assert d["accept"]==expected and not d["physical_speed_authorized"]
            assert d["reference"]==event["reference"] and d["active"]==event["active"]
            assert d["value"]==(event["checked"]["candidate"] if expected else None)
            if d["policy"]=="context_no_ood_plus_radar" and agree and not expected:
                assert d["reason"]=="learned_context_rejection"
            groups[(d["track"],d["case"],d["policy"])].append(d);count+=1
    for (track,case,policy),rows in groups.items():
        metrics=report["results"][track][case][policy]
        accepted=[d for d in rows if d["accept"]]
        errors=[abs(d["value"]-d["reference"]) for d in accepted if d["reference"] is not None]
        assert len(rows)==metrics["requests"] and len(accepted)==metrics["accepted"]
        assert sum(v>2 for v in errors)==metrics["accepted_discrepancy_above_2_mps"]
        assert metrics["unscorable_accepted"]==len(accepted)-len(errors)
        p95=float(np.quantile(errors,.95)) if errors else None
        assert (p95 is None and metrics["p95_discrepancy_mps"] is None) or abs(p95-metrics["p95_discrepancy_mps"])<1e-12
    assert count==report["policy_decisions"]==len(events)*8
    original=json.loads((ROOT/"initial_v1_snapshot"/"radar_experiment_summary.json").read_text())
    for track in report["results"]:
        for case,policies in report["results"][track].items():
            for policy,current in policies.items():
                assert {k:v for k,v in current.items() if k!="reasons"}=={k:v for k,v in original["results"][track][case][policy].items() if k!="reasons"}
    status=dict(all_passed=True,matched_raw_frames=matched,independent_clock_selection_checks=clock_checks,
        accepted_motion_geometry_checks=quality_checks,signed_radar_packets=packet_count,
        fusion_decisions=count,original_acceptance_metrics_preserved=True,physical_promotion_allowed=False)
    (ROOT/"independent_verification.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))
if __name__=="__main__": main()
