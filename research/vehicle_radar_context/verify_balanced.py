"""Independent timing/geometry/HMAC/attack/model/decision/metric audit; no estimator imports."""
import hashlib,hmac,json,math
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import numpy as np
from verify_radar import geometry,ORDER
ROOT=Path(__file__).resolve().parent
def rows(path):
    with path.open() as f: return [json.loads(l) for l in f]
def canonical(body):
    return b"COACTCPS-EVIDENCE-HMAC-SHA256-v1\x00"+json.dumps([body[k] for k in ORDER],ensure_ascii=True,separators=(",",":")).encode("ascii")
def main():
    report=json.loads((ROOT/"balanced_experiment_summary.json").read_text());runtime=ROOT/report["runtime_relative_path"]
    protocol=json.loads((ROOT/"protocol_v2.json").read_text())
    assert hashlib.sha256((ROOT/"protocol_v2.json").read_bytes()).hexdigest()==report["protocol_sha256"]
    assert json.loads((ROOT/"protocol_v2_freeze.json").read_text())["sha256"]==report["protocol_sha256"]
    assert not report["promotion_allowed"] and not report["production_policy_changed"]
    sources={};poses={}
    for name,sha in report["source_frame_hashes"].items():
        folder=ROOT/"runtime"/name;p=folder/"frames.jsonl"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
        sources[name]=rows(p);poses[name]=geometry(json.loads((folder/"transforms.json").read_text()))
        assert all(f["raw_header_ns"]-f["corrected_header_ns"]==106000000 for f in sources[name])
    old=ROOT/report["baseline_runtime"]
    baseline={(r["recording"],r["index"],r["track"]):r for r in rows(old/"radar_estimates.jsonl")}
    estimates={};clock_checks=geometry_checks=0
    for e in rows(runtime/"estimates.jsonl"):
        name=e["recording"];now=e["now_us"]*1000
        field="raw_header_ns" if e["track"]=="raw_header_association" else "corrected_header_ns"
        chosen={}
        for f in sources[name]:
            if f["available_ns"]<=now and 0<=now-f[field]<=100000000:
                if f["sensor"] not in chosen or f["raw_header_ns"]>chosen[f["sensor"]]["raw_header_ns"]: chosen[f["sensor"]]=f
        frames=[chosen[k] for k in sorted(chosen)]
        assert e["frame_keys"]==[[f["sensor"],f["raw_header_ns"],f["payload_sha256"]] for f in frames]
        assert e["measured_us"]==(min(f[field] for f in frames)//1000 if frames else None)
        if e["estimator"]=="point_count_v1":
            assert {k:v for k,v in e.items() if k!="estimator"}==baseline[name,e["index"],e["track"]]
        fitted=e["estimate"]
        if fitted["accepted"]:
            aa=[];bb=[];ss=[]
            for f in frames:
                rotation,position=poses[name][f["sensor"]]
                angular=np.cross([0.,0.,1.],position)
                for target in f["targets"]:
                    r,radial,az,el=target[:4]
                    if r<=0: continue
                    ray=rotation@np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
                    aa.append([ray[0],ray[1],float(ray@angular)]);bb.append(-radial);ss.append(f["sensor"])
            aa=np.array(aa);bb=np.array(bb);ss=np.array(ss);state=np.array(fitted["state"])
            residual=np.abs(aa@state-bb);mask=residual<=.25
            assert len(bb)>=30 and len(set(ss[mask]))>=3 and np.flatnonzero(mask).tolist()==fitted["inlier_indices"]
            weights=np.array([1/ss.tolist().count(s) for s in ss])
            if e["estimator"]=="sensor_balanced_v2":
                fraction=sum(weights[mask])/sum(weights)
                assert fraction>=.5 and abs(fraction-fitted["balanced_inlier_fraction"])<1e-12
                condition=np.linalg.cond((aa*np.sqrt(weights)[:,None])[mask])
            else:
                assert mask.sum()>=math.ceil(.5*len(bb));condition=np.linalg.cond(aa[mask])
            assert condition<=1000 and abs(condition-fitted["matrix_condition"])<1e-8
            assert math.hypot(*state[:2])<=60 and abs(state[2])<=2
            assert abs(math.hypot(*state[:2])-fitted["speed_mps"])<1e-12
            geometry_checks+=1
        clock_checks+=1;estimates[e["estimator"],name,e["index"],e["track"]]=e
    can=ROOT.parent/"vehicle_context_policy"/report["can_runtime"]
    nominal={(e["recording"],e["index"]):e for e in rows(can/"packet_events.jsonl") if e["case"]=="nominal"}
    online={(r["recording"],r["index"]):r for r in json.loads((can/"online_inputs.json").read_text()) if r["role"]=="evaluation"}
    model=json.loads((can/"fitted_model.json").read_text())
    onsets={name:min(r["now_us"] for r in online.values() if r["recording"]==name)+1000000 for name in sources}
    keys=[bytes.fromhex(k) for k in json.loads((runtime/"research_keys.json").read_text())]
    events={};flags={};sequences={};packet_checks=0;crafted=Counter()
    for e in rows(runtime/"events.jsonl"):
        est,name,index,track=e["estimator"],e["recording"],e["index"],e["track"]
        row=online[name,index];item=estimates[est,name,index,track];fitted=item["estimate"]
        active=e["case"]!="nominal" and row["now_us"]>=onsets[name];assert e["active"]==active
        case=e["case"] if active else "nominal";c=row["candidate"];f=dict(row["features"])
        rv=round(fitted["speed_mps"]*1000)/1000 if fitted["accepted"] else None
        if case=="coordinated_plus_3": c+=3;f["peer"]+=3
        elif case=="coordinated_minus_3_clipped": c=max(0,c-3);f["peer"]=max(0,f["peer"]-3)
        elif case=="coordinated_zero": c=0;f["peer"]=0
        elif case=="coordinated_times_2": c*=2;f["peer"]*=2
        elif case=="coordinated_radar_match":
            if rv is None: c=None;f=None
            else: c=rv;f["peer"]=rv
        else: assert case=="nominal"
        if c is not None: c=round(c*1000)/1000;f["peer"]=round(f["peer"]*1000)/1000
        assert e["crafted"]==(c is not None and f is not None)
        if active and e["crafted"]: crafted[est+"|"+track+"|"+e["case"]]+=1
        capture=hashlib.sha256((est+"|"+track+"|"+e["case"]).encode("ascii")).hexdigest()[:32]
        for role,keyid,channel,wire,key,value,measured in [
            ("candidate","kc","speed",e["candidate_wire"],keys[0],c,row["measured_us"]),
            ("peer","ka","motion_context",e["context_wire"],keys[1],f["peer"] if f else None,row["measured_us"]),
            ("radar","kr","radar_speed",e["radar_wire"],keys[2],rv,item["measured_us"])]:
            if value is None: assert wire is None;continue
            envelope=json.loads(wire)
            if role=="peer":
                assert envelope["features"]==f
                body=envelope["report"]["body"]
                binding=b"COACT-CONTEXT-BINDING-v1\x00"+canonical(body)+json.dumps(f,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
                assert hmac.compare_digest(hmac.new(key,binding,hashlib.sha256).hexdigest(),envelope["context_tag"])
                envelope=envelope["report"]
            body=envelope["body"]
            expected=dict(schema=1,source_id=role,key_id=keyid,vehicle_id="dev_vehicle",session_id=name,
                capture_id=capture,channel=channel,seq=index,measured_us=measured,speed_mmps=round(value*1000))
            assert body==expected and 0<=e["now_us"]-measured<=100000
            assert hmac.compare_digest(hmac.new(key,canonical(body),hashlib.sha256).hexdigest(),envelope["tag"])
            scope=(est,track,name,e["case"],role);assert index>sequences.get(scope,-1);sequences[scope]=index
            packet_checks+=1
        checked=e["checked"];assert checked["candidate_ok"]==(c is not None) and checked["context_ok"]==(f is not None)
        assert checked["candidate"]==c and checked["features"]==f and e["radar_ok"]==(rv is not None) and e["radar_speed"]==rv
        no_ood=full=False
        if c is not None and f is not None:
            p=f["peer"];y=abs(f["yaw"]);s=abs(f["steering"]);a=abs(f["acceleration"])
            basis=np.array([1,p,y,s,a,p*s*s,p*y,p*p])
            predicted=float(np.sum(((basis-np.array(model["mean"]))/np.array(model["scale"]))*np.array(model["coef"])))
            no_ood=abs(c-p-predicted)<=model["threshold"]
            raw=np.array([p,y,s,a]);margins=np.array([.5,.05,.1,1.])
            ood=bool(np.any(raw<np.array(model["raw_min"])-margins) or np.any(raw>np.array(model["raw_max"])+margins))
            full=no_ood and not ood
            assert abs(predicted-e["learned"]["prediction"])<1e-12
        assert e["learned"]["accept"]==no_ood
        agree=bool(c is not None and rv is not None and abs(round(c*1000)-round(rv*1000))<=1000)
        flags[e["event_id"]]=dict(context_learned=full,context_no_ood=no_ood,radar_agreement=agree,context_no_ood_plus_radar=agree and no_ood)
        assert e["reference"]==nominal[name,index]["reference"]
        assert e["event_id"] not in events;events[e["event_id"]]=e
    assert dict(crafted)==report["crafted_active_counts"]
    groups=defaultdict(list);seen=set()
    for d in rows(runtime/"decisions.jsonl"):
        e=events[d["event_id"]];unique=d["event_id"],d["policy"]
        assert unique not in seen;seen.add(unique)
        accept=flags[d["event_id"]][d["policy"]]
        assert d["accept"]==accept and d["active"]==e["active"] and d["reference"]==e["reference"]
        assert d["value"]==(e["checked"]["candidate"] if accept else None) and not d["physical_speed_authorized"]
        groups[d["estimator"],d["track"],d["case"],d["policy"]].append(d)
    for (est,track,case,p),rr in groups.items():
        m=report["results"][est][track][case][p];accepted=[r for r in rr if r["accept"]]
        errors=[abs(r["value"]-r["reference"]) for r in accepted if r["reference"] is not None]
        assert m["requests"]==len(rr) and m["accepted"]==len(accepted)
        assert m["coverage_pct"]==100*len(accepted)/len(rr) and m["unscorable_accepted"]==len(accepted)-len(errors)
        assert m["accepted_discrepancy_above_2_mps"]==sum(v>2 for v in errors)
        assert m["active_accepted"]==sum(r["active"] and r["accept"] for r in rr)
        p95=float(np.percentile(errors,95)) if errors else None
        assert p95==m["p95_discrepancy_mps"]
    for est,tracks in report["quality"].items():
        for track,q in tracks.items():
            vv=[r for (ee,n,i,t),r in estimates.items() if ee==est and t==track]
            errors=[abs(v["estimate"]["speed_mps"]-nominal[v["recording"],v["index"]]["reference"])
                for v in vv if v["estimate"]["accepted"] and nominal[v["recording"],v["index"]]["reference"] is not None]
            assert q["accepted"]==sum(v["estimate"]["accepted"] for v in vv)
            assert q["above_2_mps"]==sum(x>2 for x in errors)
            assert q["maximum_discrepancy_mps"]==(max(errors) if errors else None)
    assert len(seen)==report["policy_decisions"]==len(events)*4
    result=dict(all_passed=True,causal_frame_selection_checks=clock_checks,accepted_geometry_checks=geometry_checks,
        signed_packet_checks=packet_checks,events=len(events),policy_decisions=len(seen),
        baseline_estimates_unchanged=True,production_promotion_allowed=False)
    (ROOT/"balanced_independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
