"""Independent packet, replay, model and decision audit; imports no policy implementation."""
import hashlib,hmac,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
ORDER=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","speed_mmps")
def canonical(body):
    assert set(body)==set(ORDER) and body["schema"]==1
    return b"COACTCPS-EVIDENCE-HMAC-SHA256-v1\x00"+json.dumps([body[k] for k in ORDER],ensure_ascii=True,separators=(",",":")).encode("ascii")
def basis(f):
    assert set(f)=={"peer","yaw","steering","acceleration"}
    p,y,s,a=f["peer"],abs(f["yaw"]),abs(f["steering"]),abs(f["acceleration"])
    return np.array([1.,p,y,s,a,p*s*s,p*y,p*p])
def main():
    spec=json.loads((ROOT/"protocol_v1.json").read_text())
    report=json.loads((ROOT/"experiment_summary.json").read_text())
    runtime=ROOT/report.get("runtime_relative_path","runtime")
    keys=[bytes.fromhex(k) for k in json.loads((runtime/"research_signing_keys.json").read_text())]
    online=json.loads((runtime/"online_inputs.json").read_text())
    refs=json.loads((runtime/"offline_references.json").read_text())
    model=json.loads((ROOT/"fitted_model.json").read_text())
    assert hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()==report["protocol_sha256"]
    assert not report["promotion_allowed"] and not report["physical_witness_enabled"]
    eligible=lambda r:r["candidate"] is not None and r["features"] is not None and 0<=r["now_us"]-r["measured_us"]<=100000
    train=[r for r in online if r["role"]=="train" and eligible(r)]
    cal=[r for r in online if r["role"]=="calibration" and eligible(r)]
    x=np.array([basis(r["features"]) for r in train])
    target=np.array([r["candidate"]-r["features"]["peer"] for r in train])
    center=x.mean(axis=0);center[0]=0.
    scale=x.std(axis=0);scale[scale<1e-9]=1.;scale[0]=1.
    z=(x-center)/scale;penalty=np.eye(8);penalty[0,0]=0.
    beta=np.linalg.solve(z.T@z+penalty,z.T@target)
    assert np.allclose(beta,model["coef"],rtol=0,atol=1e-12)
    prediction=lambda f:float(((basis(f)-center)/scale)@beta)
    errors=sorted(abs(r["candidate"]-r["features"]["peer"]-prediction(r["features"])) for r in cal)
    threshold=max(.05,errors[math.ceil((len(errors)-1)*.995)])
    assert threshold==model["threshold"]
    assert len(train)==report["training_count"] and len(cal)==report["calibration_count"]
    for record in spec["records"]:
        rows=[r for r in online if r["recording"]==record]
        times={role:[r["now_us"] for r in rows if r["role"]==role] for role in ("train","calibration","evaluation")}
        assert max(times["train"])+10**6<=min(times["calibration"])
        assert max(times["calibration"])+10**6<=min(times["evaluation"])
        assert all(set(r)=={"recording","index","role","now_us","measured_us","candidate","features"} for r in rows)
    state={};event_map={};expected_decisions={}
    def checked(envelope,now,record,case,source):
        body=envelope["body"];key=keys[0 if source=="candidate" else 1]
        if not hmac.compare_digest(hmac.new(key,canonical(body),hashlib.sha256).hexdigest(),envelope["tag"]): return "bad_tag"
        assert body["source_id"]==source and body["vehicle_id"]=="dev_vehicle"
        assert body["session_id"]==record and body["capture_id"]==case
        if not body["measured_us"]<=now: return "invalid_clock_order"
        if now-body["measured_us"]>100000: return "measurement_stale"
        scope=(record,case,source)
        if body["seq"]<=state.get(scope,-1): return "replay_or_reorder"
        state[scope]=body["seq"]
        return "authenticated"
    with (runtime/"packet_events.jsonl").open() as inp:
        for line in inp:
            e=json.loads(line);event_map[e["event_id"]]=e
            now=e["now_us"];record=e["recording"];case=e["case"]
            candidate=json.loads(e["candidate_wire"]) if e["candidate_wire"] else None
            context=json.loads(e["context_wire"]) if e["context_wire"] else None
            cs=checked(candidate,now,record,case,"candidate") if candidate else "missing"
            if context:
                b=context["report"]["body"];f=context["features"]
                bound=b"COACT-CONTEXT-BINDING-v1\x00"+canonical(b)+json.dumps(f,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
                tag=hmac.new(keys[1],bound,hashlib.sha256).hexdigest()
                context_status=checked(context["report"],now,record,case,"peer") if hmac.compare_digest(tag,context["context_tag"]) else "bad_context_binding"
            else: context_status="missing"
            assert cs==e["checked"]["candidate_status"] and context_status==e["checked"]["context_status"]
            assert e["checked"]["candidate_ok"]==(cs=="authenticated")
            assert e["checked"]["context_ok"]==(context_status=="authenticated")
            values=refs[record];t=[v[0] for v in values];j=int(np.searchsorted(t,now))
            if j<len(t) and t[j]==now: reference=values[j][1]
            elif j==0 or j==len(t) or t[j]-t[j-1]>50000: reference=None
            else: reference=values[j-1][1]+(now-t[j-1])/(t[j]-t[j-1])*(values[j][1]-values[j-1][1])
            assert (reference is None and e["reference"] is None) or abs(reference-e["reference"])<1e-12
            results={p:False for p in report["policies"]}
            results["unguarded"]=candidate is not None
            results["authenticated_only"]=cs=="authenticated"
            if cs=="authenticated" and context_status=="authenticated":
                c=candidate["body"]["speed_mmps"]/1000;f=context["features"];residual=c-f["peer"]
                results["fixed_agreement"]=abs(residual)<=1.
                results["global_learned"]=abs(residual-model["global_center"])<=model["global_threshold"]
                normal=abs(residual-prediction(f))<=threshold
                raw=np.array([f["peer"],abs(f["yaw"]),abs(f["steering"]),abs(f["acceleration"])])
                inside=bool(np.all(raw>=np.array(model["raw_min"])-np.array([.5,.05,.1,1.])) and np.all(raw<=np.array(model["raw_max"])+np.array([.5,.05,.1,1.])))
                results["context_no_ood"]=normal
                results["context_learned"]=normal and inside
                stat_key=(record,case,"stat")
                hi,lo,ew=state.get(stat_key,(0.,0.,0.))
                normalized=(residual-model["monitor_mean"])/model["monitor_sd"]
                hi=max(0.,hi+normalized-.5);lo=max(0.,lo-normalized-.5);ew=.3*normalized+.7*ew
                state[stat_key]=(hi,lo,ew)
                results["cusum"]=max(hi,lo)<=5.
                results["ewma"]=abs(ew)<=3*math.sqrt(.3/1.7)
            expected_decisions[e["event_id"]]=results
    groups=defaultdict(list);seen=set();count=0
    with (runtime/"decisions.jsonl").open() as inp:
        for line in inp:
            r=json.loads(line);key=(r["event_id"],r["policy"])
            assert key not in seen;seen.add(key)
            e=event_map[r["event_id"]]
            assert r["accept"]==expected_decisions[r["event_id"]][r["policy"]]
            assert r["reference"]==e["reference"] and r["active"]==e["active"]
            if r["accept"]:
                body=json.loads(e["candidate_wire"])["body"]
                assert r["value"]==body["speed_mmps"]/1000
            else: assert r["value"] is None
            assert not r["physical_speed_authorized"] and not r["production_policy_changed"]
            groups[(r["case"],r["policy"])].append(r);count+=1
    assert count==len(event_map)*8==report["policy_decisions"]
    for (case,policy),rows in groups.items():
        result=report["results"][case][policy]
        accepted=[r for r in rows if r["accept"]]
        errors=[abs(r["value"]-r["reference"]) for r in accepted if r["reference"] is not None]
        assert len(rows)==result["requests"] and len(accepted)==result["accepted"]
        assert result["unscorable_accepted"]==sum(r["reference"] is None for r in accepted)
        assert sum(v>2 for v in errors)==result["accepted_discrepancy_above_2_mps"]
        p95=float(np.quantile(errors,.95)) if errors else None
        assert (p95 is None and result["p95_discrepancy_mps"] is None) or abs(p95-result["p95_discrepancy_mps"])<1e-12
    status=dict(all_passed=True,independent_packet_events=len(event_map),independent_policy_decisions=count,
        model_coefficients_and_calibration_recomputed=True,chronological_purges_checked=True,
        unknown_reference_outputs_counted=True,production_promotion=False)
    (ROOT/"verification_summary.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))
if __name__=="__main__": main()
