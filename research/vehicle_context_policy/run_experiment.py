"""Frozen development comparison. Research signing keys model transport, not sensor truth."""
import copy,hashlib,json,math,os,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from fractions import Fraction
import context_core as core
import data_adapter as data
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"vehicle_monitor_comparison"))
from monitor_core import Cusum,Ewma
POLICIES=("unguarded","authenticated_only","fixed_agreement","global_learned","context_no_ood","context_learned","cusum","ewma")
def make_body(ctx,source,key,channel,seq,measured,speed):
    return dict(schema=1,source_id=source,key_id=key,vehicle_id=ctx.vehicle_id,session_id=ctx.session_id,
        capture_id=ctx.capture_id,channel=channel,seq=seq,measured_us=measured,speed_mmps=round(speed*1000))
def packets(row,case,active,onset,history,ctx,keys,last_clean):
    now=row["now_us"]; measured=row["measured_us"]; seq=row["index"]
    candidate=row["candidate"]; features=copy.deepcopy(row["features"])
    ctime,atime=measured,measured
    if active:
        if case=="candidate_bias": candidate=candidate+3 if candidate is not None else None
        if case=="candidate_slow_drift" and candidate is not None: candidate+=min(3.,.3*(now-onset)/10**6)
        if case=="candidate_retimed_replay":
            old=[r for r in history if r["now_us"]<=now-10**6 and r["candidate"] is not None]
            candidate=old[-1]["candidate"] if old else None
        if case=="coordinated_bias":
            if candidate is not None: candidate+=3.
            if features is not None: features["peer"]+=3.
        if case=="missing_context": features=None
        if case=="stale_context": atime=now-100001
        if case=="stale_candidate": ctime=now-100001
        if case=="future_candidate": ctime=now+1
    cw=aw=None
    if candidate is not None:
        envelope=core.auth.sign(make_body(ctx,"candidate","kc","speed",seq,ctime,candidate),keys[0])
        if active and case=="tampered_candidate": envelope["body"]["speed_mmps"]+=3000
        cw=json.dumps(envelope,separators=(",",":"))
    if features is not None:
        envelope=core.bind_context(make_body(ctx,"peer","ka","motion_context",seq,atime,features["peer"]),features,keys[1])
        if active and case=="tampered_context": envelope["features"]["yaw"]+=1.
        aw=json.dumps(envelope,separators=(",",":"))
    if active and case=="original_wire_replay": cw=last_clean
    return cw,aw
def decisions(model,checked,unchecked,detectors):
    out={p:dict(accept=False,reason="candidate_"+checked["candidate_status"]) for p in POLICIES}
    out["unguarded"]=dict(accept=unchecked is not None,reason="no_authentication_or_context")
    if not checked["candidate_ok"]: return out
    out["authenticated_only"]=dict(accept=True,reason="authenticated_report_only")
    if not checked["context_ok"]:
        for p in POLICIES[2:]: out[p]["reason"]="context_"+checked["context_status"]
        return out
    c,features=checked["candidate"],checked["features"]
    residual=c-features["peer"]
    out["fixed_agreement"]=dict(accept=abs(residual)<=1.,reason="fixed_agreement_gate",residual=residual,threshold=1.)
    out["global_learned"]=dict(accept=abs(residual-model["global_center"])<=model["global_threshold"],
        reason="global_residual_gate",residual=residual-model["global_center"],threshold=model["global_threshold"])
    out["context_no_ood"]=core.assess(model,c,features,False)
    out["context_learned"]=core.assess(model,c,features,True)
    z=(residual-model["monitor_mean"])/model["monitor_sd"]
    for p,detector in zip(("cusum","ewma"),detectors):
        try:
            state,alarm=detector.step(z)
            out[p]=dict(accept=not alarm,reason="statistical_alarm" if alarm else "statistically_consistent",state=state)
        except ArithmeticError: out[p]=dict(accept=False,reason="statistical_numerical_failure")
    return out
def metrics(rows):
    accepted=[r for r in rows if r["accept"]]
    scored=[r for r in accepted if r["reference"] is not None]
    errors=[abs(r["value"]-r["reference"]) for r in scored]
    attack=[r for r in rows if r["active"]]
    empirical=None
    if len(scored)==len(accepted):
        if errors:
            ordered=sorted(errors)
            index=Fraction(19,20)*(len(ordered)-1)
            lo=index.numerator//index.denominator
            weight=index-lo
            exact_p95=Fraction(ordered[lo])*(1-weight)+Fraction(ordered[min(lo+1,len(ordered)-1)])*weight
            empirical=bool(exact_p95<=Fraction(1,2) and len(accepted)*100>=95*len(rows) and not any(e>2 for e in errors))
        else: empirical=False
    return dict(requests=len(rows),accepted=len(accepted),coverage_pct=100*len(accepted)/len(rows),
        scored_accepted=len(scored),unscorable_accepted=len(accepted)-len(scored),
        p95_discrepancy_mps=float(np.quantile(errors,.95)) if errors else None,
        accepted_discrepancy_above_2_mps=sum(e>2 for e in errors),
        active_requests=len(attack),active_accepted=sum(r["accept"] for r in attack),
        active_rejection_pct=100*sum(not r["accept"] for r in attack)/len(attack) if attack else None,
        empirical_all_request_gate=empirical,physical_gate_pass=None,
        reasons=dict(Counter(r["reason"] for r in rows)))
def main():
    spec_path=ROOT/"protocol_v1.json"
    protocol_hash=hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert protocol_hash==json.loads((ROOT/"protocol_freeze.json").read_text())["protocol_sha256"]
    spec=json.loads(spec_path.read_text())
    runtime=ROOT/"runtime"/("run_"+os.urandom(8).hex())
    online,references,input_hashes=data.build(spec,runtime)
    def eligible(r):
        return r["candidate"] is not None and r["features"] is not None and 0<=r["now_us"]-r["measured_us"]<=100000
    training=[r for r in online if r["role"]=="train" and eligible(r)]
    calibration=[r for r in online if r["role"]=="calibration" and eligible(r)]
    model=core.fit_model(training,calibration)
    (ROOT/"fitted_model.json").write_text(json.dumps(model,indent=2)+"\n")
    (runtime/"fitted_model.json").write_text(json.dumps(model,indent=2)+"\n")
    keys=(os.urandom(32),os.urandom(32))
    (runtime/"research_signing_keys.json").write_text(json.dumps([k.hex() for k in keys]))
    split_counts={name:dict(Counter(r["role"] for r in online if r["recording"]==name)) for name in spec["records"]}
    collection=defaultdict(list);per_record=defaultdict(list);event_count=0
    with (runtime/"packet_events.jsonl").open("w") as events,(runtime/"decisions.jsonl").open("w") as audit:
        for name in spec["records"]:
            history=[r for r in online if r["recording"]==name]
            evaluation=[r for r in history if r["role"]=="evaluation"]
            onset=evaluation[0]["now_us"]+10**6
            for case in spec["cases"]:
                context=core.auth.Context("dev_vehicle",name,case)
                state=runtime/(name+"_"+case+".sqlite")
                if state.exists(): raise ValueError("Existing replay state; use a new run directory instead of resetting security state")
                core.auth.create_state(state,context)
                guard=core.auth.Guard(state,context,[
                    core.auth.KeyRule("candidate","kc","speed",keys[0]),
                    core.auth.KeyRule("peer","ka","motion_context",keys[1])])
                detectors=(Cusum(),Ewma());last_clean=None
                for row in evaluation:
                    active=case!="nominal" and row["now_us"]>=onset
                    cw,aw=packets(row,case,active,onset,history,context,keys,last_clean)
                    if not active: last_clean=cw
                    checked=core.authenticate(guard,cw,aw,row["now_us"])
                    unchecked=json.loads(cw)["body"]["speed_mmps"]/1000 if cw is not None else None
                    outputs=decisions(model,checked,unchecked,detectors)
                    reference=data.interpolate(references[name],row["now_us"])
                    event_id=f"{name}:{case}:{row['index']}"
                    event=dict(event_id=event_id,recording=name,case=case,index=row["index"],now_us=row["now_us"],
                        active=active,candidate_wire=cw,context_wire=aw,reference=reference,checked=checked)
                    events.write(json.dumps(event,allow_nan=False)+"\n")
                    for policy,result in outputs.items():
                        value=unchecked if policy=="unguarded" else checked["candidate"]
                        item=dict(event_id=event_id,recording=name,case=case,policy=policy,
                            active=active,accept=result["accept"],value=value if result["accept"] else None,
                            reference=reference,reason=result["reason"],explanation=result,
                            physical_speed_authorized=False,production_policy_changed=False)
                        audit.write(json.dumps(item,allow_nan=False)+"\n")
                        collection[(case,policy)].append(item)
                        per_record[(name,case,policy)].append(item)
                    event_count+=1
                print(name+" / "+case+": complete",flush=True)
    results={case:{policy:metrics(collection[(case,policy)]) for policy in POLICIES} for case in spec["cases"]}
    record_results={name:{case:{p:metrics(per_record[(name,case,p)]) for p in POLICIES} for case in spec["cases"]} for name in spec["records"]}
    report=dict(protocol_sha256=protocol_hash,source_extracted_sha256=input_hashes,split_counts=split_counts,
        runtime_relative_path=str(runtime.relative_to(ROOT)),
        training_count=len(training),calibration_count=len(calibration),packet_events=event_count,
        policy_decisions=event_count*len(POLICIES),policies=list(POLICIES),results=results,per_record=record_results,
        development_only=True,previously_inspected_nominal_development_data=True,independent_confirmation=False,
        shared_can_failure_domain=True,research_transport_simulation=True,acquisition_authenticated=False,
        physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False,
        limitations=["All records from one acquisition day","All context signals share vehicle CAN origin",
            "Statistical baselines are adaptations, not full published AV defense reproductions",
            "Nominal-only ridge regression is a development AI baseline; novelty is not established",
            "Calibration quantiles are empirical, without an exchangeability or adversarial coverage guarantee",
            "Offline INS comparisons may be unavailable near recording boundaries; unknown errors are counted",
            "Research signatures and durable replay checks do not authenticate archived sensor acquisition"])
    (ROOT/"experiment_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (runtime/"experiment_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(packet_events=event_count,policy_decisions=report["policy_decisions"],
        nominal=results["nominal"],coordinated_bias=results["coordinated_bias"]),indent=2),flush=True)
if __name__=="__main__": main()
