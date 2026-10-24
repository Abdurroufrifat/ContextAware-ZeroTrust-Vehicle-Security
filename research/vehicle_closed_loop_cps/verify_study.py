"""Independent control/actuation, model, trajectory and metric verification."""
from cps_common import R,P,CS,BINARY,sha,dump
import sys,json,gzip,math,hmac,hashlib,bisect,random,xml.etree.ElementTree as ET
from fractions import Fraction
import numpy as np
import joblib
model_path=CS/"models"/"forest.joblib"
assert sha(model_path)==json.loads((CS/"model_lock.json").read_text())[str(model_path.relative_to(P))],"model integrity before deserialization"
MODEL=joblib.load(model_path)
KEYS={n:hashlib.sha256(("COACT-SUMO-PUBLIC-TEST-"+n).encode()).digest() for n in ("scene","radar","actuator")}
SCENE_FIELDS=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","gap_um","lead_mmps")
def near(a,b,tol=1e-7):
    if a is None or b is None:assert a is b,(a,b)
    else:assert abs(a-b)<=tol,(a,b)
def verify_packet(role,packet,now,ctx,counters):
    if packet is None:return None,"missing"
    obj=json.loads(packet["wire"]);body=obj["body"]
    assert all(body[k]==v for k,v in ctx.items())
    if role=="scene":
        data=b"COACTCPS-SCENE-HMAC-SHA256-v1\0"+json.dumps([body[k] for k in SCENE_FIELDS],ensure_ascii=True,separators=(",",":")).encode()
    else:data=("COACTCPS-CLOSEDLOOP-"+role+"-v1\0").encode()+json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
    if not hmac.compare_digest(obj["tag"],hmac.new(KEYS[role],data,hashlib.sha256).hexdigest()):return None,"bad_tag"
    stamp=body["issued_us" if role=="actuator" else "measured_us"]
    if stamp!=now or packet["received_us"]!=now:return None,"time_mismatch"
    if body["seq"]<=counters.get(role,-1):return None,"replayed"
    counters[role]=body["seq"];return body,"authenticated"
def nominal(ego,lead,gap):
    headway=2.+max(0.,1.5*ego+ego*(ego-lead)/(2.*(8.**.5)))
    result=2.*(1.-(ego/18.)**4-(headway/max(.001,gap))**2)
    return int(min(2000,max(-4000,round(result*1000))))
def full_path_margin(command,ve,vl,gap):
    minimum=gap-2.
    for _ in range(10):
        ve=max(0.,ve+min(2.,command/1000+1.)*.01)
        vl=max(0.,vl-.04);gap+=(vl-ve)*.01
        minimum=min(minimum,gap-2.)
    count=max(math.ceil(ve/.03),math.ceil(vl/.04))
    if count:
        j=np.arange(1,count+1,dtype=np.float64)
        me=np.minimum(j,max(0,math.ceil(ve/.03)-1))
        ml=np.minimum(j,max(0,math.ceil(vl/.04)-1))
        xe=.01*(me*ve-.03*me*(me+1)/2)
        xl=.01*(ml*vl-.04*ml*(ml+1)/2)
        minimum=min(minimum,float((gap+xl-xe-2.).min()))
    return minimum
def fuse(speed,scene,radar):
    if speed is None or scene is None or radar is None:return None
    lowgap=max(scene["gap_um"]/1e6-.0015,radar["gap_um"]/1e6-.05)
    highgap=min(scene["gap_um"]/1e6+.0015,radar["gap_um"]/1e6+.05)
    elow=max(0,speed-.5);ehigh=speed+.5
    lowlead=max(0,scene["lead_mmps"]/1000-.075,radar["relative_mmps"]/1000-.1+elow)
    highlead=min(scene["lead_mmps"]/1000+.075,radar["relative_mmps"]/1000+.1+ehigh)
    if lowgap>highgap or lowlead>highlead:return None
    return {"ego_low":elow,"ego_high":ehigh,"gap_low":lowgap,"gap_high":highgap,"lead_low":lowlead,"lead_high":highlead,
            "gap_m":(lowgap+highgap)/2,"lead_mps":(lowlead+highlead)/2}
def forest_prediction(features):
    x=np.asarray(features,dtype=np.float32)
    answers=[]
    for estimator in MODEL.estimators_:
        t=estimator.tree_;node=0
        while t.children_left[node]!=t.children_right[node]:
            node=t.children_left[node] if x[t.feature[node]]<=t.threshold[node] else t.children_right[node]
        answers.append(float(t.value[node,0,0]))
    return math.fsum(answers)/len(answers)
def main():
    pilot="--pilot" in sys.argv
    result=json.loads((R/("runtime/pilot_results_v3.json" if pilot else "results.json")).read_text())
    if not pilot:
        assert all(sha(P/n)==v for n,v in json.loads((R/"experiment_lock.json").read_text()).items())
        assert len(result["trials"])==540
    cfg=json.loads((R/"PROTOCOL.json").read_text())
    counts={"trials":0,"decisions":0,"queried_vehicle_states":0,"ego_transitions":0,"lead_transitions":0,
            "forest_predictions":0,"operator_cards":0,"proposed_contexts":0,"actual_context_violations":0,
            "qualified_contexts":0,"qualified_context_violations":0,"selected_speed_bound_violations":0,
            "qualified_selected":0,"qualified_selected_bound_violations":0,"qualified_selected_strict_errors":0,
            "nonnegative_margin_followups":0,"qualified_followups":0,"qualified_followup_gap_violations":0,"changed_actuator_frames":0,"changed_frames_accepted":0,
            "proposed_changed_actuator_frames":0,"proposed_changed_frames_accepted":0,"proposed_advisory_instructions":0,"proposed_advisory_accepted":0}
    per=[]
    for trial in result["trials"]:
        root=R/PathLike(trial["trace"]).parent
        with gzip.open(root/"trace.jsonl.gz","rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
        assert len(rows)==200
        states={}
        with gzip.open(root/"fcd.xml.gz","rt",encoding="utf-8") as f:tree=ET.parse(f).getroot()
        for tick in tree:
            stamp=round(float(tick.attrib["time"])*1e6)+10000
            states[stamp]={v.attrib["id"]:{k:float(v.attrib[k]) for k in ("x","speed")} for v in tick}
        counters={};previous_scene_ok=False;previous_scene_time=None;errors=[];fallback=unrecoverable=advice_accepted=tamper_accepted=0
        qualified=qualified_errors=contexts_bad=certified_followups=followups_bad=0
        qtimes=[round(r["time_s"]*1e6) for r in rows]
        rng=random.Random(trial["seed"]+10000);disturbance=[rng.randint(-500,500) for _ in range(201)]
        times=sorted(states)
        for a,b in zip(times,times[1:]):
            assert b-a==10000
            index=bisect.bisect_right(qtimes,a)-1
            acceleration=rows[index]["applied_mmps2"]/1000 if index>=0 else disturbance[0]/1000
            for name,acc in [("ego",acceleration),("lead",cfg["profiles"][trial["profile"]]["accelerations"][min(4,a//4000000)])]:
                before=states[a][name];after=states[b][name]
                near(after["speed"],max(0,before["speed"]+acc*.01),2e-6)
                near(after["x"]-before["x"],after["speed"]*.01,3e-6)
                counts[name+"_transitions"]+=1
        for i,row in enumerate(rows):
            inp=row["input"];out=row["output"];truth=row["truth"];now=inp["now_us"]
            assert set(inp)=={"now_us","risk","packets","scene_packet","radar_packet","advisory"}
            assert now==1000000+qtimes[i]
            current=states[qtimes[i]]
            ve=current["ego"]["speed"];vl=current["lead"]["speed"]
            gap=current["lead"]["x"]-current["ego"]["x"]-5
            near(ve,truth["ego_speed_mps"],1e-6);near(vl,truth["lead_speed_mps"],1e-6);near(gap,truth["gap_m"],2e-6)
            counts["queried_vehicle_states"]+=2
            scene,ss=verify_packet("scene",inp["scene_packet"],now,row["context"],counters)
            radar,rs=verify_packet("radar",inp["radar_packet"],now,row["context"],counters)
            assert out["radar_status"]==rs
            mode=trial["mode"];proposed=mode.startswith("proposed")
            if mode=="raw":
                scene=json.loads(inp["scene_packet"]["wire"])["body"] if inp["scene_packet"] else None
                value=json.loads(inp["packets"]["candidate"]["wire"])["body"]["speed_mmps"]
                assert out["estimate_mmps"]==value
                assert out["scene_status"]==("unverified" if scene else ss)
            else:
                assert out["scene_status"]==ss
                decision=out["estimator_decision"];value=decision["research_speed_mmps"]
                assert out["estimate_mmps"]==value
                if decision.get("refined_budget_satisfied"):
                    lo=Fraction(decision["refined_low_n"],decision["refined_low_d"]);hi=Fraction(decision["refined_high_n"],decision["refined_high_d"])
                    prediction=float((lo+hi)/2)
                    if decision["residual_prediction_mps"] is not None:
                        residual=forest_prediction(decision["relative_features"])
                        near(residual,decision["residual_prediction_mps"],1e-10);prediction+=residual*1000
                        counts["forest_predictions"]+=1
                    expected=min(math.floor(lo+500),max(max(0,math.ceil(hi-500)),round(prediction)))
                    assert value==expected
            speed=None if value is None else value/1000
            error=None if speed is None else abs(speed-ve)
            near(error,row["selected_error_mps"],1e-6)
            if error is not None:
                errors.append(row["selected_error_mps"])
                counts["selected_speed_bound_violations"]+=error>.500002
            previous_applied=rows[i-1]["applied_mmps2"] if i else disturbance[0]
            previous_accepted=rows[i-1]["accepted_mmps2"] if i else 0
            assert truth["previous_applied_mmps2"]==previous_applied and truth["previous_accepted_mmps2"]==previous_accepted
            execution_ok=abs(previous_applied-previous_accepted)<=1000
            scene_ok=scene is not None and ss=="authenticated" and abs(scene["gap_um"]/1e6-gap)<=.001503 and abs(scene["lead_mmps"]/1000-vl)<=.075002
            radar_ok=radar is not None and abs(radar["gap_um"]/1e6-gap)<=.050003 and abs(radar["relative_mmps"]/1000-(vl-ve))<=.100003
            decision=out["estimator_decision"]
            needs_execution=bool(decision and decision.get("refined_budget_satisfied"))
            pair_scene_ok=scene_ok and previous_scene_ok and previous_scene_time is not None and now-previous_scene_time<=100000
            speed_premises=pair_scene_ok and (execution_ok or not needs_execution)
            if ss=="authenticated":previous_scene_ok=scene_ok;previous_scene_time=now
            if error is not None and mode!="raw" and speed_premises:
                counts["qualified_selected"]+=1;qualified+=1
                counts["qualified_selected_strict_errors"]+=row["selected_error_mps"]>.5
                counts["qualified_selected_bound_violations"]+=error>.500002;qualified_errors+=error>.500002
            observed=None;context=None;desired=-4000;reason="observation_unavailable"
            if speed is not None and scene is not None:
                if proposed:
                    context=fuse(speed,scene,radar)
                    if context is None:reason="context_disagreement" if radar is not None else "radar_"+rs
                    else:observed={"ego_mps":speed,"lead_mps":context["lead_mps"],"gap_m":context["gap_m"]}
                else:observed={"ego_mps":speed,"lead_mps":scene["lead_mmps"]/1000,"gap_m":scene["gap_um"]/1e6}
            if observed is not None:desired=nominal(observed["ego_mps"],observed["lead_mps"],observed["gap_m"])
            assert out["controller_input"]==observed and out["fused_context"]==context
            assert out["desired_mmps2"]==desired
            command=out["requested_mmps2"]
            if proposed and context is not None:
                c=context;actual_margin=full_path_margin(command,c["ego_high"],c["lead_low"],c["gap_low"])
                near(actual_margin,out["projected_margin_m"],1e-7)
                assert -4000<=command<=desired
                if actual_margin<0:assert command==-4000;reason="reserve_unrecoverable";unrecoverable+=1
                else:
                    if command<desired:assert full_path_margin(command+1,c["ego_high"],c["lead_low"],c["gap_low"])<1e-7
                    reason="reserve_limited" if command!=desired else "observed_context_accepted"
                contained=c["ego_low"]-2e-6<=ve<=c["ego_high"]+2e-6 and c["lead_low"]-2e-6<=vl<=c["lead_high"]+2e-6 and c["gap_low"]-3e-6<=gap<=c["gap_high"]+3e-6
                counts["proposed_contexts"]+=1;counts["actual_context_violations"]+=not contained;contexts_bad+=not contained
                qualified_context=scene_ok and radar_ok and speed_premises
                if qualified_context:
                    counts["qualified_contexts"]+=1;counts["qualified_context_violations"]+=not contained
                if actual_margin>=0 and i<199:
                    counts["nonnegative_margin_followups"]+=1
                    current_execution=row["applied_mmps2"]<=min(2000,command+1000)
                    if contained and current_execution:
                        segment=[t for t in times if qtimes[i]<=t<=qtimes[i+1]]
                        lowest=min(states[t]["lead"]["x"]-states[t]["ego"]["x"]-5 for t in segment)
                        counts["qualified_followups"]+=1;certified_followups+=1
                        counts["qualified_followup_gap_violations"]+=lowest<2.-3e-6;followups_bad+=lowest<2.-3e-6
            elif observed is not None:
                reason="raw_observations" if mode=="raw" else "authenticated_scene_control"
                expected=2000 if mode=="raw" and "accelerate" in inp["advisory"].casefold() else desired
                assert command==expected
            else:assert command==-4000 or (mode=="raw" and inp["advisory"])
            if mode=="raw" and "accelerate" in inp["advisory"].casefold():
                assert command==2000;reason="untrusted_advisory_accepted";advice_accepted+=1
            else:assert out["advisory_action"] is None
            assert out["reason"]==reason,(trial["trial_id"],i,out["reason"],reason)
            body,status=verify_packet("actuator",row["delivered_command"],now,row["context"],counters)
            if proposed:accepted=body["accel_mmps2"] if body else -4000;expected_status=status
            else:accepted=json.loads(row["delivered_command"]["wire"])["body"]["accel_mmps2"];expected_status="unverified"
            assert row["accepted_mmps2"]==accepted and row["actuator_status"]==expected_status
            changed=row["delivered_command"]!=row["issued_command"]
            counts["changed_actuator_frames"]+=changed
            counts["changed_frames_accepted"]+=changed and expected_status=="unverified"
            if proposed:
                counts["proposed_changed_actuator_frames"]+=changed
                counts["proposed_changed_frames_accepted"]+=changed and status=="authenticated"
                counts["proposed_advisory_instructions"]+=bool(inp["advisory"])
                counts["proposed_advisory_accepted"]+=out["advisory_action"] is not None
            expected_actual=max(-4000,min(2000,accepted+(2500 if row["active"] and trial["case"]=="execution_break" else disturbance[i+1])))
            assert row["applied_mmps2"]==expected_actual
            tamper_accepted+=row["active"] and trial["case"]=="actuator_tamper" and expected_status=="unverified"
            fallback+=observed is None or expected_status not in ("authenticated","unverified")
            card=row["operator_card"]
            assert card["reason"]==reason and card["accepted_command_mmps2"]==accepted and card["actuator_status"]==expected_status
            assert card["controller_input"]==observed and card["requested_command_mmps2"]==command
            assert not ({"truth","ego_speed_mps","attack_case","label"} & set(card))
            assert card["assurance"]=="conditional simulation evidence" and card["human_evaluation"]=="not performed"
            counts["operator_cards"]+=1;counts["decisions"]+=1
        gaps=[v["lead"]["x"]-v["ego"]["x"]-5 for v in states.values()]
        with gzip.open(root/"collisions.xml.gz","rt",encoding="utf-8") as f:collisions=list(ET.parse(f).getroot())
        near(trial["mae_mps"],math.fsum(errors)/len(errors) if errors else None)
        assert trial["selected"]==len(errors)
        assert trial["over_0_5"]==sum(x>.5 for x in errors) and trial["over_2"]==sum(x>2 for x in errors)
        near(trial["minimum_gap_m"],min(gaps),2e-6);near(trial["seconds_gap_below_2m"],sum(x<2 for x in gaps)*.01)
        near(trial["progress_m"],rows[-1]["truth"]["ego_position_m"]-60)
        assert trial["collision_events"]==len(collisions)
        assert trial["fallback_queries"]==fallback and trial["unrecoverable_queries"]==unrecoverable
        assert trial["advisory_accepted"]==advice_accepted and trial.get("tamper_case_unverified_queries",trial.get("actuator_tamper_accepted"))==tamper_accepted
        counts["trials"]+=1
        per.append({"trial_id":trial["trial_id"],"qualified_selected":qualified,"qualified_selected_errors":qualified_errors,
                    "actual_context_violations":contexts_bad,"qualified_followups":certified_followups,"qualified_followup_gap_violations":followups_bad})
    assert counts["qualified_context_violations"]==0 and counts["qualified_selected_bound_violations"]==0
    assert counts["qualified_followup_gap_violations"]==0
    dump(R/("runtime/pilot_verification.json" if pilot else "verification.json"),{"passed":True,"counts":counts,"per_trial":per,
        "scope":"independent new controller/actuator replay, manual forest trees/projection, raw SUMO FCD dynamics and summaries; frozen estimator gateway component separately verified"})
    print("INDEPENDENT_CPS_VERIFICATION_PASS",json.dumps(counts),flush=True)
from pathlib import Path as PathLike
if __name__=="__main__":main()
