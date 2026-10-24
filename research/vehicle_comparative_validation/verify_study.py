"""Independent raw-state, baseline-set/control, forest and summary checks."""
from cv_common import B,P,OLD,KEYS,sha,dump
import sys,json,csv,gzip,math,hmac,hashlib,bisect,random,itertools,importlib.util
from fractions import Fraction as F
from verification_contracts import expected_public_run,expected_delivery,verify_checkpoint,verify_refinement_metadata,expected_reason,accepted_history,verify_command_history
from verification_math import reference_sets,reference_context,reference_cbf,close,midpoint_integer_valid
spec=importlib.util.spec_from_file_location("frozen_independent_cps_math",OLD/"verify_study.py")
REF=importlib.util.module_from_spec(spec);spec.loader.exec_module(REF)
CFG=json.loads((B/"PROTOCOL.json").read_text());PUBLIC=json.loads((B/"public_sequences.json").read_text())
FIELDS=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","speed_mmps")
def speed_body(packet,role,now,ctx,seq):
    if packet is None:return None
    obj=json.loads(packet["wire"]);body=obj["body"]
    data=b"COACTCPS-EVIDENCE-HMAC-SHA256-v1\0"+json.dumps([body[k] for k in FIELDS],ensure_ascii=True,separators=(",",":")).encode("ascii")
    assert hmac.compare_digest(obj["tag"],hmac.new(KEYS[role],data,hashlib.sha256).hexdigest())
    assert body["seq"]==seq and all(body[k]==v for k,v in ctx.items())
    assert body["schema"]==1 and body["source_id"]==role and body["key_id"]==("c-key" if role=="candidate" else "a-key")
    assert body["channel"]==("wheel_speed" if role=="candidate" else "protected_speed")
    assert body["measured_us"]<=packet["received_us"]==now
    return body
def verify_public():
    counts=0
    for label,series in PUBLIC.items():
        files={PathLike(n).name:P/n for n in series["source_files"]}
        for n,h in series["source_files"].items():assert sha(P/n)==h
        with files["commands.csv"].open(newline="",encoding="utf-8") as f:commands=[int(r["time_ns"]) for r in csv.DictReader(f)]
        with files["reference.csv"].open(newline="",encoding="utf-8") as f:refs=list(csv.DictReader(f))
        times=[int(r["time_ns"]) for r in refs];speeds=[]
        for i,stored_age in enumerate(series["age_us"]):
            target=times[0]+i*100_000_000;j=bisect.bisect_right(times,target)-1
            r=refs[j];v=(float(r["x"])**2+float(r["y"])**2+float(r["z"])**2)**.5;speeds.append(v)
            k=bisect.bisect_right(commands,target)-1
            assert stored_age==(None if k<0 else (target-commands[k])//1000)
            close(v,series["reference_speed_mps"][i],1e-12)
            close(None if i<2 else speeds[-1]-2*speeds[-2]+speeds[-3],series["innovation_mps"][i],1e-12);counts+=1
    return counts
def main():
    pilot="--pilot" in sys.argv
    result_path=PathLike(sys.argv[sys.argv.index("--pilot")+1]) if pilot else B/"results.json"
    result=json.loads(result_path.read_text());trials=result["trials"]
    if not pilot:
        assert len(trials)==1512
        assert result["lock_digest"]==sha(B/"experiment_lock.json")
        assert sorted(t["trial_id"] for t in trials)==list(range(1512))
        lock=json.loads((B/"experiment_lock.json").read_text());assert all(sha(P/n)==h for n,h in lock.items())
        expected=set()
        for group,cases,methods in [("primary",CFG["primary_cases"],CFG["methods"]),("stress",CFG["stress_cases"],CFG["stress_methods"])]:
            expected|={(group,*x) for x in itertools.product(CFG["profiles"],cases,CFG["seeds"],methods)}
        assert {(t["group"],t["profile"],t["case"],t["seed"],t["method"]) for t in trials}==expected
    counts={"trials":0,"decisions":0,"vehicle_transitions":0,"forest_predictions":0,"operator_cards":0,"baseline_updates":0,
            "baseline_qualified_sets":0,"baseline_qualified_set_violations":0,"point_certified_qualified":0,"point_certified_qualified_errors":0,
            "admitted_contexts":0,"actual_context_violations":0,"qualified_followups":0,"qualified_followup_gap_violations":0,
            "changed_actuator_frames":0,"changed_frames_accepted":0,"refinement_eligible_queries":0,"public_profile_rows":verify_public()}
    per=[]
    for trial in trials:
        method=trial["method"];case=trial["case"];seed=trial["seed"];c=CFG["profiles"][trial["profile"]]
        verify_checkpoint(B,trial,result["lock_digest"])
        expected_run=expected_public_run(CFG,seed,pilot)
        with gzip.open(B/trial["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
        with gzip.open(B/trial["trajectory"],"rt",encoding="utf-8") as f:trajectory=[json.loads(x) for x in f]
        assert trial["queries"]==len(rows)==200 and len(trajectory)==2000
        states={r["time_us"]:r for r in trajectory}
        initial=trajectory[0];assert initial["time_us"]==10000
        close({k:initial[k] for k in ("ego_v","lead_v","ego_x","lead_x")},{"ego_v":c["ego_speed"],"lead_v":c["lead_speed"],"ego_x":60,"lead_x":65+c["gap"]})
        rng=random.Random(seed);noise=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(201)]
        rng=random.Random(seed+10000);dist=[rng.randint(-500,500) for _ in range(201)]
        rng=random.Random(seed+20000);sn=[(rng.randint(-1000,1000),rng.randint(-50,50)) for _ in range(201)]
        rng=random.Random(seed+30000);rn=[(rng.randint(-20000,20000),rng.randint(-50,50)) for _ in range(201)]
        for before,after in zip(trajectory,trajectory[1:]):
            assert after["time_us"]-before["time_us"]==10000
            qi=before["time_us"]//100000-1
            ea=dist[0]/1000 if qi<0 else rows[qi]["next_applied_mps2"]
            la=c["accelerations"][min(4,before["time_us"]//4000000)]
            close(after["ego_accel"],ea);close(after["lead_accel"],la)
            for role,a in (("ego",ea),("lead",la)):
                v=max(0,before[role+"_v"]+.01*a)
                close(after[role+"_v"],v,1e-10);close(after[role+"_x"],before[role+"_x"]+.01*v,1e-9)
                counts["vehicle_transitions"]+=1
        speedsets=[(0,30)];set_premises=True;counters={};previous_scene=None;previous_scene_ok=False;previous_scene_time=None
        ledger=[]
        errors=[];fallback=unrecoverable=changed=accepted_changed=eligible=0;set_bad=point_bad=follow_bad=0;contexts_bad=0
        for i,row in enumerate(rows):
            assert all(row[k]==trial[k] for k in ("trial_id","group","profile","case","seed","method"))
            assert row["step"]==i and row["time_s"]==(i+1)/10
            assert row["context"]["vehicle_id"]=="cv-ego"
            assert row["context"]["session_id"]==(CFG["study_id"] if not pilot else row["context"]["session_id"])
            assert row["context"]["capture_id"]==str(trial["trial_id"])+"-attempt-"+str(int((B/trial["trace"]).parent.name.split("_")[-1]))
            counts["decisions"]+=1;now=1000000+(i+1)*100000;seq=i+1;stamp=now-1000000
            inp=row["input"];out=row["output"];truth=row["truth"];ctx=row["context"];state=states[stamp]
            assert set(inp)=={"now_us","risk","packets","scene_packet","radar_packet","advisory"}
            assert inp["now_us"]==now and inp["risk"]=="normal" and inp["advisory"]==""
            for k in ("ego_v","lead_v","ego_x","lead_x"):close(truth[k],state[k],1e-9)
            gap=state["lead_x"]-state["ego_x"]-5;close(truth["gap_m"],gap,1e-9)
            active=5000000<=stamp<11000000;assert row["active"]==active
            prev=rows[i-1] if i else None;accepted_prev=prev["accepted_mmps2"] if prev else 0
            close(truth["previous_accepted_mmps2"],accepted_prev)
            pc=speed_body(inp["packets"]["candidate"],"candidate",now,ctx,seq)
            pa=speed_body(inp["packets"]["anchor"],"anchor",now,ctx,seq)
            public=row["public_profile"];assert public["run"]==expected_run
            series=PUBLIC[expected_run];idx=public["source_index"]
            assert idx==((seed*37)%len(series["age_us"])+i)%len(series["age_us"])
            assert public["age_us"]==series["age_us"][idx];close(public["innovation_mps"],series["innovation_mps"][idx])
            cv=max(0,round(state["ego_v"]*1000)+noise[i][0])
            if active and case=="candidate_drift":cv=max(0,cv-round((stamp-5000000)/6000000*3000))
            bias=None
            if active and case=="adaptive_candidate":
                bias=min(3500,max(250,500+round(max(0,accepted_prev)*1.5)));cv=max(0,cv-bias)
            assert row["attack_metadata"]["adaptive_bias_mmps"]==bias
            innovation=series["innovation_mps"][idx] if active and case=="public_motion_innovation" else None
            if innovation is not None:cv=max(0,cv+round(innovation*1000))
            assert pc["speed_mmps"]==cv and pc["measured_us"]==now
            age=90000
            if active and case=="public_anchor_age":age=None if public["age_us"] is None else math.ceil(public["age_us"]/10000)*10000
            anchor_stamp=None if age is None else stamp-age;anchor=states.get(anchor_stamp)
            assert truth["anchor_time_us"]==anchor_stamp
            ledger.append((1010000 if i==0 else now-100000,now,accepted_prev))
            commands=accepted_history(ledger,None if anchor_stamp is None else 1000000+anchor_stamp-5000,now)
            verify_command_history(inp["packets"]["command"],KEYS["command_logger"],ctx,seq,now,commands)
            if anchor is None:assert pa is None and truth["anchor_speed_mps"] is None
            else:
                close(truth["anchor_speed_mps"],anchor["ego_v"])
                av=max(0,round(anchor["ego_v"]*1000)+noise[i][1])
                if innovation is not None:av=max(0,av+round(innovation*1000))
                assert pa["speed_mmps"]==av and pa["measured_us"]==1000000+anchor_stamp
            scene,ss=REF.verify_packet("scene",inp["scene_packet"],now,ctx,counters)
            radar,rs=REF.verify_packet("radar",inp["radar_packet"],now,ctx,counters)
            assert out["scene_status"]==ss and out["radar_status"]==rs
            missing=active and (case=="scene_dropout" or (case=="burst_loss" and stamp%1000000<300000))
            assert (scene is None)==missing
            assert (radar is None)==(active and case=="burst_loss" and stamp%1000000<300000)
            if scene is not None:
                expected_gap=max(0,round(gap*1e6)+sn[i][0]);expected_lead=max(0,round(state["lead_v"]*1000)+sn[i][1])
                if active and case=="scene_disinformation":expected_gap+=12000000;expected_lead+=6000
                if active and case=="shared_disinformation":expected_gap+=30000000
                assert scene["gap_um"]==expected_gap and scene["lead_mmps"]==expected_lead
            if radar is not None:
                expected_gap=max(0,round(gap*1e6)+rn[i][0]);expected_relative=round((state["lead_v"]-state["ego_v"])*1000)+rn[i][1]
                if active and case=="radar_disinformation":expected_gap+=12000000;expected_relative+=6000
                if active and case=="shared_disinformation":expected_gap+=30000000
                assert radar["gap_um"]==expected_gap and radar["relative_mmps"]==expected_relative
            scene_ok=scene is not None and abs(scene["gap_um"]/1e6-gap)<=.00150001 and abs(scene["lead_mmps"]/1000-state["lead_v"])<=.07500001
            radar_ok=radar is not None and abs(radar["gap_um"]/1e6-gap)<=.05000001 and abs(radar["relative_mmps"]/1000-(state["lead_v"]-state["ego_v"]))<=.10000001
            pair_ok=scene_ok and previous_scene_ok and previous_scene_time is not None and now-previous_scene_time<=100000
            previous_applied=prev["next_applied_mps2"] if prev else dist[0]/1000
            exec_ok=abs(previous_applied-accepted_prev/1000)<=1+1e-10
            ed=out["estimator_decision"];value=out["estimate_mmps"];speed=None if value is None else value/1000;qualified_point=False
            if method in ("s3e_brake","ssf_zocbf"):
                observations=[];domains=[]
                for role,body in (("candidate",pc),("anchor",pa)):
                    admitted=body is not None and now-body["measured_us"]<=100000
                    assert (out["adapter_decision"][role+"_status"]=="authenticated")==admitted
                    if admitted:
                        a=(now-body["measured_us"])/1e6;r=.101 if role=="candidate" else .27
                        observations.append((max(0,body["speed_mmps"]/1000-r-4*a),body["speed_mmps"]/1000+r+2*a));domains.append(role)
                if scene is not None and previous_scene is not None and 0<now-previous_scene["measured_us"]<=100000:
                    dt=(now-previous_scene["measured_us"])/1e6
                    center=scene["lead_mmps"]/1000-(scene["gap_um"]-previous_scene["gap_um"])/1e6/dt
                    radius=.003/dt+.075+3*dt;lo=max(0,center-radius);hi=center+radius
                    if lo<=hi:observations.append((lo,hi));domains.append("scene_pair")
                close(ed["observations_mps"],observations);assert ed["observation_domains"]==domains
                speedsets=reference_sets(speedsets,observations,accepted_prev/1000,.1,method=="s3e_brake")
                close(ed["plausible_sets_mps"],speedsets,1e-7);counts["baseline_updates"]+=1
                expected_value=None if not speedsets else round(500*(speedsets[0][0]+speedsets[-1][1]))
                assert (value is None if expected_value is None else midpoint_integer_valid(value,speedsets[0][0],speedsets[-1][1])),(trial["trial_id"],i,value,expected_value)
                point_certificate=bool(speedsets) and max(abs(speed-speedsets[0][0]),abs(speed-speedsets[-1][1]))<=.5
                assert ed["point_budget_satisfied"]==point_certificate or abs(max(abs(speed-speedsets[0][0]),abs(speed-speedsets[-1][1]))-.5)<1e-8
                point_certificate=ed["point_budget_satisfied"]
                sensor_premises=len(observations)>=2 and sum(not(a-1e-8<=state["ego_v"]<=b+1e-8) for a,b in observations)<=1
                if method=="s3e_brake":set_premises=set_premises and exec_ok and sensor_premises
                else:set_premises=sensor_premises
                if set_premises:
                    contains=any(a-1e-8<=state["ego_v"]<=b+1e-8 for a,b in speedsets)
                    counts["baseline_qualified_sets"]+=1;counts["baseline_qualified_set_violations"]+=not contains;set_bad+=not contains
                    qualified_point=contains and point_certificate
                lohi=None if not speedsets else (speedsets[0][0],speedsets[-1][1])
            else:
                assert value==ed["research_speed_mmps"]
                source_ed=verify_refinement_metadata(out,pilot) if method=="no_refinement" else ed
                if source_ed.get("refined_budget_satisfied"):
                    lo=F(source_ed["refined_low_n"],source_ed["refined_low_d"]);hi=F(source_ed["refined_high_n"],source_ed["refined_high_d"])
                    prediction=float((lo+hi)/2)
                    if source_ed["residual_prediction_mps"] is not None:
                        residual=REF.forest_prediction(source_ed["relative_features"]);close(residual,source_ed["residual_prediction_mps"],1e-10);prediction+=residual*1000;counts["forest_predictions"]+=1
                    expected=min(math.floor(lo+500),max(max(0,math.ceil(hi-500)),round(prediction)))
                    assert source_ed["research_speed_mmps"]==expected
                if method=="no_refinement" and out.get("ablation_refinement_available"):
                    eligible+=1;counts["refinement_eligible_queries"]+=1
                    original=out["unrefined_original_decision"];lo=F(original["witness_low_n"],original["witness_low_d"]);hi=F(original["witness_high_n"],original["witness_high_d"])
                    low=max(0,math.ceil(hi-500));high=math.floor(lo+500)
                    expected=None if low>high or original["research_speed_mmps"] is None else min(high,max(low,round((lo+hi)/2)))
                    assert value==expected
                qualified_point=pair_ok and (exec_ok or not ed.get("refined_budget_satisfied"))
                lohi=None if speed is None else (max(0,speed-.5),speed+.5)
            error=None if speed is None else abs(speed-state["ego_v"]);close(error,row["selected_error_mps"],1e-9)
            if error is not None:
                errors.append(error)
                if qualified_point:
                    counts["point_certified_qualified"]+=1;counts["point_certified_qualified_errors"]+=error>.50000001;point_bad+=error>.50000001
            context=None
            if speed is not None and scene is not None and (radar is not None or method=="no_fusion"):
                context=reference_context(*lohi,scene,None if method=="no_fusion" else radar)
            observed=None if context is None else {"ego_mps":speed,"lead_mps":context["lead_mps"],"gap_m":context["gap_m"]}
            close(out["fused_context"],context);close(out["controller_input"],observed)
            desired=-4000 if observed is None else REF.nominal(**{"ego":speed,"lead":observed["lead_mps"],"gap":observed["gap_m"]})
            assert out["desired_mmps2"]==desired
            command=out["requested_mmps2"]
            if context is None:assert command==-4000 and out["projected_margin_m"] is None
            else:
                margin_fn=reference_cbf if method=="ssf_zocbf" else REF.full_path_margin
                margin=margin_fn(command,context["ego_high"],context["lead_low"],context["gap_low"]);close(margin,out["projected_margin_m"],1e-7)
                assert -4000<=command<=desired
                if method=="no_brake":assert command==desired
                elif margin<0:assert command==-4000
                elif command<desired:assert margin_fn(command+1,context["ego_high"],context["lead_low"],context["gap_low"])<1e-7
                contained=(context["ego_low"]-1e-8<=state["ego_v"]<=context["ego_high"]+1e-8 and context["lead_low"]-1e-8<=state["lead_v"]<=context["lead_high"]+1e-8 and context["gap_low"]-1e-8<=gap<=context["gap_high"]+1e-8)
                counts["admitted_contexts"]+=1;counts["actual_context_violations"]+=not contained;contexts_bad+=not contained
                execution=row["next_applied_mps2"]<=max(-4,min(2,command/1000+1))+1e-10
                if i<199 and margin>=0 and contained and execution:
                    segment=[states[t] for t in range(stamp,stamp+100001,10000)]
                    bad=min(x["lead_x"]-x["ego_x"]-5 for x in segment)<2-1e-8
                    counts["qualified_followups"]+=1;counts["qualified_followup_gap_violations"]+=bad;follow_bad+=bad
            assert row["delivered_command"]==expected_delivery(row["issued_command"],case,active)
            expected=expected_reason(method,speed,scene,radar,rs,context,out["projected_margin_m"],command,desired,out.get("ablation_refinement_available",False))
            assert out["reason"]==expected,(trial["trial_id"],i,out["reason"],expected)
            actor,status=REF.verify_packet("actuator",row["delivered_command"],now,ctx,counters)
            if method=="no_actuator_auth":accepted=json.loads(row["delivered_command"]["wire"])["body"]["accel_mmps2"];status="unverified_ablation"
            else:accepted=actor["accel_mmps2"] if actor is not None else -4000
            assert row["accepted_mmps2"]==accepted and row["actuator_status"]==status
            issued,issued_status=REF.verify_packet("actuator",row["issued_command"],now,ctx,{})
            assert issued_status=="authenticated" and issued["accel_mmps2"]==command
            mutation=row["issued_command"]!=row["delivered_command"];changed+=mutation;accepted_changed+=mutation and status=="unverified_ablation"
            counts["changed_actuator_frames"]+=mutation;counts["changed_frames_accepted"]+=mutation and status=="unverified_ablation"
            actual=max(-4,min(2,(accepted+(2500 if active and case=="execution_break" else dist[i+1]))/1000))
            if active and case=="lower_braking":actual=max(-2.5,actual)
            close(actual,row["next_applied_mps2"])
            card=row["operator_card"]
            assert card["reason"]==out["reason"] and card["actuator_status"]==status and card["accepted_command_mmps2"]==accepted
            close(card["controller_input"],observed)
            assert card["requested_command_mmps2"]==command and card["assurance"]=="conditional simulation evidence" and card["human_evaluation"]=="not performed"
            assert not({"truth","attack_case","label"} & set(card))
            counts["operator_cards"]+=1
            fallback+=observed is None or status not in ("authenticated","unverified_ablation")
            unrecoverable+=out["reason"]=="reserve_unrecoverable"
            if scene is not None:previous_scene=scene;previous_scene_ok=scene_ok;previous_scene_time=now
        gaps=[x["lead_x"]-x["ego_x"]-5 for x in trajectory]
        close(trial["mae_mps"],math.fsum(errors)/len(errors) if errors else None,1e-10)
        assert trial["selected"]==len(errors) and trial["over_0_5"]==sum(e>.5 for e in errors)
        close(trial["minimum_gap_m"],min(gaps));close(trial["seconds_gap_below_2m"],.01*sum(g<2 for g in gaps))
        close(trial["progress_m"],trajectory[-1]["ego_x"]-60)
        assert trial["collision_trial"]==(min(gaps)<=0)
        assert trial["collision_episodes"]==sum(g<=0 and (j==0 or gaps[j-1]>0) for j,g in enumerate(gaps))
        assert trial["fallback_queries"]==fallback and trial["unrecoverable_queries"]==unrecoverable
        assert trial["changed_actuator_frames"]==changed and trial["changed_actuator_frames_accepted"]==accepted_changed
        counts["trials"]+=1;per.append({"trial_id":trial["trial_id"],"method":method,"context_violations":contexts_bad,"qualified_set_violations":set_bad,"qualified_point_errors":point_bad,"qualified_followup_gap_violations":follow_bad,"refinement_eligible_queries":eligible})
    assert counts["baseline_qualified_set_violations"]==0 and counts["point_certified_qualified_errors"]==0 and counts["qualified_followup_gap_violations"]==0
    dump(B/("runtime/pilot_verification.json" if pilot else "verification.json"),{"passed":True,"evidence_engine":{n:sha(B/n) for n in ("verify_study.py","verification_math.py","verification_contracts.py")},"input_results_sha256":sha(result_path),"lock_digest":result["lock_digest"],"counts":counts,"per_trial":per,
      "scope":"Independent new numerical plant, packet/source mutation audit, set/filter arithmetic, manual frozen forest/projection and metrics. Frozen gateway/interval construction components retain predecessor verification; no complete independent native-stack reproduction claim."})
    print("COMPARATIVE_INDEPENDENT_VERIFICATION_PASS",json.dumps(counts),flush=True)
from pathlib import Path as PathLike
if __name__=="__main__":main()
