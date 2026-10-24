"""Independent numerical, source, observer, history, certificate and metric audit."""
from lc_common import *
import sys,json,gzip,math,random,itertools,importlib.util,hashlib,hmac,numpy as np
from fractions import Fraction as Q
from verification_math import close,union,reference_context
from verification_contracts import accepted_history,verify_command_history,expected_delivery,verify_checkpoint
spec=importlib.util.spec_from_file_location("frozen_source_reference",BASE/"verify_study.py")
REF=importlib.util.module_from_spec(spec);spec.loader.exec_module(REF)
def safe_margin(u,e,l,g,E,n=10):
    a=max(-4,min(2,u/1000+E));b=4-E
    end=max(0,e+n*.01*a)
    N=n+math.ceil(end/(b*.01))+math.ceil(l/.04)+2
    k=np.arange(1,N+1,dtype=float)
    ve=np.where(k<=n,np.maximum(0,e+a*.01*k),np.maximum(0,end-b*.01*(k-n)))
    vl=np.maximum(0,l-.04*k)
    return min(g-2,float(np.min(g+.01*np.cumsum(vl-ve)-2)))
def sets_ref(previous,obs,u,E):
    p=[(max(0,a+.1*max(-4,min(2,u/1000-E))),max(0,b+.1*max(-4,min(2,u/1000+E)))) for a,b in previous]
    matches=[]
    if len(obs)>1:
        for group in itertools.combinations(obs,len(obs)-1):
            l=max(a for a,b in group);h=min(b for a,b in group)
            if l<=h:
                for a,b in p:
                    if max(l,a)<=min(h,b):matches.append((max(l,a),min(h,b)))
    return union(matches) or union(p)
def main():
    phase=sys.argv[sys.argv.index("--phase")+1];cfg=json.loads((R/("DEVELOPMENT_PROTOCOL_V2.json" if phase=="development" else "PROTOCOL_V2.json")).read_text())
    root=R/("development_v2" if phase=="development" else "confirmation_v2");path=root/"results.json";result=json.loads(path.read_text());trials=result["trials"]
    lock=R/("development_v2_lock.json" if phase=="development" else "confirmation_v2_lock.json");assert result["lock_digest"]==sha(lock)
    assert all(sha(P/n)==h for n,h in json.loads(lock.read_text()).items())
    expected=list(itertools.product(cfg["profiles"],cfg["primary_cases"],cfg["seeds"],cfg["methods"]))
    assert len(trials)==len(expected)==cfg["expected_trials"]
    assert [(t["profile"],t["case"],t["seed"],t["method"]) for t in trials]==expected
    assert [t["trial_id"] for t in trials]==list(range(len(trials)))
    counts={k:0 for k in ("trials","decisions","vehicle_transitions","packet_macs","forest_predictions","certificates","verified_certificates","qualified_certified_queries","qualified_gap_failures","state_containment_failures","actual_state_certificate_counterexamples","point_certified_errors","changed_frames_accepted","qualified_containment_failures","qualified_certificate_counterexamples")}
    per=[]
    for t in trials:
        verify_checkpoint(root,t,result["lock_digest"])
        with gzip.open(root/t["trace"],"rt") as f:rows=[json.loads(x) for x in f]
        with gzip.open(root/t["trajectory"],"rt") as f:states=[json.loads(x) for x in f]
        assert len(rows)==t["queries"]==200 and len(states)==2000
        st={x["time_us"]:x for x in states};profile=cfg["profiles"][t["profile"]];case=t["case"];method=t["method"];seed=t["seed"];E=cfg["execution_envelopes"][method]
        initial=states[0]
        close([initial[k] for k in ("ego_v","lead_v","ego_x","lead_x")],[profile["ego_speed"],profile["lead_speed"],60,65+profile["gap"]])
        rng=random.Random(seed+10000);dist=[rng.randint(-500,500) for _ in range(201)]
        for a,b in zip(states,states[1:]):
            assert b["time_us"]==a["time_us"]+10000
            qi=a["time_us"]//100000-1
            ea=dist[0]/1000 if qi<0 else rows[qi]["next_applied_mps2"]
            if qi<0:assert b["watchdog_command_mmps2"] is None and b["watchdog_status"]=="startup"
            else:
                assert b["watchdog_command_mmps2"]==rows[qi]["accepted_mmps2"]
                expected_watchdog="watchdog_expired" if rows[qi]["delivered_command"] is None else "watchdog_active"
                assert b["watchdog_status"]==expected_watchdog

            la=profile["accelerations"][min(4,a["time_us"]//4000000)]
            close(b["ego_accel"],ea);close(b["lead_accel"],la)
            for role,acc in (("ego",ea),("lead",la)):
                v=max(0,a[role+"_v"]+.01*acc);close(b[role+"_v"],v);close(b[role+"_x"],a[role+"_x"]+.01*v);counts["vehicle_transitions"]+=1
        previous=[(0,30)];hist=None;scene_old=None;counters={};ledger=[];prev_accepted=0;prefix=True;bad_cert=0;bad_bounds=0;qualified=0;errs=[];frame_accept=0
        rng=random.Random(seed);sn=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(201)]
        rng=random.Random(seed+20000);scn=[(rng.randint(-1000,1000),rng.randint(-50,50)) for _ in range(201)]
        rng=random.Random(seed+30000);rn=[(rng.randint(-20000,20000),rng.randint(-50,50)) for _ in range(201)]
        public=json.loads((R/"public_sequences.json").read_text())[cfg["public_input_policy"]["stress_runs"][cfg["seeds"].index(seed)%2]]
        offset=(seed*37)%len(public["age_us"])
        for i,row in enumerate(rows):
            assert all(row[k]==t[k] for k in ("trial_id","profile","case","seed","method","group"))
            now=1000000+(i+1)*100000;stamp=now-1000000;truth=st[stamp];gap=truth["lead_x"]-truth["ego_x"]-5
            active=(stamp>=100000 if case=="initial_false_context" else 5000000<=stamp<11000000)
            assert row["step"]==i and row["active"]==active
            record=row["input"];out=row["output"];ctx=row["context"]
            assert set(record)=={"now_us","risk","packets","scene_packet","radar_packet","advisory"} and record["now_us"]==now
            assert ctx["session_id"]==cfg["study_id"] and ctx["capture_id"].startswith(str(t["trial_id"])+"-attempt-")
            for k in ("ego_v","lead_v","ego_x","lead_x"):close(row["truth"][k],truth[k])
            cp=REF.speed_body(record["packets"]["candidate"],"candidate",now,ctx,i+1)
            ap=REF.speed_body(record["packets"]["anchor"],"anchor",now,ctx,i+1)
            cv=max(0,round(truth["ego_v"]*1000)+sn[i][0])
            if active and case=="adaptive_candidate":cv=max(0,cv-min(3500,max(250,500+round(max(0,prev_accepted)*1.5))))
            innovation=public["innovation_mps"][(offset+i)%len(public["age_us"])] if active and case=="public_motion_innovation" else None
            if innovation is not None:cv=max(0,cv+round(innovation*1000))
            assert cp["speed_mmps"]==cv
            anchor_stamp=stamp-90000;anchor=st.get(anchor_stamp)
            if anchor is None:assert ap is None
            else:
                av=max(0,round(anchor["ego_v"]*1000)+sn[i][1])
                if innovation is not None:av=max(0,av+round(innovation*1000))
                assert ap["speed_mmps"]==av and ap["measured_us"]==anchor_stamp+1000000
            ledger.append((1010000 if i==0 else now-100000,now,prev_accepted))
            verify_command_history(record["packets"]["command"],KEYS["command_logger"],ctx,i+1,now,accepted_history(ledger,anchor_stamp+1000000-5000,now))
            scene,ss=REF.REF.verify_packet("scene",record["scene_packet"],now,ctx,counters)
            radar,rs=REF.REF.verify_packet("radar",record["radar_packet"],now,ctx,counters)
            assert out["scene_status"]==ss and out["radar_status"]==rs
            missing=active and case=="burst_loss" and stamp%1000000<300000
            assert (scene is None)==missing and (radar is None)==missing
            if scene is not None:
                sg=max(0,round(gap*1e6)+scn[i][0]);sl=max(0,round(truth["lead_v"]*1000)+scn[i][1]);rg=max(0,round(gap*1e6)+rn[i][0]);rv=round((truth["lead_v"]-truth["ego_v"])*1000)+rn[i][1]
                if active and case=="scene_disinformation":sg+=12000000;sl+=6000
                if active and case in ("shared_disinformation","combined_context_execution","initial_false_context"):sg+=30000000;rg+=30000000
                if active and case=="slow_range_drift":bias=round((stamp-5000000)/6000000*30000000);sg+=bias;rg+=bias
                assert (scene["gap_um"],scene["lead_mmps"],radar["gap_um"],radar["relative_mmps"])==(sg,sl,rg,rv)
            obs=[]
            for role,body in (("candidate",cp),("anchor",ap)):
                if body is not None and 0<=now-body["measured_us"]<=100000:
                    age=(now-body["measured_us"])/1e6;rad=.101 if role=="candidate" else .27
                    obs.append((max(0,body["speed_mmps"]/1000-rad-4*age),body["speed_mmps"]/1000+rad+2*age))
            if scene is not None and scene_old is not None and 0<now-scene_old["measured_us"]<=100000:
                dt=(now-scene_old["measured_us"])/1e6;center=scene["lead_mmps"]/1000-(scene["gap_um"]-scene_old["gap_um"])/1e6/dt
                radius=.003/dt+.075+3*dt
                if max(0,center-radius)<=center+radius:obs.append((max(0,center-radius),center+radius))
            previous=sets_ref(previous,obs,prev_accepted,E)
            close(out["estimator_decision"]["plausible_sets_mps"],previous,1e-6)
            elo,ehi=previous[0][0],previous[-1][1]
            current=None if scene is None or radar is None else reference_context(elo,ehi,scene,radar)
            if i==0:
                if current is not None:hist=dict(current)
                bootstrap=(current is not None and current["gap_low"]-1e-7<=gap<=current["gap_high"]+1e-7 and current["lead_low"]-1e-7<=truth["lead_v"]<=current["lead_high"]+1e-7)
                prefix=bootstrap
            elif hist is not None:
                for _ in range(10):
                    hist["ego_low"]=max(0,hist["ego_low"]+.01*max(-4,min(2,prev_accepted/1000-E)))
                    hist["ego_high"]=max(0,hist["ego_high"]+.01*max(-4,min(2,prev_accepted/1000+E)))
                    hist["lead_low"]=max(0,hist["lead_low"]-.04);hist["lead_high"]+=.02
                    hist["gap_low"]+=.01*(hist["lead_low"]-hist["ego_high"]);hist["gap_high"]+=.01*(hist["lead_high"]-hist["ego_low"])
                hist["ego_low"]=max(hist["ego_low"],elo);hist["ego_high"]=min(hist["ego_high"],ehi)
                hist["gap_m"]=(hist["gap_low"]+hist["gap_high"])/2;hist["lead_mps"]=(hist["lead_low"]+hist["lead_high"])/2
            history=method in ("joint","no_learning","nominal_authority")
            if i==0:refresh="bootstrap_truth_assumed" if hist is not None else "bootstrap_missing"
            elif hist is None:refresh="bootstrap_missing"
            elif current is None:refresh="missing_context"
            elif current["gap_high"]<hist["gap_low"] or hist["gap_high"]<current["gap_low"]:refresh="range_conflict"
            elif current["lead_high"]<hist["lead_low"] or hist["lead_high"]<current["lead_low"]:refresh="lead_conflict"
            else:refresh="plausible_unqualified_refresh"
            assert out["certificate"]["refresh_status"]==refresh and out["certificate"]["history_used"]==history
            bound=hist if history else current
            close(out["fused_context"],bound,2e-6)
            if scene is not None:scene_old=scene
            bad_domains=sum(not(a-1e-7<=truth["ego_v"]<=b+1e-7) for a,b in obs)
            prev_applied=dist[0]/1000 if i==0 else rows[i-1]["next_applied_mps2"]
            execution_prev=max(-4,min(2,prev_accepted/1000-E))-1e-7<=prev_applied<=max(-4,min(2,prev_accepted/1000+E))+1e-7
            prefix=prefix and bad_domains<=1 and execution_prev
            logsets=out["estimator_decision"]["plausible_sets_mps"]
            qlo,qhi=logsets[0][0],logsets[-1][1]
            proposal=out["source_output"]["estimate_mmps"]
            assert out["estimator_decision"]["learned_proposal_mmps"]==proposal
            proposed=round(500*(qlo+qhi)) if proposal is None or method in ("no_learning","always_brake") else proposal
            pl=max(0,math.ceil(qhi*1000-500));ph=math.floor(qlo*1000+500)
            expected_value=min(ph,max(pl,proposed)) if pl<=ph else min(math.floor(qhi*1000),max(math.ceil(qlo*1000),proposed))
            assert out["estimate_mmps"]==expected_value,(t["trial_id"],i,out["estimate_mmps"],expected_value)
            assert out["estimator_decision"]["point_budget_satisfied"]==(pl<=ph)
            v=out["estimate_mmps"];error=abs(v/1000-truth["ego_v"]);close(error,row["selected_error_mps"]);errs.append(error)
            d=out["source_output"]["estimator_decision"]
            if d.get("refined_budget_satisfied") and d.get("residual_prediction_mps") is not None:
                close(REF.REF.forest_prediction(d["relative_features"]),d["residual_prediction_mps"],1e-9);counts["forest_predictions"]+=1
            if prefix and out["estimator_decision"]["point_budget_satisfied"]:
                counts["point_certified_errors"]+=error>.5000001
            if bound is not None:
                contained=(bound["ego_low"]-1e-6<=truth["ego_v"]<=bound["ego_high"]+1e-6 and bound["lead_low"]-1e-6<=truth["lead_v"]<=bound["lead_high"]+1e-6 and bound["gap_low"]-1e-6<=gap<=bound["gap_high"]+1e-6)
                counts["state_containment_failures"]+=not contained;bad_bounds+=not contained
                if prefix and history:counts["qualified_containment_failures"]+=not contained
                cmd=out["requested_mmps2"];m=safe_margin(cmd,bound["ego_high"],bound["lead_low"],bound["gap_low"],E)
                assert abs(m-out["projected_margin_m"])<2e-5
                slack=1e-8*(1+abs(bound["gap_low"])+bound["ego_high"]+bound["lead_low"])
                assert -4000<=cmd<=out["desired_mmps2"]
                if out["projected_margin_m"]<0:assert cmd==-4000
                elif cmd<out["desired_mmps2"]:assert safe_margin(cmd+1,bound["ego_high"],bound["lead_low"],bound["gap_low"],E)-slack<1e-6
                n=out["certificate"]["max_hold_steps"];assert type(n) is int and 0<=n<=100
                if n:assert safe_margin(cmd,bound["ego_high"],bound["lead_low"],bound["gap_low"],E,n)-slack>=-1e-6
                if n<100:assert safe_margin(cmd,bound["ego_high"],bound["lead_low"],bound["gap_low"],E,n+1)-slack<1e-6

                if out["certificate"]["conditional_certificate"]:
                    assert m>=-1e-6
                    counter=safe_margin(cmd,truth["ego_v"],truth["lead_v"],gap,E)<-1e-6
                    bad_cert+=counter;counts["actual_state_certificate_counterexamples"]+=counter
                    if prefix and history:counts["qualified_certificate_counterexamples"]+=counter
                if prefix and history and out["certificate"]["conditional_certificate"]:
                    counts["qualified_certified_queries"]+=1;qualified+=1
                    if i<199:
                        segment=[st[x] for x in range(stamp,stamp+100001,10000)]
                        ex_ok=rows[i]["next_applied_mps2"]<=max(-4,min(2,cmd/1000+E))+1e-7
                        if ex_ok:counts["qualified_gap_failures"]+=min(x["lead_x"]-x["ego_x"]-5 for x in segment)<2-1e-6
            desired=-4000 if bound is None else REF.REF.nominal(v/1000,bound["lead_mps"],bound["gap_m"])
            if method=="always_brake":desired=-4000
            assert out["desired_mmps2"]==desired
            issued=row["issued_command"];delivered=row["delivered_command"]
            loss=active and case=="actuator_packet_loss" and stamp%1000000<300000
            assert delivered==(None if loss else issued)
            bare={k:issued[k] for k in ("wire","received_us")}
            actor,actor_status=REF.REF.verify_packet("actuator",bare,now,ctx,{} if loss else counters);assert actor_status=="authenticated" and actor["accel_mmps2"]==out["requested_mmps2"]
            cert=issued["certificate"];body=cert["body"]
            data=b"COACT-CPS-LIFECYCLE-CERTIFICATE-v1\0"+json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
            assert hmac.compare_digest(cert["tag"],hmac.new(KEYS["actuator"],data,hashlib.sha256).hexdigest())
            assert body["context"]==ctx and body["issued_us"]==now and body["seq"]==i+1 and body["expires_us"]==now+100000 and body["hold_steps"]==10
            assert body["execution_error"]==E and body["command_mmps2"]==actor["accel_mmps2"]
            assert actor["duration_us"]==body["hold_steps"]*10000 and actor["issued_us"]==body["issued_us"] and actor["seq"]==body["seq"]
            assert out["certificate"]["watchdog_period_us"]==100000
            digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            assert body["observation_sha256"]==digest and body["command_wire_sha256"]==hashlib.sha256(issued["wire"].encode()).hexdigest()
            witness={"ego_high":0.,"lead_low":0.,"gap_low":0.} if bound is None else {k:bound[k] for k in ("ego_high","lead_low","gap_low")}
            close(body["bounds"],witness,2e-6)
            gate_ok=(not loss) and safe_margin(actor["accel_mmps2"],body["bounds"]["ego_high"],body["bounds"]["lead_low"],body["bounds"]["gap_low"],E)>=0
            assert row["accepted_mmps2"]==(actor["accel_mmps2"] if gate_ok else -4000)
            assert (row["actuator_status"]=="authenticated")==gate_ok
            counts["certificates"]+=1;counts["verified_certificates"]+=gate_ok;counts["packet_macs"]+=5 if ap is not None else 4
            accepted=row["accepted_mmps2"]
            actual=max(-4,min(2,(accepted+(4500 if active and case=="authority_exceed" else 2500 if active and case in ("execution_break","combined_context_execution") else dist[i+1]))/1000))
            close(actual,row["next_applied_mps2"])
            card=row["operator_card"]
            assert card["reason"]==out["reason"] and card["certificate"]==out["certificate"] and card["controller_input"]==out["controller_input"] and card["accepted_command_mmps2"]==accepted
            assert not({"truth","attack_case","label"}&set(card)) and card["human_evaluation"]=="not performed"
            prev_accepted=accepted;counts["decisions"]+=1
        gaps=[s["lead_x"]-s["ego_x"]-5 for s in states]
        close(t["mae_mps"],sum(errs)/len(errs));assert t["selected"]==len(errs) and t["over_0_5"]==sum(e>.5 for e in errs)
        assert t["collision_trial"]==(min(gaps)<=0);close(t["minimum_gap_m"],min(gaps));close(t["progress_m"],states[-1]["ego_x"]-60)
        assert t["collision_episodes"]==sum(g<=0 and (j==0 or gaps[j-1]>0) for j,g in enumerate(gaps))
        per.append({"trial_id":t["trial_id"],"method":method,"case":case,"qualified_certified_queries":qualified,"bounds_violations":bad_bounds,"actual_certificate_counterexamples":bad_cert,"premises_held_to_end":prefix})
        counts["trials"]+=1
    assert counts["qualified_gap_failures"]==counts["point_certified_errors"]==counts["qualified_containment_failures"]==counts["qualified_certificate_counterexamples"]==0
    dump(root/"verification.json",{"passed":True,"audit_source_sha256":sha(R/"verify_v3.py"),"results_sha256":sha(path),"lock_digest":sha(lock),"counts":counts,"per_trial":per,
       "scope":"All checkpoint/grid, packet/source/command/certificate bindings, independent speed-set/history reconstruction, vectorized worst-stop margin, raw plant and summary/explanation audit. Frozen learned proposal trees manually checked; frozen upstream scene adapter separately tested."})
    print("LIFECYCLE_INDEPENDENT_AUDIT_PASS",phase,json.dumps(counts),flush=True)
if __name__=="__main__":main()
