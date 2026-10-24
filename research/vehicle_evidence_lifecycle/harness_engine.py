"""New frozen/resumable closed-loop numerical validation; old540 are never rerun."""
from lc_common import *
import json
install_guard()
import random,gzip,csv,time,math,itertools,argparse,os,contextlib
from packet_policy import LifecyclePolicy as ComparisonPolicy
from plant import step,public_window
from checkpoints import completed_trial,write_checkpoint
from control_core import seal
from scene_core import sign_scene
import sumo_bridge as bridge
cg=bridge.cg;p=cg.p;ce=cg.ce
CONFIG=json.loads((B/"PROTOCOL.json").read_text())
PUBLIC=json.loads((B/"public_sequences.json").read_text())
EPOCH=1000000
def speed_packet(role,value,seq,stamp,now,ctx):
    body={"schema":1,"source_id":role,"key_id":"c-key" if role=="candidate" else "a-key",**ctx,
      "channel":"wheel_speed" if role=="candidate" else "protected_speed","seq":seq,"measured_us":stamp,"speed_mmps":value}
    return {"wire":json.dumps(p.auth.sign(body,KEYS[role]),separators=(",",":")),"received_us":now}
def corrupt(packet,field,value):
    obj=json.loads(packet["wire"]);obj["body"][field]=value
    return {**packet,"wire":json.dumps(obj,separators=(",",":"))}
def grid():
    jobs=[]
    for group,cases,methods in [("primary",CONFIG["primary_cases"],CONFIG["methods"]),("stress",CONFIG["stress_cases"],CONFIG["stress_methods"])]:
        for profile,case,seed,method in itertools.product(CONFIG["profiles"],cases,CONFIG["seeds"],methods):
            jobs.append({"trial_id":len(jobs),"group":group,"profile":profile,"case":case,"seed":seed,"method":method})
    assert len(jobs)==CONFIG["expected_trials"];return jobs
def completed_job(root,tid,digest):
    found=[]
    for a in sorted(root.glob("attempt_*")) if root.exists() else []:
        result=completed_trial(a,tid,digest)
        if result is not None:found.append(result)
    if len(found)>1:raise ValueError("duplicate completed attempts")
    return found[0] if found else None
def run_trial(task):
    job,pilot_id,digest=task;tid=job["trial_id"];profile=job["profile"];case=job["case"];seed=job["seed"];method=job["method"]
    base=B/("runtime/"+pilot_id if pilot_id else "confirmation")/f"trial_{tid:04d}"
    existing=completed_job(base,tid,digest)
    if existing is not None:return existing
    base.mkdir(parents=True,exist_ok=True);attempt=len(list(base.glob("attempt_*")))+1
    root=base/f"attempt_{attempt:03d}";root.mkdir()
    ctx={"vehicle_id":"cv-ego","session_id":pilot_id or CONFIG["study_id"],"capture_id":str(tid)+"-attempt-"+str(attempt)}
    state_dir=P/"research"/"project_release"/"runtime"/("cv-"+ctx["session_id"]+"-"+ctx["capture_id"])
    policy=ComparisonPolicy(state_dir,ctx,KEYS,method)
    rng=random.Random(seed);speed_noise=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(201)]
    rng=random.Random(seed+10000);disturbance=[rng.randint(-500,500) for _ in range(201)]
    rng=random.Random(seed+20000);scene_noise=[(rng.randint(-1000,1000),rng.randint(-50,50)) for _ in range(201)]
    rng=random.Random(seed+30000);radar_noise=[(rng.randint(-20000,20000),rng.randint(-50,50)) for _ in range(201)]
    source_name=CONFIG["public_input_policy"]["stress_runs"][CONFIG["seeds"].index(seed)%2] if seed in CONFIG["seeds"] else "heldout_medium"
    source=PUBLIC[source_name];offset=(seed*37)%len(source["age_us"]);public=public_window(source,200,offset)
    cfg=CONFIG["profiles"][profile]
    state={"ego_v":cfg["ego_speed"],"lead_v":cfg["lead_speed"],"ego_x":60.,"lead_x":65.+cfg["gap"]}
    history={10000:dict(state)};trajectory=[{"time_us":10000,**state,"ego_accel":None,"lead_accel":None}]
    ledger=[];rows=[];previous_stamp=10000;accepted_previous=0;applied=disturbance[0]/1000
    for stamp in range(20000,20_000_001,10000):
        previous_us=stamp-10000;lead_accel=cfg["accelerations"][min(4,previous_us//4000000)]
        state=step(state,applied,lead_accel);history[stamp]=dict(state)
        trajectory.append({"time_us":stamp,**state,"ego_accel":applied,"lead_accel":lead_accel})
        if stamp%100000:continue
        i=stamp//100000-1;seq=i+1;now=EPOCH+stamp;active=(stamp>=100000 if case=="initial_false_context" else 5_000_000<=stamp<11_000_000)
        ledger.append((EPOCH+previous_stamp,now,accepted_previous))
        age_us=90000
        if case=="public_anchor_age" and active:
            raw_age=public["age_us"][i];age_us=None if raw_age is None else math.ceil(raw_age/10000)*10000
        anchor_stamp=None if age_us is None else stamp-age_us
        anchor_state=history.get(anchor_stamp)
        candidate=max(0,round(state["ego_v"]*1000)+speed_noise[i][0])
        avalue=None if anchor_state is None else max(0,round(anchor_state["ego_v"]*1000)+speed_noise[i][1])
        if active and case=="candidate_drift":candidate=max(0,candidate-round((stamp-5_000_000)/6_000_000*3000))
        adaptive_bias=None
        if active and case=="adaptive_candidate":
            adaptive_bias=min(3500,max(250,500+round(max(0,accepted_previous)*1.5)))
            candidate=max(0,candidate-adaptive_bias)
        innovation=public["innovation_mps"][i] if active and case=="public_motion_innovation" else None
        if innovation is not None:
            candidate=max(0,candidate+round(innovation*1000))
            if avalue is not None:avalue=max(0,avalue+round(innovation*1000))
        cp=speed_packet("candidate",candidate,seq,now,now,ctx)
        ap=None if avalue is None else speed_packet("anchor",avalue,seq,EPOCH+anchor_stamp,now,ctx)
        try:commands=bridge.history_window(ledger,EPOCH+anchor_stamp-5000,now) if anchor_stamp is not None else []
        except ValueError:commands=[]
        hb={"schema":1,"source_id":"command_logger","key_id":"h-key",**ctx,"channel":"command_history","seq":seq,"issued_us":now,
            "commands":[[a,b,c,1] for a,b,c in commands]}
        hp={"wire":json.dumps(ce.sign(hb,KEYS["command_logger"]),separators=(",",":")),"received_us":now}
        gap=state["lead_x"]-state["ego_x"]-5
        sb={"schema":1,"source_id":"scene","key_id":"s-key",**ctx,"channel":"lead_range","seq":seq,"measured_us":now,
            "gap_um":max(0,round(gap*1e6)+scene_noise[i][0]),"lead_mmps":max(0,round(state["lead_v"]*1000)+scene_noise[i][1])}
        rb={"schema":1,**ctx,"seq":seq,"measured_us":now,"gap_um":max(0,round(gap*1e6)+radar_noise[i][0]),
            "relative_mmps":round((state["lead_v"]-state["ego_v"])*1000)+radar_noise[i][1]}
        if active and case=="scene_disinformation":sb["gap_um"]+=12000000;sb["lead_mmps"]+=6000
        if active and case=="radar_disinformation":rb["gap_um"]+=12000000;rb["relative_mmps"]+=6000
        if active and case in ("shared_disinformation","combined_context_execution","initial_false_context"):sb["gap_um"]+=30000000;rb["gap_um"]+=30000000
        if active and case=="slow_range_drift":
            drift=round((stamp-5_000_000)/6_000_000*30000000);sb["gap_um"]+=drift;rb["gap_um"]+=drift
        sp={"wire":json.dumps(sign_scene(sb,KEYS["scene"]),separators=(",",":")),"received_us":now}
        rp=seal("radar",rb,KEYS["radar"],now)
        if active and case=="scene_dropout":sp=None
        if active and case=="burst_loss" and stamp%1000000<300000:sp=None;rp=None
        record={"now_us":now,"risk":"normal","packets":{"candidate":cp,"anchor":ap,"command":hp},"scene_packet":sp,"radar_packet":rp,"advisory":""}
        start=time.perf_counter();out=policy.process(record)
        issued=policy.command_packet(out["requested_mmps2"],now,seq)
        delivered=corrupt(issued,"accel_mmps2",2000) if active and case=="actuator_tamper" else issued
        accepted,status=policy.actuate(delivered,now)
        applied=max(-4,min(2,(accepted+(4500 if active and case=="authority_exceed" else 2500 if active and case in ("execution_break","combined_context_execution") else disturbance[i+1]))/1000))
        if active and case=="lower_braking":applied=max(-2.5,applied)
        elapsed=time.perf_counter()-start
        row={**job,"step":i,"time_s":stamp/1e6,"context":ctx,"active":active,"input":record,"output":out,
            "issued_command":issued,"delivered_command":delivered,"accepted_mmps2":accepted,"next_applied_mps2":applied,
            "actuator_status":status,"operator_card":policy.card(out,accepted,status),"decision_wall_seconds":elapsed,
            "selected_error_mps":None if out["estimate_mmps"] is None else abs(out["estimate_mmps"]/1000-state["ego_v"]),
            "public_profile":{"run":source_name,"source_index":(offset+i)%len(source["age_us"]),"age_us":public["age_us"][i],"innovation_mps":public["innovation_mps"][i]},
            "attack_metadata":{"adaptive_bias_mmps":adaptive_bias},"truth":{**state,"gap_m":gap,"anchor_time_us":anchor_stamp,
                "anchor_speed_mps":None if anchor_state is None else anchor_state["ego_v"],"previous_accepted_mmps2":accepted_previous}}
        rows.append(row);previous_stamp=stamp;accepted_previous=accepted
    assert len(rows)==200 and len(trajectory)==2000
    trace=root/"trace.jsonl.gz";traj=root/"trajectory.jsonl.gz"
    for path,objects in ((trace,rows),(traj,trajectory)):
        with gzip.open(path,"wt",encoding="utf-8",compresslevel=3) as f:
            for r in objects:f.write(json.dumps(r,separators=(",",":"),allow_nan=False)+"\n")
    gaps=[r["lead_x"]-r["ego_x"]-5 for r in trajectory]
    errors=[r["selected_error_mps"] for r in rows if r["selected_error_mps"] is not None]
    summary={**job,"queries":len(rows),"selected":len(errors),"mae_mps":math.fsum(errors)/len(errors) if errors else None,
        "over_0_5":sum(e>.5 for e in errors),"minimum_gap_m":min(gaps),"seconds_gap_below_2m":sum(g<2 for g in gaps)*.01,
        "collision_trial":min(gaps)<=0,"collision_episodes":sum(g<=0 and (j==0 or gaps[j-1]>0) for j,g in enumerate(gaps)),
        "progress_m":trajectory[-1]["ego_x"]-60,"fallback_queries":sum(r["output"]["controller_input"] is None or r["actuator_status"] not in ("authenticated","unverified_ablation") for r in rows),
        "unrecoverable_queries":sum(r["output"]["reason"]=="reserve_unrecoverable" for r in rows),
        "changed_actuator_frames":sum(r["issued_command"]!=r["delivered_command"] for r in rows),
        "changed_actuator_frames_accepted":sum(r["issued_command"]!=r["delivered_command"] and r["actuator_status"]=="unverified_ablation" for r in rows),
        "maximum_decision_seconds":max(r["decision_wall_seconds"] for r in rows),
        "trace":str(trace.relative_to(B)),"trajectory":str(traj.relative_to(B))}
    write_checkpoint(root,summary,digest,[trace,traj]);return summary
