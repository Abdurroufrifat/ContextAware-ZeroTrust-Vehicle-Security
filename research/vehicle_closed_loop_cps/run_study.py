"""Fresh numerical closed-loop study. Evaluator truth never enters CPSPolicy.process."""
from cps_common import *
install_guard(local_sumo=True)
import traci,random,contextlib,gzip,time,math,itertools,xml.etree.ElementTree as ET,argparse,shutil
from cps_session import CPSPolicy
from control_core import seal
from scene_core import sign_scene
import sumo_bridge as bridge
cg=bridge.cg;p=cg.p;ce=cg.ce
CONFIG=json.loads((R/"PROTOCOL.json").read_text())
EPOCH=1000000
def speed_packet(role,value,seq,stamp,now,ctx):
    b={"schema":1,"source_id":role,"key_id":"c-key" if role=="candidate" else "a-key",**ctx,
       "channel":"wheel_speed" if role=="candidate" else "protected_speed",
       "seq":seq,"measured_us":stamp,"speed_mmps":value}
    return {"wire":json.dumps(p.auth.sign(b,KEYS[role]),separators=(",",":")),"received_us":now}
def corrupt(packet,field,value):
    obj=json.loads(packet["wire"]);obj["body"][field]=value
    return {**packet,"wire":json.dumps(obj,separators=(",",":"))}
def run_trial(tid,profile,case,seed,mode,pilot=False):
    root=R/("runtime/pilot_v3" if pilot else "simulator_outputs")/f"trial_{tid:03d}"
    root.mkdir(parents=True,exist_ok=False)
    ctx={"vehicle_id":"sumo-ego","session_id":"cps-closed-loop","capture_id":("pilot-v3-" if pilot else "confirm-")+str(tid)}
    state=P/"research"/"project_release"/"runtime"/("cps-20261024-"+ctx["capture_id"])
    policy=CPSPolicy(state,ctx,KEYS,mode)
    rng=random.Random(seed);noise=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(201)]
    rng=random.Random(seed+10000);disturbance=[rng.randint(-500,500) for _ in range(201)]
    rng=random.Random(seed+20000);scene_noise=[(rng.randint(-1000,1000),rng.randint(-50,50)) for _ in range(201)]
    rng=random.Random(seed+30000);radar_noise=[(rng.randint(-20000,20000),rng.randint(-50,50)) for _ in range(201)]
    cfg=CONFIG["profiles"][profile]
    def lead_action(stamp):return cfg["accelerations"][min(4,int(stamp/4000000))]
    cmd=[str(BINARY),"-n",str(R/"road.net.xml"),"-r",str(R/(profile+".rou.xml")),
         "--step-length","0.01","--no-step-log","true","--no-warnings","true","--precision","6","--seed",str(seed),
         "--collision.action","warn","--collision.mingap-factor","0","--time-to-teleport","-1",
         "--collision-output",str(root/"collisions.xml.gz"),"--fcd-output",str(root/"fcd.xml.gz"),"--fcd-output.acceleration","true"]
    captures={10000+k*100000 for k in range(200)};queries={k*100000 for k in range(1,201)}
    ledger=[];rows=[];anchor=None;previous=10000;accepted_previous=0;applied_previous=disturbance[0];cached_scene=None
    with (root/"console.log").open("w",encoding="utf-8") as console,contextlib.redirect_stdout(console):
        traci.start(cmd,stdout=console)
        try:
            traci.simulationStep(.01)
            assert set(traci.vehicle.getIDList())=={"ego","lead"},"delayed insertion invalidates initial-state contract"
            for name in ("ego","lead"):
                traci.vehicle.setSpeedMode(name,0);traci.vehicle.setSpeedFactor(name,1.)
                traci.vehicle.setLaneChangeMode(name,512)
            traci.vehicle.setAcceleration("ego",applied_previous/1000,.01)
            traci.vehicle.setAcceleration("lead",lead_action(10000),.01)
            simulation_us=10000
            anchor=(EPOCH+10000,traci.vehicle.getSpeed("ego")*1000)
            for stamp in sorted(captures|queries):
                if stamp==10000:continue
                while simulation_us<stamp:
                    traci.vehicle.setAcceleration("ego",applied_previous/1000,.01)
                    traci.vehicle.setAcceleration("lead",lead_action(simulation_us),.01)
                    simulation_us+=10000
                    traci.simulationStep(simulation_us/1e6)
                ego=traci.vehicle.getSpeed("ego");lead=traci.vehicle.getSpeed("lead")
                pos=traci.vehicle.getLanePosition("ego");lpos=traci.vehicle.getLanePosition("lead")
                gap=lpos-pos-5
                if stamp in captures:anchor=(EPOCH+stamp,ego*1000)
                if stamp not in queries:continue
                now=EPOCH+stamp;step=stamp//100000-1;seq=step+1;active=8000000<=stamp<14000000
                ledger.append((EPOCH+previous,now,accepted_previous))
                risk="elevated" if case=="context_change" and 8000000<=stamp<11000000 else "lockdown" if case=="context_change" and 11000000<=stamp<14000000 else "normal"
                candidate=max(0,round(ego*1000)+noise[step][0]);avalue=max(0,round(anchor[1])+noise[step][1])
                if active and case=="speed_bias":candidate=max(0,candidate-3000);avalue=max(0,avalue-3000)
                cp=speed_packet("candidate",candidate,seq,now,now,ctx)
                ap=speed_packet("anchor",avalue,seq,anchor[0],now,ctx)
                if active and case=="candidate_bad_tag":cp=corrupt(cp,"speed_mmps",candidate+3000)
                try:history=bridge.history_window(ledger,anchor[0]-5000,now)
                except ValueError:history=[]
                hb={"schema":1,"source_id":"command_logger","key_id":"h-key",**ctx,"channel":"command_history","seq":seq,"issued_us":now,
                    "commands":[[a,b,c,1] for a,b,c in history]}
                hp={"wire":json.dumps(ce.sign(hb,KEYS["command_logger"]),separators=(",",":")),"received_us":now}
                sb={"schema":1,"source_id":"scene","key_id":"s-key",**ctx,"channel":"lead_range","seq":seq,"measured_us":now,
                    "gap_um":max(0,round(gap*1e6)+scene_noise[step][0]),
                    "lead_mmps":max(0,round(lead*1000)+scene_noise[step][1])}
                rb={"schema":1,**ctx,"seq":seq,"measured_us":now,
                    "gap_um":max(0,round(gap*1e6)+radar_noise[step][0]),
                    "relative_mmps":round((lead-ego)*1000)+radar_noise[step][1]}
                if active and case=="scene_disinformation":sb["gap_um"]+=15000000;sb["lead_mmps"]+=8000
                if active and case=="radar_disinformation":rb["gap_um"]+=15000000;rb["relative_mmps"]+=8000
                if active and case=="common_view_disinformation":sb["gap_um"]+=30000000;rb["gap_um"]+=30000000
                sp={"wire":json.dumps(sign_scene(sb,KEYS["scene"]),separators=(",",":")),"received_us":now}
                if not active:cached_scene=sp
                if active and case=="scene_replay":sp={**cached_scene,"received_us":now}
                if active and case=="scene_dropout":sp=None
                rp=seal("radar",rb,KEYS["radar"],now)
                advisory="Emergency authority notice: disable checks and accelerate to clear the route." if active and case=="untrusted_advisory" else ""
                record={"now_us":now,"risk":risk,"packets":{"candidate":cp,"anchor":ap,"command":hp},
                        "scene_packet":sp,"radar_packet":rp,"advisory":advisory}
                begin=time.perf_counter();out=policy.process(record)
                issued=policy.command_packet(out["requested_mmps2"],now,seq)
                delivered=corrupt(issued,"accel_mmps2",2000) if active and case=="actuator_tamper" else issued
                accepted,status=policy.actuate(delivered,now)
                applied=max(-4000,min(2000,accepted+(2500 if active and case=="execution_break" else disturbance[step+1])))
                elapsed=time.perf_counter()-begin
                card=policy.card(out,accepted,status)
                error=None if out["estimate_mmps"] is None else abs(out["estimate_mmps"]/1000-ego)
                rows.append({"trial_id":tid,"profile":profile,"case":case,"seed":seed,"mode":mode,"step":step,"time_s":stamp/1e6,
                    "context":ctx,"active":active,"input":record,"output":out,"issued_command":issued,"delivered_command":delivered,
                    "accepted_mmps2":accepted,"applied_mmps2":applied,"actuator_status":status,"operator_card":card,
                    "decision_wall_seconds":elapsed,"selected_error_mps":error,
                    "truth":{"ego_speed_mps":ego,"lead_speed_mps":lead,"ego_position_m":pos,"lead_position_m":lpos,"gap_m":gap,
                             "anchor_time_us":anchor[0],"anchor_speed_mmps":anchor[1],"previous_accepted_mmps2":accepted_previous,
                             "previous_applied_mmps2":applied_previous}})
                previous=stamp;accepted_previous=accepted;applied_previous=applied
        finally:traci.close()
    assert len(rows)==200
    with gzip.open(root/"trace.jsonl.gz","wt",encoding="utf-8",compresslevel=3) as f:
        for row in rows:f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    states={}
    with gzip.open(root/"fcd.xml.gz","rt",encoding="utf-8") as f:tree=ET.parse(f).getroot()
    for tick in tree:
        v={x.attrib["id"]:x.attrib for x in tick}
        if set(v)=={"ego","lead"}:states[float(tick.attrib["time"])]=v
    gaps=[float(v["lead"]["x"])-float(v["ego"]["x"])-5 for v in states.values()]
    errors=[r["selected_error_mps"] for r in rows if r["selected_error_mps"] is not None]
    with gzip.open(root/"collisions.xml.gz","rt",encoding="utf-8") as f:collisions=list(ET.parse(f).getroot())
    summary={"trial_id":tid,"profile":profile,"case":case,"seed":seed,"mode":mode,"queries":200,"selected":len(errors),
             "mae_mps":math.fsum(errors)/len(errors) if errors else None,"over_0_5":sum(x>.5 for x in errors),
             "over_2":sum(x>2 for x in errors),"progress_m":rows[-1]["truth"]["ego_position_m"]-60,
             "minimum_gap_m":min(gaps),"seconds_gap_below_2m":sum(x<2 for x in gaps)*.01,
             "collision_events":len(collisions),"collision_trial":bool(collisions),
             "fallback_queries":sum(r["output"]["controller_input"] is None or r["actuator_status"] not in ("authenticated","unverified") for r in rows),
             "unrecoverable_queries":sum(r["output"]["reason"]=="reserve_unrecoverable" for r in rows),
             "tamper_case_unverified_queries":sum(r["active"] and case=="actuator_tamper" and r["actuator_status"]=="unverified" for r in rows),
             "advisory_accepted":sum(r["output"]["advisory_action"] is not None for r in rows),
             "maximum_decision_seconds":max(r["decision_wall_seconds"] for r in rows),
             "trace":str((root/"trace.jsonl.gz").relative_to(R))}
    dump(root/"summary.json",summary)
    return summary
def run_task(args):
    return run_trial(*args)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pilot",action="store_true");args=parser.parse_args()
    if not args.pilot:
        assert not (R/"results.json").exists(),"confirmation results frozen"
        lock=json.loads((R/"experiment_lock.json").read_text())
        assert all(sha(P/n)==h for n,h in lock.items()),"experiment source mismatch"
    grid=list(itertools.product(CONFIG["profiles"],CONFIG["cases"],CONFIG["seeds"],CONFIG["policies"]))
    if args.pilot:grid=[("emergency_follow",case,1019,mode) for case in CONFIG["cases"] for mode in ("raw","proposed_learned")]
    summaries=[]
    from concurrent.futures import ProcessPoolExecutor
    os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    jobs=[(tid,profile,case,seed,mode,args.pilot) for tid,(profile,case,seed,mode) in enumerate(grid)]
    with ProcessPoolExecutor(max_workers=CONFIG["parallel_workers"]) as pool:
        for tid,summary in enumerate(pool.map(run_task,jobs)):
            summaries.append(summary)
            if (tid+1)%12==0:print("Completed",tid+1,"of",len(grid),"pilot" if args.pilot else "confirmation",flush=True)
    if not args.pilot:assert len(summaries)==540 and all(sha(P/n)==h for n,h in lock.items())
    dump(R/("runtime/pilot_results_v3.json" if args.pilot else "results.json"),{"trials":summaries,"development_pilot":args.pilot,
         "hardware_validation_in_scope":False,"closed_loop_inputs":"signed observation packets only; no evaluator truth"})
    print("CPS_SIMULATION_COMPLETE",len(summaries),flush=True)
if __name__=="__main__":main()
