"""Joint lifecycle policy; all safety claims are conditional on the recorded contract."""
from lc_common import *
from policy import ComparisonPolicy,context_bounds,observation_interval
from baseline_core import predict_sets,agreement,intersection
from control_core import idm
from lifecycle_core import ContextTube,choose,lease_steps,margin
from certificate import seal_certificate,verify_certificate
import json,hashlib,math
METHODS=("joint","robust_current","nominal_authority","no_learning","always_brake")
class LifecyclePolicy:
    def __init__(self,state_directory,context,keys,method):
        if method not in METHODS:raise ValueError("method")
        self.method=method;self.context=dict(context);self.keys=keys
        self.source=ComparisonPolicy(state_directory,context,keys,"full")
        self.error=1. if method=="nominal_authority" else 2.5
        self.sets=[(0.,30.)];self.tube=None;self.last=None;self.accepted=0;self.pending=None;self.seen=-1
    def process(self,record):
        source=self.source.process(record);now=record["now_us"]
        d=source["estimator_decision"];obs=[];domains=[]
        dt=.1 if self.last is None else (now-self.last)/1e6
        prediction=predict_sets(self.sets,self.accepted/1000,dt,self.error)
        for role in ("candidate","anchor"):
            packet=record["packets"][role]
            if d.get(role+"_status")!="authenticated" or packet is None:continue
            body=self.source.inner.raw_body(packet);age=(now-body["measured_us"])/1e6
            if age<0 or (role=="candidate" and age!=0):continue
            obs.append(observation_interval(role,body["speed_mmps"]/1000,age));domains.append(role)
        if d.get("witness_low_n") is not None:
            obs.append((d["witness_low_n"]/d["witness_low_d"]/1000,d["witness_high_n"]/d["witness_high_d"]/1000));domains.append("scene_pair")
        measured=agreement(obs,1)
        narrowed=intersection(prediction,measured)
        self.sets=narrowed or prediction
        update="secure_measurement_update" if narrowed else "prediction_only"
        elo,ehi=self.sets[0][0],self.sets[-1][1]
        center=round(500*(elo+ehi))
        proposed=source["estimate_mmps"]
        desired_point=center if proposed is None or self.method in ("no_learning","always_brake") else proposed
        admissible_lo=max(0,math.ceil(ehi*1000-500));admissible_hi=math.floor(elo*1000+500)
        point_cert=admissible_lo<=admissible_hi
        if point_cert:value=min(admissible_hi,max(admissible_lo,desired_point))
        else:value=min(math.floor(ehi*1000),max(math.ceil(elo*1000),desired_point))
        scene=self.source.inner.raw_body(record["scene_packet"]) if source["scene_status"]=="authenticated" else None
        radar=self.source.inner.raw_body(record["radar_packet"]) if source["radar_status"]=="authenticated" else None
        current=context_bounds(elo,ehi,scene,radar) if scene is not None and radar is not None else None
        history=self.method in ("joint","no_learning","nominal_authority")
        if self.last is None:
            if current is not None:self.tube=ContextTube((elo,ehi),(current["lead_low"],current["lead_high"]),(current["gap_low"],current["gap_high"]))
            refresh="bootstrap_truth_assumed" if self.tube is not None else "bootstrap_missing"
        else:
            if self.tube is not None:
                self.tube.advance(self.accepted,dt,self.error)
                assert self.tube.tighten_ego((elo,ehi)),"ego prediction disagreement"
                refresh=self.tube.observe(None if current is None else (current["gap_low"],current["gap_high"]),None if current is None else (current["lead_low"],current["lead_high"]))
            else:refresh="bootstrap_missing"
        bounds=self.tube.bounds() if history and self.tube is not None else current if not history else None
        desired=-4000 if bounds is None else idm(value/1000,bounds["lead_mps"],bounds["gap_m"])
        if self.method=="always_brake":desired=-4000
        if bounds is None:command=-4000;m=None;valid_steps=0
        else:
            command,m=choose(desired,bounds["ego_high"],bounds["lead_low"],bounds["gap_low"],self.error)
            valid_steps=lease_steps(command,bounds["ego_high"],bounds["lead_low"],bounds["gap_low"],self.error)
        observed=None if bounds is None else {"ego_mps":value/1000,"lead_mps":bounds["lead_mps"],"gap_m":bounds["gap_m"]}
        witness={"ego_high":0.,"lead_low":0.,"gap_low":0.} if bounds is None else {k:bounds[k] for k in ("ego_high","lead_low","gap_low")}
        digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.pending={"bounds":witness,"digest":digest,"now":now,"command":command}
        self.last=now
        return {"method":self.method,"mode":self.method,"now_us":now,"estimate_mmps":value,
          "estimator_decision":{"plausible_sets_mps":self.sets,"point_budget_satisfied":point_cert,"observation_domains":domains,"update":update,"learned_proposal_mmps":proposed},
          "source_output":source,"scene_status":source["scene_status"],"radar_status":source["radar_status"],
          "controller_input":observed,"fused_context":bounds,"desired_mmps2":desired,"requested_mmps2":command,"projected_margin_m":m,
          "reason":"bootstrap_missing" if bounds is None else "reserve_unrecoverable" if m<0 else "history_limited" if history and command<desired else "certificate_authorized",
          "certificate":{"execution_error":self.error,"max_hold_steps":valid_steps,"issued_hold_steps":10,"observation_sha256":digest,
             "refresh_status":refresh,"history_used":history,"bootstrap_truth_established":False,"conditional_certificate":m is not None and m>=0},
          "simulation_only":True}
    def command_packet(self,command,now,seq):
        if self.pending is None or now!=self.pending["now"]:raise ValueError("no fresh evidence")
        packet=self.source.command_packet(command,now,seq)
        packet["certificate"]=seal_certificate(packet["wire"],self.keys["actuator"],self.context,seq,now,command,self.pending["bounds"],self.error,10,self.pending["digest"])
        return packet
    def actuate(self,packet,now):
        bare=None if packet is None else {k:packet[k] for k in ("wire","received_us") if k in packet}
        accepted,status=self.source.actuate(bare,now)
        if status=="authenticated":
            good,why=verify_certificate(packet.get("certificate"),packet["wire"],self.keys["actuator"],self.context,now,self.pending["bounds"],self.error,self.pending["digest"],self.seen)
            if good:self.seen=json.loads(packet["wire"])["body"]["seq"]
            else:accepted=-4000;status="certificate_"+why
        self.accepted=accepted
        return accepted,status
    def card(self,out,accepted,status):
        return {"time_us":out["now_us"],"reason":out["reason"],"scene_status":out["scene_status"],"radar_status":out["radar_status"],
          "controller_input":out["controller_input"],"requested_command_mmps2":out["requested_mmps2"],"accepted_command_mmps2":accepted,"actuator_status":status,
          "projected_margin_m":out["projected_margin_m"],"certificate":dict(out["certificate"]),"point_budget_satisfied":out["estimator_decision"]["point_budget_satisfied"],
          "assurance":"conditional simulation evidence","human_evaluation":"not performed"}
