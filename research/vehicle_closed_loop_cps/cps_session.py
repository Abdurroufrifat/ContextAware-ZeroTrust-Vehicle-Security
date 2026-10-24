"""Simulation policy: inputs are signed observations; evaluator truth is forbidden."""
from cps_common import P,KEYS
import json,sys,math
from control_core import PacketGate,seal,idm,fuse_context,safe_command,advisory_action
from scene_core import SceneGuard
from command_scene_session import CommandSceneSession
sys.path.insert(0,str(P/"research"/"project_release"))
from research_session import ResearchSession
MODES=("raw","sensor_auth","learned","proposed_midpoint","proposed_learned")
class CPSPolicy:
    def __init__(self,state_directory,context,keys,mode):
        if mode not in MODES or set(keys)!=set(KEYS):raise ValueError("mode/keys")
        if len(set(keys.values()))!=len(keys):raise ValueError("keys must differ")
        self.mode=mode;self.context=dict(context);self.keys=dict(keys);self.last=-1
        self.scene_gate=SceneGuard(keys["scene"],context)
        self.radar_gate=PacketGate("radar",keys["radar"],context)
        self.actuator_gate=PacketGate("actuator",keys["actuator"],context)
        base_keys={n:keys[n] for n in ("candidate","anchor","command_logger","scene")}
        if mode=="raw":self.estimator=None
        elif mode=="sensor_auth":self.estimator=ResearchSession(state_directory,context,base_keys,mode="kinematic")
        else:self.estimator=CommandSceneSession(state_directory,context,base_keys,mode="refined_midpoint" if mode=="proposed_midpoint" else "refined_learned")
    @staticmethod
    def raw_body(packet):
        try:
            obj=json.loads(packet["wire"])
            return obj["body"] if type(obj["body"]) is dict else None
        except (KeyError,TypeError,ValueError):return None
    def process(self,record):
        if type(record) is not dict or set(record)!={"now_us","risk","packets","scene_packet","radar_packet","advisory"}:raise ValueError("packet-only input schema")
        now=record["now_us"];risk=record["risk"]
        if type(now) is not int or now<=self.last or risk not in ("normal","elevated","lockdown"):raise ValueError("time/context")
        if type(record["advisory"]) is not str or len(record["advisory"])>512:raise ValueError("advisory")
        if type(record["packets"]) is not dict or set(record["packets"])!={"candidate","anchor","command"}:raise ValueError("source schema")
        self.last=now
        scene,ss=self.scene_gate.verify(record["scene_packet"],now)
        radar,rs=self.radar_gate.verify(record["radar_packet"],now)
        if self.estimator is None:
            body=self.raw_body(record["packets"]["candidate"])
            value=body.get("speed_mmps") if body else None
            if type(value) is not int or value<0:value=None
            decision=None;scene=self.raw_body(record["scene_packet"])
            ss="unverified" if scene is not None else ss
        else:
            decision=self.estimator.process({k:record[k] for k in ("now_us","risk","packets","scene_packet")})
            value=decision["research_speed_mmps"]
        speed=value/1000 if value is not None else None
        observed=None;fused=None;margin=None;desired=-4000;requested=-4000;reason="observation_unavailable"
        proposed=self.mode.startswith("proposed")
        if speed is not None and scene is not None:
            if proposed:
                fused=fuse_context(speed,scene,radar)
                if fused is None:reason="context_disagreement" if radar is not None else "radar_"+rs
                else:
                    observed={"ego_mps":speed,"lead_mps":fused["lead_mps"],"gap_m":fused["gap_m"]}
                    desired=idm(**{"ego":speed,"lead":fused["lead_mps"],"gap":fused["gap_m"]})
                    requested,margin=safe_command(desired,fused["ego_high"],fused["lead_low"],fused["gap_low"])
                    reason="reserve_unrecoverable" if margin<0 else "reserve_limited" if requested!=desired else "observed_context_accepted"
            else:
                observed={"ego_mps":speed,"lead_mps":scene["lead_mmps"]/1000,"gap_m":scene["gap_um"]/1e6}
                desired=requested=idm(speed,observed["lead_mps"],observed["gap_m"])
                reason="raw_observations" if self.mode=="raw" else "authenticated_scene_control"
        override=advisory_action(self.mode,record["advisory"])
        if override is not None:requested=override;reason="untrusted_advisory_accepted"
        return {"mode":self.mode,"now_us":now,"risk_context":risk,"estimate_mmps":value,
                "estimator_decision":decision,"scene_status":ss,"radar_status":rs,"controller_input":observed,
                "fused_context":fused,"desired_mmps2":desired,"requested_mmps2":requested,
                "projected_margin_m":margin,"reason":reason,"advisory_present":bool(record["advisory"]),
                "advisory_action":override,"simulation_only":True}
    def command_packet(self,command,now,seq):
        return seal("actuator",{"schema":1,**self.context,"seq":seq,"issued_us":now,
                    "accel_mmps2":command,"duration_us":100000},self.keys["actuator"],now)
    def actuate(self,packet,now):
        if self.mode.startswith("proposed"):
            b,status=self.actuator_gate.verify(packet,now)
            return (b["accel_mmps2"] if b is not None else -4000),status
        body=self.raw_body(packet)
        value=body.get("accel_mmps2") if body else None
        return (value if type(value) is int and -4000<=value<=2000 else -4000),"unverified"
    def card(self,out,accepted,actuator_status):
        return {"time_us":out["now_us"],"reason":out["reason"],"risk_context":out["risk_context"],
                "scene_status":out["scene_status"],"radar_status":out["radar_status"],
                "actuator_status":actuator_status,"controller_input":out["controller_input"],
                "requested_command_mmps2":out["requested_mmps2"],"accepted_command_mmps2":accepted,
                "projected_margin_m":out["projected_margin_m"],
                "untrusted_advisory":"accepted by raw baseline" if out["advisory_action"] is not None else "ignored" if out["advisory_present"] else "absent",
                "assurance":"conditional simulation evidence",
                "assumptions":["bounded honest sensor observations","specified longitudinal dynamics","bounded actuator execution"],
                "human_evaluation":"not performed"}
