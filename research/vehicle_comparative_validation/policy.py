"""Common packet-only experiment interface; no simulator/evaluator reference."""
from cv_common import P,KEYS
import json,math
from cps_session import CPSPolicy
from control_core import idm,safe_command,projected_margin
from baseline_core import s3e_update,agreement,closest_point,zocbf_command
METHODS=("full","no_learning","no_refinement","no_fusion","no_brake","no_actuator_auth","s3e_brake","ssf_zocbf")
def control_choice(method,desired,bounds):
    args=(bounds["ego_high"],bounds["lead_low"],bounds["gap_low"])
    if method=="no_brake":return desired,projected_margin(desired,*args)
    if method=="ssf_zocbf":return zocbf_command(desired,*args)
    return safe_command(desired,*args)
def context_bounds(ego_low,ego_high,scene,radar=None):
    if scene is None:return None
    glo=scene["gap_um"]/1e6-.0015;ghi=scene["gap_um"]/1e6+.0015
    llo=max(0,scene["lead_mmps"]/1000-.075);lhi=scene["lead_mmps"]/1000+.075
    if radar is not None:
        glo=max(glo,radar["gap_um"]/1e6-.05);ghi=min(ghi,radar["gap_um"]/1e6+.05)
        llo=max(llo,radar["relative_mmps"]/1000-.1+ego_low)
        lhi=min(lhi,radar["relative_mmps"]/1000+.1+ego_high)
    if glo>ghi or llo>lhi:return None
    return {"ego_low":ego_low,"ego_high":ego_high,"gap_low":glo,"gap_high":ghi,
            "lead_low":llo,"lead_high":lhi,"gap_m":(glo+ghi)/2,"lead_mps":(llo+lhi)/2}
def observation_interval(role,speed,age):
    radius=.27 if role=="anchor" else .101
    return max(0,speed-radius-4*age),speed+radius+2*age
class ComparisonPolicy:
    def __init__(self,state_directory,context,keys,method):
        if method not in METHODS:raise ValueError("method")
        mode="sensor_auth" if method in ("s3e_brake","ssf_zocbf") else "proposed_midpoint" if method in ("no_learning","no_refinement") else "proposed_learned"
        self.inner=CPSPolicy(state_directory,context,keys,mode)
        self.method=method;self.sets=[(0.,30.)];self.previous_now=1000000;self.previous_accepted=0
    def process(self,record):
        out=self.inner.process(record);out["mode"]=self.method
        now=record["now_us"]
        scene=self.inner.raw_body(record["scene_packet"]) if out["scene_status"]=="authenticated" else None
        radar=self.inner.raw_body(record["radar_packet"]) if out["radar_status"]=="authenticated" else None
        if self.method in ("s3e_brake","ssf_zocbf"):
            d=out["estimator_decision"];observations=[];domains=[]
            for role,status in (("candidate",d["candidate_status"]),("anchor",d["anchor_status"])):
                packet=record["packets"][role]
                if status!="authenticated" or packet is None:continue
                body=self.inner.raw_body(packet)
                age=(now-body["measured_us"])/1e6
                if age<0 or (role=="candidate" and age!=0):continue
                speed=body["speed_mmps"]/1000
                lo,hi=observation_interval(role,speed,age)
                observations.append((lo,hi));domains.append(role)
            if d.get("witness_low_n") is not None:
                observations.append((d["witness_low_n"]/d["witness_low_d"]/1000,d["witness_high_n"]/d["witness_high_d"]/1000))
                domains.append("scene_pair")
            sets=s3e_update(self.sets,observations,self.previous_accepted/1000,(now-self.previous_now)/1e6) if self.method=="s3e_brake" else agreement(observations,1)
            self.sets=sets;self.previous_now=now
            point=closest_point(sets)
            estimate=None if point is None else round(point*1000)
            out["adapter_decision"]=d
            out["estimator_decision"]={"plausible_sets_mps":sets,"observation_domains":domains,"observations_mps":observations,
                "point_budget_satisfied":False if estimate is None else max(abs(estimate/1000-sets[0][0]),abs(estimate/1000-sets[-1][1]))<=.5,
                "q":1,"adaptation":"scalar interval-union with clipped longitudinal dynamics" if self.method=="s3e_brake" else "current interval-valued reconstruction plus finite-grid headway ZO-CBF"}
            out.update(estimate_mmps=estimate,controller_input=None,fused_context=None,desired_mmps2=-4000,requested_mmps2=-4000,projected_margin_m=None,reason="plausible_state_unavailable")
            if estimate is not None and scene is not None and radar is not None:
                bounds=context_bounds(sets[0][0],sets[-1][1],scene,radar)
                out["reason"]="context_disagreement" if bounds is None else "observed_context_accepted"
                if bounds is not None:self._control(out,estimate/1000,bounds)
        elif self.method=="no_refinement":
            d=out["estimator_decision"]
            out["ablation_refinement_available"]=d.get("refinement_status")=="available"
            if out["ablation_refinement_available"]:
                from fractions import Fraction as F
                original=d["scene_decision"];lo=F(original["witness_low_n"],original["witness_low_d"]);hi=F(original["witness_high_n"],original["witness_high_d"])
                lower=max(0,math.ceil(hi-500));upper=math.floor(lo+500)
                value=min(upper,max(lower,round((lo+hi)/2))) if lower<=upper and original["research_speed_mmps"] is not None else None
                out["unrefined_original_decision"]=original
                out["refinement_ablation_source_decision"]=d
                out["estimator_decision"]={**original,"selection":"scene_midpoint_ablation","reason":"successful_refinement_removed","research_speed_mmps":value,"research_speed_mps":None if value is None else value/1000,"refined_budget_satisfied":False}
                out.update(estimate_mmps=value,controller_input=None,fused_context=None,desired_mmps2=-4000,requested_mmps2=-4000,projected_margin_m=None,reason="observation_unavailable")
                if value is not None and scene is not None and radar is not None:
                    bounds=context_bounds(max(0,value/1000-.5),value/1000+.5,scene,radar)
                    if bounds is None:out["reason"]="context_disagreement"
                    else:self._control(out,value/1000,bounds)
        elif self.method=="no_fusion":
            out.update(controller_input=None,fused_context=None,desired_mmps2=-4000,requested_mmps2=-4000,projected_margin_m=None,reason="observation_unavailable")
            if out["estimate_mmps"] is not None and scene is not None:
                speed=out["estimate_mmps"]/1000
                self._control(out,speed,context_bounds(max(0,speed-.5),speed+.5,scene))
        elif self.method=="no_brake" and out["fused_context"] is not None:
            out["requested_mmps2"],out["projected_margin_m"]=control_choice("no_brake",out["desired_mmps2"],out["fused_context"])
            out["reason"]="control_filter_removed"
        out["method"]=self.method
        return out
    def _control(self,out,speed,bounds):
        out["fused_context"]=bounds;out["controller_input"]={"ego_mps":speed,"lead_mps":bounds["lead_mps"],"gap_m":bounds["gap_m"]}
        desired=idm(speed,bounds["lead_mps"],bounds["gap_m"])
        command,margin=control_choice(self.method,desired,bounds)
        out.update(desired_mmps2=desired,requested_mmps2=command,projected_margin_m=margin,
            reason="reserve_unrecoverable" if margin<0 else "filter_limited" if command!=desired else "observed_context_accepted")
    def command_packet(self,command,now,seq):return self.inner.command_packet(command,now,seq)
    def actuate(self,packet,now):
        if self.method=="no_actuator_auth":
            body=self.inner.raw_body(packet);value=body.get("accel_mmps2") if body else None
            result=(value if type(value) is int and -4000<=value<=2000 else -4000),"unverified_ablation"
        else:
            # Baseline estimator mode is sensor_auth; all non-ablated methods share this actuator verifier.
            body,status=self.inner.actuator_gate.verify(packet,now)
            result=(body["accel_mmps2"] if body is not None else -4000),status
        self.previous_accepted=result[0];return result
    def card(self,out,accepted,status):return self.inner.card(out,accepted,status)
