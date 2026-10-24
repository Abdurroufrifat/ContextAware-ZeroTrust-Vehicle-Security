"""Isolated development adapter; no changes to the frozen comparison policy."""
from dev_common import *
from policy import ComparisonPolicy
from ambiguity_guard import corroborate
import copy
class CorroboratedPolicy:
    def __init__(self,state_directory,context,keys):
        self.original=ComparisonPolicy(state_directory,context,keys,"full")
    def process(self,record):
        source=self.original.process(record)
        value,status=corroborate(source["estimate_mmps"],source["estimator_decision"].get("base_decision",{}))
        out=copy.deepcopy(source)
        out["mode"]=out["method"]="corroborated"
        out["source_output"]=source
        out["guard"]={"status":status,"conditional_budget_satisfied":status=="corroborated",
          "source_truth_established":False,"premises":"honest bounded anchor, authenticated accepted-command history, declared clock and execution bounds",
          "policy":"retain compatible learned estimate; abstain on unresolved source conflict"}
        if value is None:
            decision=out["estimator_decision"]
            decision.update(selection="guard_abstain",reason="guard_"+status,research_speed_mmps=None,research_speed_mps=None,
                conditional_budget_satisfied=False,scene_budget_satisfied=False,refined_budget_satisfied=False)
            out.update(estimate_mmps=None,controller_input=None,fused_context=None,desired_mmps2=-4000,
                requested_mmps2=-4000,projected_margin_m=None,reason="guard_"+status)
        else:
            assert value==source["estimate_mmps"]
        return out
    def command_packet(self,command,now,seq):return self.original.command_packet(command,now,seq)
    def actuate(self,packet,now):return self.original.actuate(packet,now)
    def card(self,out,accepted,status):
        card=self.original.card(out,accepted,status)
        card["selection_guard"]=copy.deepcopy(out["guard"])
        return card
