"""Read-only explanation and secondary-analysis definitions."""
import math
NEGATIVE={"initial_false_context","authority_exceed","public_motion_innovation"}
def quantile(values,p):
    if not values:return None
    a=sorted(values);return a[max(0,math.ceil(p*len(a))-1)]
def facts(card):
    c=card["certificate"]
    point=card["point_budget_satisfied"] is True
    accepted=card["actuator_status"]=="authenticated"
    return {"speed_budget_certified":point,
      "speed_text":"Conditional +/-0.5 m/s point budget satisfied" if point else "Speed estimate has no certified +/-0.5 m/s point budget",
      "command_gate_accepted":accepted,
      "command_text":"Conditional command check passed" if accepted else "Command not authorized; backup requested",
      "assumptions_text":"Requires honest initial context, the stated honest-source subset, bounded actuator error and modeled timing.",
      "measurement_truth_text":"Authentication does not establish measurement truth.",
      "hold_ms":c["issued_hold_steps"]*10,"maximum_hold_ms":c["max_hold_steps"]*10,
      "execution_error_mps2":c["execution_error"],"refresh_status":c["refresh_status"],
      "requested_mps2":card["requested_command_mmps2"]/1000,"accepted_mps2":card["accepted_command_mmps2"]/1000,
      "human_effectiveness_evaluated":False}
def dominates(a,b):
    return a["contact_fraction"]<=b["contact_fraction"] and a["mean_progress_m"]>=b["mean_progress_m"] and (a["contact_fraction"]<b["contact_fraction"] or a["mean_progress_m"]>b["mean_progress_m"])
def summary(rows):
    n=len(rows);q=sum(t["queries"] for t in rows)
    return {"trials":n,"contacts":sum(t["collision_trial"] for t in rows),"contact_fraction":sum(t["collision_trial"] for t in rows)/n,
      "mean_progress_m":math.fsum(t["progress_m"] for t in rows)/n,
      "mean_speed_error_mps":math.fsum(t["mae_mps"]*t["selected"] for t in rows)/sum(t["selected"] for t in rows),
      "queries":q,"point_budget_certified_queries":sum(t["point_certified"] for t in rows),
      "point_budget_coverage":sum(t["point_certified"] for t in rows)/q,
      "command_accepted_queries":sum(t["command_accepted"] for t in rows),
      "accepted_with_uncertified_point":sum(t["accepted_with_uncertified_point"] for t in rows),
      "over_100ms":sum(t["over_100ms"] for t in rows),"over_100ms_fraction":sum(t["over_100ms"] for t in rows)/q}
