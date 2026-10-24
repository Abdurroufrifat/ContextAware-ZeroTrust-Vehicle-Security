"""Post-study limitation witnesses. Not pooled with the frozen 1056-trial grid."""
from common import *
install_guard()
from tube_core import admit
base=dict(candidate_mmps=13000,anchor_mmps=13000,uncertainty_mmps=250,
 measured_us=0,now_us=100000,clock_uncertainty_us=0,execution_error_mmps2=1000,
 commands=[(0,100000,0)],assumptions=True)
honest=admit(**base);forged=admit(**base)
assert honest==forged and honest["value_mmps"]==13000
worlds=[{"true_speed_mmps":10000,"reports_forged":True},{"true_speed_mmps":13000,"reports_forged":False}]
for world in worlds:world["selected_error_mmps"]=abs(honest["value_mmps"]-world["true_speed_mmps"])
assert [w["selected_error_mmps"] for w in worlds]==[3000,0]
assert worlds[0]["true_speed_mmps"]+500<worlds[1]["true_speed_mmps"]-500
# A valid anchor at 10000, but the supplied +2000 command differs from genuine -4000.
# The same observation is possible when the command is genuine +2000 and actual input is -4000.
extra={**base,"candidate_mmps":None,"anchor_mmps":10000,"commands":[(0,100000,2000)]}
decision=admit(**extra);true_speed=10000-4000/10
assert decision["selection"]=="propagated_anchor" and decision["value_mmps"]==10150
assert decision["value_mmps"]-true_speed==550
limitations=[]
for violated,genuine_command in [("command_history_integrity",-4000),("execution_error_bound",2000)]:
 limitations.append({"violated_premise":violated,"inputs":extra,"genuine_command_mmps2":genuine_command,
 "actual_acceleration_input_mmps2":-4000,"true_speed_mmps":9600,"decision":decision,
 "selected_error_mmps":550,"anchor_bound_holds":True,"age_bound_holds":True})
assert honest["physical_validation_established"] is False and honest["promotion_allowed"] is False
write("limitation_witness.json",{"passed":True,"post_study_not_pooled":True,"identical_observable_inputs":base,
 "identical_decision":honest,"worlds":worlds,"acceptable_intervals_mmps":[[9500,10500],[12500,13500]],
 "conclusion":"Identical compromised observations cannot certify both disjoint true-speed budgets.",
 "additional_broken_premise_witnesses":limitations,"novel_theorem_claim":False,"physical_validation_established":False})
print("PASS: source-forgery indistinguishability (3000 mm/s), forged-log and broken-execution witnesses (550 mm/s)")
