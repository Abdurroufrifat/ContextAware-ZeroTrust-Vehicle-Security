"""A constructed bounded-error example, not a dataset attack result."""
import math
from cadence_core import CadencePolicy
def construct(reference_mps=10.,epsilon_mps=.5):
 if not all(type(x) in (int,float) and math.isfinite(x) for x in (reference_mps,epsilon_mps)) or epsilon_mps<=0 or reference_mps<epsilon_mps:raise ValueError('invalid bounded-error construction')
 return {'observation':{'gnss_mps':reference_mps+epsilon_mps,'wheel_mps':reference_mps+epsilon_mps,'radar_prediction_mps':reference_mps},'benign_true_mps':reference_mps+epsilon_mps,'attacked_true_mps':reference_mps-epsilon_mps,'epsilon_mps':epsilon_mps}
def check(params):
 world=construct();o=world['observation'];policies=[CadencePolicy(params),CadencePolicy(params)];decisions=[[],[]]
 for i in range(241):
  t=i*50000000
  for policy,trace in zip(policies,decisions):trace.append(policy.step(t,o['gnss_mps'],o['wheel_mps'],o['radar_prediction_mps'],True))
 return {'scope':'Constructed mathematical counterexample for this speed-only observation interface, not measured true speed or a new held-out experiment. Epsilon is hypothetical, not estimated from NavINST.','worlds':world,'injected_shift_mps':o['gnss_mps']-world['attacked_true_mps'],'identical_permission_sequences':decisions[0]==decisions[1],'granted_in_both_worlds':sum(a and b for a,b in zip(*decisions)),'endpoints':len(decisions[0]),'necessary_condition_for_zero_grants_at_shift_ge_1_and_no_benign_denials':'A worst-case witness error bound epsilon < 0.5 m/s is necessary for disjoint speed hypotheses at a 1 m/s attack threshold under this interface. It is not sufficient for a complete secure, available system. Additional trusted observations can change the interface and the conclusion.'}
if __name__=='__main__':
 import json
 from pathlib import Path
 root=Path(__file__).resolve().parent;params=json.loads((root/'policy_lock.json').read_text())['parameters'];result=check(params);(root/'observability_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
