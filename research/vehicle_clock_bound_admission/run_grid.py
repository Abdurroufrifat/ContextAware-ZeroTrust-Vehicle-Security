from pathlib import Path
import sys,json,csv,hashlib,itertools
from fractions import Fraction
R=Path(__file__).resolve().parent;P=R.parent.parent
sys.path.insert(0,str(R))
from clock_core import select
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,obj):(R/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
assert not (R/"summary.json").exists(),"Results frozen; use verify_grid.py"
legacy=R.parent/"vehicle_conditional_source"
sys.path.insert(0,str(legacy))
from conditional_core import choose
source_names=["research/vehicle_conditional_source/conditional_core.py","research/vehicle_can_attack_audit/attack_core.py",
"src/coact_cps/vehicle/witness_certificate.py","research/vehicle_reference_qualification/reference_registry.json",
"research/vehicle_reference_qualification/REFERENCE_EVIDENCE_REQUIREMENTS.json",
"research/vehicle_evidence_freeze/FINAL_VALIDATION_REQUIREMENTS.json","FINAL_PROJECT_README.md","CURRENT_PROJECT_STATUS.md",
"research/vehicle_comparison_scope_audit/method_scope.json","research/vehicle_equal_input_ablation/file_manifest.json"]
source_names += [str((R/name).relative_to(P)) for name in ("PROTOCOL.md","clock_core.py","test_clock_core.py","run_grid.py")]
inputs={name:sha(P/name) for name in source_names};write("input_lock.json",inputs)
registry=json.loads((P/source_names[3]).read_text())
assert all(not ref["independent_absolute_truth_qualified"] for ref in registry["references"])
actual=select(candidate_mmps=10000,anchor_mmps=10000,uncertainty_mmps=None,measured_us=950000,now_us=1000000,
              clock_uncertainty_us=None,acceleration_mmps2=None,assumptions=False)
write("actual_data_appraisal.json",{"equipment":"PC and existing datasets only, confirmed by user",
"existing_reference_entries":len(registry["references"]),"independently_qualified_entries":0,
"current_dataset_physical_bounds_supplied":False,"unknown_bound_diagnostic":actual,
"real_data_policy_evaluation_run":False,"physical_validation_established":False,
"interpretation":"Missing prerequisite evidence; abstention is not demonstrated operational security or availability."})
policies=("static_native","nominal_age_control","clock_aware")
summary={p:{g:dict(cases=0,selected=0,candidate=0,anchor=0,abstain=0,possible_error_above500=0,
                        selected_outside_uncertain_time_window=0) for g in ("all","assumed","unassumed")} for p in policies}
rows=[]
grid=itertools.product((0,1,50000,95000,100000),(0,1,1000,5000),(0,100,250,500),(0,2000,8000),(-500,-250,0,250,500),(False,True))
for i,(age,clock,u,accel,offset,assumed) in enumerate(grid):
    anchor=10000;c=anchor+offset;now=1000000;stamp=now-age
    legacy_value,code=choose(c/1000,now,anchor/1000,u/1000,stamp,now,assumed)
    legacy_mmps=None if legacy_value is None else int(Fraction(legacy_value)*1000)
    assert legacy_value is None or Fraction(legacy_value)*1000==legacy_mmps
    results={"static_native":{"selection":{0:"abstain",1:"candidate",2:"anchor"}[code],"value_mmps":legacy_mmps,"reason":"frozen_static_native"}}
    for p,delta in (("nominal_age_control",0),("clock_aware",clock)):
        results[p]=select(candidate_mmps=c,anchor_mmps=anchor,uncertainty_mmps=u,measured_us=stamp,now_us=now,
                          clock_uncertainty_us=delta,acceleration_mmps2=accel,assumptions=assumed)
    # Independent endpoint construction from the actual stipulated timing interval.
    true_radius=Fraction(u)+Fraction(accel*(age+clock),1000000)
    low=Fraction(anchor)-true_radius;high=Fraction(anchor)+true_radius
    assert low>=0
    for p,result in results.items():
        chosen=result["value_mmps"];bad=chosen is not None and max(abs(chosen-low),abs(chosen-high))>500
        temporal=chosen is not None and not (age-clock>=0 and age+clock<=100000)
        row=dict(case_id=i,policy=p,age_us=age,clock_us=clock,uncertainty_mmps=u,acceleration_mmps2=accel,
                 offset_mmps=offset,assumptions=assumed,selection=result["selection"],value_mmps=chosen,
                 reason=result["reason"],truth_low_num=low.numerator,truth_low_den=low.denominator,
                 truth_high_num=high.numerator,truth_high_den=high.denominator,
                 possible_error_above500=bool(bad),selected_outside_uncertain_time_window=bool(temporal))
        rows.append(row)
        for group in ("all","assumed" if assumed else "unassumed"):
            cell=summary[p][group];cell["cases"]+=1;cell[result["selection"]]+=1;cell["selected"]+=chosen is not None
            cell["possible_error_above500"]+=bad;cell["selected_outside_uncertain_time_window"]+=temporal
assert len(rows)==7200
with (R/"grid.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
violation=select(candidate_mmps=10000,anchor_mmps=10000,uncertainty_mmps=0,measured_us=950000,now_us=1000000,
                 clock_uncertainty_us=0,acceleration_mmps2=0,assumptions=True)
write("assumption_violation_control.json",{"assumed_uncertainty_mmps":0,"assumed_acceleration_mmps2":0,
"selected":violation,"actual_truth_mmps":11000,"actual_error_mmps":1000,
"violates_stipulated_model":True,"conditional_theorem_applies":False,"physical_validation_established":False})
result={"cases":2400,"policy_rows":7200,"policies":summary,"scope":"Exact finite-grid stipulated-model endpoint experiment",
        "unknown_assumptions_are_not_evidence":True,"real_vehicle_experiment_run":False,"physical_validation_established":False,
        "formula_novelty_claimed":False,"specific_algorithmic_novelty_established":False,"promotion_allowed":False}
assert summary["clock_aware"]["assumed"]["possible_error_above500"]==0
assert summary["clock_aware"]["assumed"]["selected"]>0
assert summary["clock_aware"]["assumed"]["selected_outside_uncertain_time_window"]==0
assert all(sha(P/name)==digest for name,digest in inputs.items())
write("summary.json",result);print(json.dumps(result,indent=2))
