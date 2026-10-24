"""Independent integer endpoint oracle; does not import clock_core.select."""
from pathlib import Path
import csv,json,hashlib,itertools
R=Path(__file__).resolve().parent;P=R.parent.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
lock=json.loads((R/"input_lock.json").read_text())
assert all(sha(P/name)==digest for name,digest in lock.items())
summary=json.loads((R/"summary.json").read_text())
with (R/"grid.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
assert len(rows)==7200
policies=("static_native","nominal_age_control","clock_aware")
counts={p:{g:dict(cases=0,selected=0,candidate=0,anchor=0,abstain=0,possible_error_above500=0,
selected_outside_uncertain_time_window=0) for g in ("all","assumed","unassumed")} for p in policies}
grid=list(itertools.product((0,1,50000,95000,100000),(0,1,1000,5000),(0,100,250,500),(0,2000,8000),(-500,-250,0,250,500),(False,True)))
for i,(age,clock,u,accel,offset,assumed) in enumerate(grid):
    truth_radius=u*1000000+accel*(age+clock)
    low=10000*1000000-truth_radius;high=10000*1000000+truth_radius
    for j,policy in enumerate(policies):
        row=rows[i*3+j]
        assert int(row["case_id"])==i and row["policy"]==policy
        for key,value in (("age_us",age),("clock_us",clock),("uncertainty_mmps",u),("acceleration_mmps2",accel),("offset_mmps",offset)):
            assert int(row[key])==value
        assert row["assumptions"]==str(assumed)
        assert int(row["truth_low_num"])*1000000==low*int(row["truth_low_den"])
        assert int(row["truth_high_num"])*1000000==high*int(row["truth_high_den"])
        fresh=(0<=age<=100000) if policy!="clock_aware" else age-clock>=0 and age+clock<=100000
        radius=u*1000000+(0 if policy=="static_native" else accel*(age if policy=="nominal_age_control" else age+clock))
        if not assumed or not fresh or radius>500000000:value=None;selection="abstain"
        elif abs(offset)*1000000+radius<=500000000:value=10000+offset;selection="candidate"
        else:value=10000;selection="anchor"
        actual=None if row["value_mmps"]=="" else int(row["value_mmps"])
        assert actual==value and row["selection"]==selection,(i,policy)
        bad=value is not None and max(abs(value*1000000-low),abs(value*1000000-high))>500000000
        temporal=value is not None and not (age-clock>=0 and age+clock<=100000)
        assert row["possible_error_above500"]==str(bad)
        assert row["selected_outside_uncertain_time_window"]==str(temporal)
        for group in ("all","assumed" if assumed else "unassumed"):
            c=counts[policy][group];c["cases"]+=1;c["selected"]+=value is not None;c[selection]+=1
            c["possible_error_above500"]+=bad;c["selected_outside_uncertain_time_window"]+=temporal
assert counts==summary["policies"]
assert counts["clock_aware"]["assumed"]["selected"]>0
assert counts["clock_aware"]["assumed"]["abstain"]>0
assert counts["clock_aware"]["assumed"]["possible_error_above500"]==0
violation=json.loads((R/"assumption_violation_control.json").read_text())
assert violation["violates_stipulated_model"] and not violation["conditional_theorem_applies"]
assert violation["actual_truth_mmps"]-violation["selected"]["value_mmps"]==violation["actual_error_mmps"]==1000
appraisal=json.loads((R/"actual_data_appraisal.json").read_text())
assert not appraisal["physical_validation_established"] and not appraisal["real_data_policy_evaluation_run"]
assert appraisal["unknown_bound_diagnostic"]["selection"]=="abstain"
assert all(sha(P/name)==digest for name,digest in lock.items())
print(json.dumps({"passed":True,"model_cases":2400,"exact_rows_checked":7200,"unchanged_inputs":len(lock),
"physical_validation_established":False,"specific_novelty_established":False},indent=2))
