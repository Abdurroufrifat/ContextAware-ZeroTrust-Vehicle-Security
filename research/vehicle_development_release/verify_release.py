"""Read-only verification of source identities and exact numerical sidecar."""
from shared import *
from fractions import Fraction
from release_core import validate_scenario_identity
import numpy as np,math
for name,digest in read("experiment_lock.json").items():assert sha(R/name)==digest
for relative,digest in read("input_lock.json").items():
    if relative in {"CURRENT_PROJECT_STATUS.md","FINAL_PROJECT_README.md"}:
        backup="PROJECT_STATUS_BEFORE_RELEASE.md" if relative.startswith("CURRENT") else "FINAL_README_BEFORE_RELEASE.md"
        assert sha(R/backup)==digest
        if (R/"completion.json").exists():
            field="current_status_sha256" if relative.startswith("CURRENT") else "final_readme_sha256"
            assert sha(P/relative)==read("completion.json")[field]
        else:assert sha(P/relative)==digest
    else:
        path=(P/relative).resolve();assert path.is_relative_to(P.resolve())
        low=str(path).lower();assert "library03" not in low and "rav4" not in low and "vehicle_final_evaluation" not in low and not ("urban02" in low and "night" in low)
        assert sha(path)==digest,relative
Q=R.parent/"vehicle_reference_qualification";arrays={}
for seq in ["Library01","Library02"]:arrays[seq]=np.load(Q/(seq+"_reference_audit.npz"),allow_pickle=False)
old=read("NUMERICAL_CORRECTION.json");rows=read("exact_sensitivity_rows.json")
def percentile(values):
    if not values:return None
    values=sorted(values);n=19*(len(values)-1);i,r=divmod(n,20)
    return (values[i]*(20-r)+values[min(i+1,len(values)-1)]*r)/20
changed=0
historical=json.loads((Q/"sensitivity_rows.json").read_text())
assert len(rows)==len(historical)==182
for index,(row,prior) in enumerate(zip(rows,historical)):
    validate_scenario_identity(row,prior,index)
    d=arrays[row["sequence"]];mask=np.ones(len(d["queries"]),bool)
    if row["group"]!="all":
        start=int(row["group"].split("_")[1]);mask[:]=False;mask[start:start+1200]=True
    p=d["original_prediction"] if row["scope"]=="original_radar" else d["current_prediction"]
    accepted=d["original_admitted"] if row["scope"]=="original_radar" else d["current_admitted"]
    present=mask&accepted&np.isfinite(p);scored=present&np.isfinite(d["reference"]);unknown=int(sum(present&~np.isfinite(d["reference"])))
    assert row["admitted"]==sum(present) and row["scored"]==sum(scored) and row["unknown_reference"]==unknown
    delta=[abs(Fraction.from_float(float(a))-Fraction.from_float(float(b))) for a,b in zip(p[scored],d["reference"][scored])]
    observed=percentile(delta);assert row["observed_proxy_p95_exact"]==str(observed)
    epsilon=row["epsilon_assumption_mps"]
    if epsilon is None:
        assert row["true_error_p95_upper_exact"] is None and not row["scored_subset_upper_within_0p5"]
    else:
        epsilon=Fraction.from_float(float(epsilon))
        lower=[max(Fraction(0),value-epsilon) for value in delta];upper=[value+epsilon for value in delta]
        high=percentile(upper);lo=percentile(lower)
        assert row["true_error_p95_upper_exact"]==str(high) and row["true_error_p95_lower_exact"]==str(lo)
        assert row["scored_subset_upper_within_0p5"]==bool(high<=Fraction(1,2))
        assert row["conditional_full_admitted_set_upper_within_0p5"]==bool(high<=Fraction(1,2) and unknown==0)
        assert row["possibly_above2"]==sum(value>2 for value in upper) and row["definitely_above2"]==sum(value>2 for value in lower)
    keys=["scored_subset_upper_within_0p5","conditional_full_admitted_set_upper_within_0p5","definitely_above2","possibly_above2"]
    changed+=any(row[k]!=prior[k] for k in keys)
assert changed==old["actual_rows_with_changed_flags_or_counts"]
boundary=old["exact_boundary_result"];exact=Fraction(1,4)+Fraction.from_float(math.nextafter(.25,math.inf))
assert exact>Fraction(1,2) and boundary["true_error_p95_upper_exact"]==str(exact)
assert boundary["true_error_p95_upper_display"]==.5 and not boundary["scored_subset_upper_within_0p5"]
index=read("evidence_index.json");assert not index["pooled_cross_cohort_metric_produced"] and not index["promotion_allowed"]
assert len({c["id"] for c in index["cohorts"]})==len(index["cohorts"])==5
out={"passed":True,"input_files_reverified":len(read("input_lock.json")),"exact_sensitivity_rows_independently_checked":len(rows),
 "actual_changed_flag_rows":changed,"rounding_boundary_correction_verified":True,
 "historical_reports_unchanged":True,"cohort_separation_preserved":True,
 "physical_reference_appraised":False,"promotion_allowed":False}
if __name__=="__main__":print(json.dumps(out,indent=2))
