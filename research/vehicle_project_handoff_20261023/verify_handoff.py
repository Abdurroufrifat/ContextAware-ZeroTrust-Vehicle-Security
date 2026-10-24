"""Independent replay of all frozen-coefficient diagnostic rows."""
from pathlib import Path
import json,sys,struct,math,hashlib
import numpy as np
HD=Path(__file__).resolve().parent;BR=HD.parent/"vehicle_boreas_rt_reference_study"
sys.path.insert(0,str(BR))
from common import checked,sha,install_guard
install_guard()
d=json.loads(checked(HD/"diagnosis.json").read_text())
assert all(sha(BR/n)==h for n,h in d["input_hashes"].items())
p=json.loads(checked(BR/"predictions.json").read_text())
fits={x["frame_id"]:x for x in p["rows"] if x["method"]=="huber3"}
results=json.loads(checked(BR/"results.json").read_text())
source={(x["frame_id"],x["method"],x["age_us"]):x for x in results["rows"]}
rows=d["rows"];assert len(rows)==96 and {x["frame_id"] for x in rows}==set(fits)==set(range(96))
for row in rows:
    fit=fits[row["frame_id"]];assert row["key"]==fit["key"]
    expected=source[(row["frame_id"],"huber3",0)]
    assert row["sequence"]==expected["sequence"] and row["reference_speed_mps"]==expected["reference_mps"]
    assert row["absolute_disagreement_mps"]==expected["absolute_disagreement_mps"]
    assert row["condition_number"]==fit["solution"]["condition_number"] and row["diagnostic_only"] is True
    records=list(struct.iter_unpack("<8fQ",checked(BR/"data"/row["key"]).read_bytes()))
    eligible=[i for i,r in enumerate(records) if all(math.isfinite(v) for v in r[:8]) and 9<=sum(v*v for v in r[:3])<=10000]
    count=min(10000,len(eligible))
    # Positive positions are truncated by the production linspace. Build separately.
    step=(len(eligible)-1)/(count-1)
    positions=[int(k*step) for k in range(count)];positions[-1]=len(eligible)-1
    chosen=np.array([eligible[k] for k in positions],dtype="<i8")
    assert row["used_points"]==count and row["sample_indices_sha256"]==hashlib.sha256(chosen.tobytes()).hexdigest()
    b=fit["solution"]["beta"];residuals=[]
    speed=math.sqrt(math.fsum(v*v for v in b))
    assert row["estimated_vertical_component_mps"]==b[2]
    assert math.isclose(row["vertical_fraction_of_speed"],abs(b[2])/speed if speed>0 else 0.,abs_tol=1e-12)
    for i in chosen:
        r=records[i];distance=math.sqrt(sum(v*v for v in r[:3]))
        residuals.append(abs(math.fsum(r[j]*b[j] for j in range(3))/distance+r[3]))
    residuals.sort();median=(residuals[(count-1)//2]+residuals[count//2])/2
    assert math.isclose(median,row["median_absolute_fit_residual_mps"],abs_tol=1e-9)
    assert math.isclose(sum(x<=.3 for x in residuals)/count,row["fraction_fit_residual_within_fixed_huber_delta"],abs_tol=1e-12)
    assert math.isclose(math.sqrt(math.fsum(v*v for v in b)),row["predicted_speed_mps"],abs_tol=1e-9)
    assert math.isclose(abs(row["predicted_speed_mps"]-row["reference_speed_mps"]),row["absolute_disagreement_mps"],abs_tol=1e-9)
s=d["summary"]
assert s["frames"]==96 and s["above_05_mps"]==sum(r["absolute_disagreement_mps"]>.5 for r in rows)
assert s["above_2_mps"]==sum(r["absolute_disagreement_mps"]>2 for r in rows)
assert s["above_2_and_majority_within_huber_delta"]==sum(r["absolute_disagreement_mps"]>2 and r["fraction_fit_residual_within_fixed_huber_delta"]>.5 for r in rows)
assert s["refit_performed"] is s["threshold_retuned"] is False
print(json.dumps({"diagnostic_replay_passed":True,"verified_frames":96,"new_estimator_claim":False},indent=2))
