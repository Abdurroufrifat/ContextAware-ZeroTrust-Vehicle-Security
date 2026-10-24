"""Post-evaluation diagnosis only: no fitting, tuning or admission decisions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HD=Path(__file__).resolve().parent
BR=HD.parent/"vehicle_boreas_rt_reference_study"
sys.path.insert(0,str(BR))
from common import checked,sha,install_guard
from boreas_core import decode
install_guard()
pred=json.loads(checked(BR/"predictions.json").read_text())
result=json.loads(checked(BR/"results.json").read_text())
lock=json.loads(checked(BR/"data_lock.json").read_text())
objects={x["key"]:x for x in lock["objects"]}
lookup={(x["frame_id"],x["method"],x["age_us"]):x for x in result["rows"]}
fits=[x for x in pred["rows"] if x["method"]=="huber3"]
assert len(fits)==len({x["frame_id"] for x in fits})==96
rows=[]
for item in fits:
    entry=objects[item["key"]];path=checked(BR/"data"/item["key"])
    assert sha(path)==entry["sha256"]
    points,_=decode(path.read_bytes())
    ranges=np.linalg.norm(points[:,:3],axis=1)
    ids=np.flatnonzero(np.isfinite(points).all(axis=1)&(ranges>=3)&(ranges<=100))
    ids=ids[np.linspace(0,len(ids)-1,min(10000,len(ids)),dtype=np.int64)]
    sample=points[ids];x=sample[:,:3]/np.linalg.norm(sample[:,:3],axis=1)[:,None]
    beta=np.asarray(item["solution"]["beta"]);residual=x@beta+sample[:,3]
    source=lookup[(item["frame_id"],"huber3",0)]
    assert len(ids)==item["solution"]["n_used"] and source["scored"]
    rows.append({"frame_id":item["frame_id"],"key":item["key"],"sequence":source["sequence"],
        "used_points":len(ids),"sample_indices_sha256":hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "predicted_speed_mps":source["prediction_mps"],"reference_speed_mps":source["reference_mps"],
        "absolute_disagreement_mps":source["absolute_disagreement_mps"],
        "estimated_vertical_component_mps":float(beta[2]),
        "vertical_fraction_of_speed":float(abs(beta[2])/np.linalg.norm(beta)) if np.linalg.norm(beta)>0 else 0.,
        "condition_number":item["solution"]["condition_number"],
        "median_absolute_fit_residual_mps":float(np.median(abs(residual))),
        "fraction_fit_residual_within_fixed_huber_delta":float(np.mean(abs(residual)<=.3)),
        "diagnostic_only":True})
assert {x["frame_id"] for x in rows}==set(range(96))
summary={"frames":96,"above_05_mps":sum(x["absolute_disagreement_mps"]>.5 for x in rows),
    "above_2_mps":sum(x["absolute_disagreement_mps"]>2 for x in rows),
    "above_2_and_majority_within_huber_delta":sum(x["absolute_disagreement_mps"]>2 and x["fraction_fit_residual_within_fixed_huber_delta"]>.5 for x in rows),
    "refit_performed":False,"threshold_retuned":False,"causal_dynamic_object_labels_available":False,
    "root_cause_scope":"Independent replay reproduces fitted coefficients, residuals and recorded-reference disagreements. Large fitted vertical components and low-residual disagreements are observed; model, measurement, timing and reference causes are not isolated."}
out={"input_hashes":{n:sha(BR/n) for n in ("predictions.json","results.json","boreas_core.py","membership.json","data_lock.json")},"summary":summary,"rows":rows}
checked(HD/"diagnosis.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
