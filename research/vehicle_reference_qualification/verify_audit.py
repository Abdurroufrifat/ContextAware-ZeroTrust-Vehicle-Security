from shared import *
import numpy as np
verify_lock()
summary=read("summary.json");rows=read("sensitivity_rows.json")
for path,digest in read("input_lock.json").items():
    if Path(path)==P/"CURRENT_PROJECT_STATUS.md":
        assert sha(R/"PROJECT_STATUS_BEFORE_REFERENCE_AUDIT.md")==digest
        expected=read("completion.json")["current_status_sha256"] if (R/"completion.json").exists() else digest
        assert sha(path)==expected
    else:assert sha(path)==digest,path
arrays={};queries_verified=0;strata_verified=0
for seq in ["Library01","Library02"]:
    d=np.load(R/(seq+"_reference_audit.npz"),allow_pickle=False);arrays[seq]=d
    raw=[line.split(",") for line in (H/"data"/seq/"sensor_data/inspva.csv").read_text().splitlines() if line.strip()]
    times=np.array([int(line[0]) for line in raw],np.int64)
    codes=np.array([int(line[10].strip().split(":")[1]) for line in raw],np.int64)
    np.testing.assert_array_equal(times,d["ins_times"]);np.testing.assert_array_equal(codes,d["ins_status"])
    cursor=-1;expected=[];ages=[]
    for q in d["queries"]:
        while cursor+1<len(times) and times[cursor+1]<=q:cursor+=1
        age=int(q-times[cursor]) if cursor>=0 else -1
        expected.append(int(codes[cursor]) if cursor>=0 and 0<=age<=100_000_000 else -1);ages.append(age)
    np.testing.assert_array_equal(expected,d["query_status"]);np.testing.assert_array_equal(ages,d["status_age_ns"])
    h=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
    for name,oldname in [("queries","queries"),("original_prediction","request_prediction"),("original_admitted","ready"),("reference","request_reference")]:
        np.testing.assert_array_equal(d[name],h[oldname])
    for group,entry in summary["sequences"][seq]["groups"].items():
        mask=np.ones(len(d["queries"]),bool) if group=="all" else np.zeros(len(d["queries"]),bool)
        if group!="all":start=int(group.split("_")[1]);mask[start:start+1200]=True
        assert sum(mask)==entry["queries"]
        for scope,values in entry["scopes"].items():
            p=d["original_prediction"] if scope=="original_radar" else d["current_prediction"]
            a=d["original_admitted"] if scope=="original_radar" else d["current_admitted"]
            records=[(mask,values["unchanged_all_status_metrics"])]
            records += [(mask&(d["query_status"]==int(code)),record) for code,record in values["post_hoc_by_status"].items()]
            for selected,record in records:
                admitted=selected&a&np.isfinite(p);scored=admitted&np.isfinite(d["reference"])
                e=abs(p[scored]-d["reference"][scored])
                assert record["attempts"]==sum(selected) and record["admitted"]==sum(admitted) and record["scored"]==sum(scored)
                assert record["unknown_reference"]==sum(admitted&~np.isfinite(d["reference"])) and record["above2"]==sum(e>2)
                if len(e):
                    assert abs(record["p95_mps"]-float(np.quantile(e,.95)))<1e-12
                    assert abs(record["mae_mps"]-float(e.mean()))<1e-12
                else:assert record["p95_mps"] is None
                strata_verified+=1
    queries_verified+=len(d["queries"])
for row in rows:
    d=arrays[row["sequence"]];group=row["group"]
    mask=np.ones(len(d["queries"]),bool) if group=="all" else np.zeros(len(d["queries"]),bool)
    if group!="all":start=int(group.split("_")[1]);mask[start:start+1200]=True
    p=d["original_prediction"] if row["scope"]=="original_radar" else d["current_prediction"]
    a=d["original_admitted"] if row["scope"]=="original_radar" else d["current_admitted"]
    admitted=mask&a&np.isfinite(p);scored=admitted&np.isfinite(d["reference"]);error=abs(p[scored]-d["reference"][scored])
    eps=row["epsilon_assumption_mps"];unknown=int(sum(admitted&~np.isfinite(d["reference"])))
    assert row["unknown_reference"]==unknown
    observed=float(np.quantile(error,.95)) if len(error) else None
    budget=.5-observed if observed is not None and observed<=.5 else None
    assert row["max_uniform_epsilon_for_scored_p95"]==budget
    if eps is None:
        assert not row["reference_bound_known"] and row["true_error_p95_upper"] is None
        assert not row["conditional_full_admitted_set_upper_within_0p5"]
    else:
        lo=np.maximum(0,error-eps);hi=error+eps
        assert abs(row["true_error_p95_lower"]-float(np.percentile(lo,95)))<1e-12
        assert abs(row["true_error_p95_upper"]-float(np.percentile(hi,95)))<1e-12
        assert row["definitely_above2"]==sum(lo>2) and row["possibly_above2"]==sum(hi>2)
        within=bool(np.quantile(hi,.95)<=.5)
        assert row["scored_subset_upper_within_0p5"]==within
        assert row["conditional_full_admitted_set_upper_within_0p5"]==bool(within and unknown==0)
registry=read("reference_registry.json");assert all(not ref["independent_absolute_truth_qualified"] for ref in registry["references"])
assert not summary["actual_independent_absolute_reference_found"] and summary["verified_reference_error_bound_mps"] is None
tail=summary["sequences"]["Library01"]["groups"]["segment_6000"]["scopes"]["original_radar"]["unchanged_all_status_metrics"]
assert tail["attempts"]==302 and tail["scored"]==301 and abs(tail["p95_mps"]-1.102297507859495)<1e-12
out={"passed":True,"queries_verified":queries_verified,"strata_metrics_verified":strata_verified,
 "hypothetical_sensitivity_rows_verified":len(rows),"causal_status_scan_verified":True,"original_failed_window_unchanged":True,
 "actual_independent_reference_found":False,"verified_reference_error_bound_mps":None,"physical_witness_admitted":False}
write("independent_verification.json",out);print(json.dumps(out,indent=2))
