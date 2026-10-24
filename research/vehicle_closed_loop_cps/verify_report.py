from cps_common import R,dump
import json,csv,math,itertools,gzip
def main():
    trials=json.loads((R/"results.json").read_text())["trials"]
    cfg=json.loads((R/"PROTOCOL.json").read_text())
    expected=set(itertools.product(cfg["profiles"],cfg["cases"],cfg["seeds"],cfg["policies"]))
    actual={(t["profile"],t["case"],t["seed"],t["mode"]) for t in trials}
    assert len(trials)==len(actual)==540 and actual==expected
    with (R/"aggregate_results.csv").open(newline="",encoding="utf-8") as f:cells=list(csv.DictReader(f))
    assert len(cells)==65
    for cell in cells:
        values=[t for t in trials if t["mode"]==cell["mode"] and (cell["case"]=="all_cases" or t["case"]==cell["case"])]
        selected=sum(t["selected"] for t in values);q=sum(t["queries"] for t in values)
        ints={"trials":len(values),"queries":q,"selected":selected,"over_0_5":sum(t["over_0_5"] for t in values),
              "over_2":sum(t["over_2"] for t in values),"collision_trials":sum(bool(t["collision_trial"]) for t in values),
              "collision_events":sum(t["collision_events"] for t in values),"unrecoverable_queries":sum(t["unrecoverable_queries"] for t in values)}
        for k,v in ints.items():assert int(cell[k])==v
        floats={"availability":selected/q,"minimum_gap_m":min(t["minimum_gap_m"] for t in values),
                "mean_progress_m":math.fsum(t["progress_m"] for t in values)/len(values),
                "fallback_fraction":sum(t["fallback_queries"] for t in values)/q}
        for k,v in floats.items():assert abs(float(cell[k])-v)<1e-10
        if selected:assert abs(float(cell["mae_mps"])-math.fsum(t["mae_mps"]*t["selected"] for t in values if t["selected"])/selected)<1e-10
        else:assert cell["mae_mps"]==""
    pairs=json.loads((R/"paired_results.json").read_text())["pairs"];assert len(pairs)==108
    lookup={(t["profile"],t["case"],t["seed"],t["mode"]):t for t in trials}
    for row in pairs:
        key=(row["profile"],row["case"],row["seed"])
        a=lookup[*key,"proposed_learned"];b=lookup[*key,"sensor_auth"];m=lookup[*key,"proposed_midpoint"]
        assert row["proposed_collision"]==a["collision_trial"] and row["sensor_auth_collision"]==b["collision_trial"]
        assert abs(row["progress_difference_m"]-(a["progress_m"]-b["progress_m"]))<1e-10
        assert row["proposed_mae_mps"]==a["mae_mps"] and row["midpoint_mae_mps"]==m["mae_mps"]
    assert len({(r["profile"],r["case"],r["seed"]) for r in pairs})==108
    seed_rows=json.loads((R/"seed_sensitivity.json").read_text())
    assert {r["seed"] for r in seed_rows}==set(cfg["seeds"]) and len(seed_rows)==3
    for sr in seed_rows:
        assert len(sr["policies"])==5 and {x["mode"] for x in sr["policies"]}==set(cfg["policies"])
        for c in sr["policies"]:
            ts=[t for t in trials if t["seed"]==sr["seed"] and t["mode"]==c["mode"]]
            assert c["trials"]==len(ts) and c["selected"]==sum(t["selected"] for t in ts)
            assert c["collision_trials"]==sum(bool(t["collision_trial"]) for t in ts)
            expected=math.fsum(t["mae_mps"]*t["selected"] for t in ts if t["selected"])/c["selected"]
            assert abs(c["mae_mps"]-expected)<1e-10
    examples=json.loads((R/"operator_examples.json").read_text())
    assert set(examples)==set(cfg["cases"])
    for case,cards in examples.items():
        t=lookup["emergency_follow",case,601,"proposed_learned"]
        with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:original=[json.loads(x)["operator_card"] for x in f]
        assert cards==original and len(cards)==200
    page=(R/"operator_view.html").read_text(encoding="utf-8")
    embedded=page.split("const data=",1)[1].split(";const select=",1)[0]
    assert json.loads(embedded)==examples
    compare=json.loads((R/"comparison.json").read_text())
    rows={(c["mode"],c["case"]):c for c in cells}
    a=rows["proposed_learned","nominal"];b=rows["sensor_auth","nominal"];m=rows["proposed_midpoint","nominal"]
    assert abs(compare["nominal_progress_retention_vs_sensor_auth"]-float(a["mean_progress_m"])/float(b["mean_progress_m"]))<1e-10
    assert compare["nominal_learned_mae_lower_than_midpoint"]==(float(a["mae_mps"])<float(m["mae_mps"]))
    assert compare["collision_trial_reduction_vs_sensor_auth"]==int(rows["sensor_auth","all_cases"]["collision_trials"])-int(rows["proposed_learned","all_cases"]["collision_trials"])
    assert compare["all_case_learned_mae_lower_than_midpoint"]==(float(rows["proposed_learned","all_cases"]["mae_mps"])<float(rows["proposed_midpoint","all_cases"]["mae_mps"]))
    dump(R/"report_verification.json",{"passed":True,"confirmation_trials":540,"decisions":108000,"aggregate_cells":65,"paired_cells":108,"operator_examples":2400,"seed_clusters":3})
    print("INDEPENDENT_REPORT_PASS 540 trials, 65 cells, 108 pairs, 2400 operator examples")
if __name__=="__main__":main()
