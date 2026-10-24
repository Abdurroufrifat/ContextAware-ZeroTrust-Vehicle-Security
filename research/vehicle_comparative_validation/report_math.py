"""Predeclared aggregation: scenario pairs and six-seed descriptive cluster effects."""
import math,random
METRICS=("mae_mps","collision_trial","progress_m","seconds_gap_below_2m","fallback_queries")
def summarize(trials):
    n=len(trials);assert n
    selected=sum(x["selected"] for x in trials);queries=sum(x["queries"] for x in trials)
    return {"trials":n,"queries":queries,"selected":selected,"availability":selected/queries,
        "selected_mae_mps":math.fsum(x["mae_mps"]*x["selected"] for x in trials if x["selected"])/selected if selected else None,
        "over_0_5":sum(x["over_0_5"] for x in trials),"collision_trials":sum(x["collision_trial"] for x in trials),
        "collision_episodes":sum(x["collision_episodes"] for x in trials),
        "mean_progress_m":math.fsum(x["progress_m"] for x in trials)/n,
        "minimum_gap_m":min(x["minimum_gap_m"] for x in trials),
        "mean_seconds_gap_below_2m":math.fsum(x["seconds_gap_below_2m"] for x in trials)/n,
        "fallback_queries":sum(x["fallback_queries"] for x in trials),
        "unrecoverable_queries":sum(x["unrecoverable_queries"] for x in trials),
        "changed_actuator_frames":sum(x["changed_actuator_frames"] for x in trials),
        "changed_actuator_frames_accepted":sum(x["changed_actuator_frames_accepted"] for x in trials),
        "maximum_decision_seconds":max(x["maximum_decision_seconds"] for x in trials)}
def paired(trials,left,right,group,case=None):
    def index(method):
        rows=[t for t in trials if t["method"]==method and t["group"]==group and (case is None or t["case"]==case)]
        result={(t["profile"],t["case"],t["seed"]):t for t in rows}
        assert len(result)==len(rows),"duplicate matched scenario"
        return result
    a,b=index(left),index(right);assert a and set(a)==set(b),"unmatched comparison"
    rows=[]
    for key in sorted(a):
        d={"profile":key[0],"case":key[1],"seed":key[2]}
        for metric in METRICS:
            x,y=a[key][metric],b[key][metric]
            d[metric]=None if x is None or y is None else x-y
        d["availability"]=a[key]["selected"]/a[key]["queries"]-b[key]["selected"]/b[key]["queries"]
        rows.append(d)
    return rows
def cluster_effect(rows,metric,resamples=2000):
    seeds=sorted({r["seed"] for r in rows})
    means=[math.fsum(r[metric] for r in rows if r["seed"]==s and r[metric] is not None)/sum(r["seed"]==s and r[metric] is not None for r in rows) for s in seeds]
    rng=random.Random(2911)
    samples=sorted(math.fsum(rng.choice(means) for _ in means)/len(means) for _ in range(resamples))
    return {"mean_left_minus_right":math.fsum(means)/len(means),"seed_clusters":len(seeds),"seed_effects":dict(zip(seeds,means)),
        "descriptive_cluster_bootstrap_95":[samples[math.floor(.025*(resamples-1))],samples[math.ceil(.975*(resamples-1))]],
        "resamples":resamples,"interpretation":"Exploratory six-seed cluster interval; not query-independent inference or proof of general superiority."}
