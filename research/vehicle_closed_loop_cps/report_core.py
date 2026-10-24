"""Trial-level aggregation; selection and query denominators stay explicit."""
from collections import defaultdict
import math
def aggregate(trials):
    groups=defaultdict(list)
    for t in trials:groups[(t["mode"],t["case"])].append(t)
    out=[]
    for (mode,case),values in sorted(groups.items()):
        selected=sum(t["selected"] for t in values);queries=sum(t["queries"] for t in values)
        out.append({"mode":mode,"case":case,"trials":len(values),"queries":queries,"selected":selected,
            "availability":selected/queries,"mae_mps":math.fsum(t["mae_mps"]*t["selected"] for t in values if t["selected"])/selected if selected else None,
            "over_0_5":sum(t["over_0_5"] for t in values),"over_2":sum(t["over_2"] for t in values),
            "collision_trials":sum(bool(t["collision_trial"]) for t in values),"collision_events":sum(t["collision_events"] for t in values),
            "minimum_gap_m":min(t["minimum_gap_m"] for t in values),"mean_progress_m":math.fsum(t["progress_m"] for t in values)/len(values),
            "fallback_fraction":sum(t["fallback_queries"] for t in values)/queries,
            "unrecoverable_queries":sum(t["unrecoverable_queries"] for t in values)})
    return out
