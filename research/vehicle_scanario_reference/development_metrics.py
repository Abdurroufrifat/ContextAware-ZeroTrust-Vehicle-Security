"""Fixed descriptive metrics. Empirical discrepancy gates do not establish physical truth."""
import math, statistics
from fractions import Fraction
def summarize(estimates,references):
    if len(estimates)!=len(references) or not references:
        raise ValueError("Equal nonempty query arrays required")
    errors=[]
    for estimate,reference in zip(estimates,references):
        if not math.isfinite(reference) or (estimate is not None and not math.isfinite(estimate)):
            raise ValueError("Nonfinite measurement")
        if estimate is not None: errors.append(abs(estimate-reference))
    errors.sort()
    p95=None
    if errors:
        index=Fraction(19,20)*(len(errors)-1)
        lo=index.numerator//index.denominator
        weight=index-lo
        p95=Fraction(errors[lo])*(1-weight)+Fraction(errors[min(lo+1,len(errors)-1)])*weight
    large=sum(x>2 for x in errors)
    empirical=bool(p95 is not None and p95<=Fraction(1,2) and
        100*len(errors)>=95*len(references) and large==0)
    return dict(queries=len(references),accepted=len(errors),
        coverage_pct=100*len(errors)/len(references),p95_discrepancy_mps=float(p95) if p95 is not None else None,
        median_discrepancy_mps=statistics.median(errors) if errors else None,
        max_discrepancy_mps=max(errors) if errors else None,discrepancy_above_2_mps=large,
        empirical_discrepancy_gate=empirical,physical_gate_pass=None)
def rear_wheel_speed(message):
    wheels=[w for w in message["wheel_speed"] if w["id"] in (3,4)]
    ids=[w["id"] for w in wheels]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate rear-wheel ID")
    if set(ids)!={3,4} or any(w["speed"]["availability"]["data"] not in (1,2) for w in wheels): return None
    values=[w["speed"]["mean"] for w in wheels]
    if not all(math.isfinite(v) for v in values): return None
    return statistics.median(values)
def bin_name(value,edges):
    if not math.isfinite(value): return "unavailable"
    for low,high in zip(edges,edges[1:]):
        if low<=value<high: return f"[{low},{high})"
    return "outside"
