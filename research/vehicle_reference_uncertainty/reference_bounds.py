"""Conditional numerical error intervals. These functions do not appraise sensors."""
import math
from fractions import Fraction

def number(x):
    if type(x) not in (int,float,Fraction) or isinstance(x,float) and not math.isfinite(x):
        raise ValueError('Expected a finite real number, not a flag or string')
    return Fraction(x)

def error_interval(output,reference,bound):
    if bound is not None and number(bound)<0:raise ValueError('Negative reference uncertainty')
    if output is None or reference is None or bound is None:return None
    d=abs(number(output)-number(reference));u=number(bound)
    return max(Fraction(0),d-u),d+u

def position_difference_budget(error_at_start,error_at_end,interval_seconds):
    a,b,t=map(number,(error_at_start,error_at_end,interval_seconds))
    if min(a,b)<0 or t<=0:raise ValueError('Invalid positional bounds or time interval')
    # Bounds only the interval-mean velocity VECTOR derived from two positions.
    # It is not a bound on instantaneous speed, path-average speed, or clock error.
    return (a+b)/t

def percentile95(values):
    if not values:return None
    x=sorted(values);at=Fraction(95,100)*(len(x)-1);lo=at.numerator//at.denominator
    return x[lo]+(x[min(lo+1,len(x)-1)]-x[lo])*(at-lo)

def summarize(rows):
    """Rows are (accepted output or None, reported reference or None, supplied bound or None)."""
    requests=accepted=0;intervals=[]
    for output,reference,bound in rows:
        requests+=1
        if output is None:continue
        number(output);accepted+=1
        interval=error_interval(output,reference,bound)
        if interval is not None:intervals.append(interval)
    qualified=len(intervals);complete=qualified==accepted and accepted>0
    lo=percentile95([x[0] for x in intervals]) if complete else None
    hi=percentile95([x[1] for x in intervals]) if complete else None
    possible=sum(x[1]>2 for x in intervals);unavoidable=sum(x[0]>2 for x in intervals)
    coverage=100*accepted/requests if requests else None
    return dict(requests=requests,accepted=accepted,coverage_percent=coverage,
                accepted_with_qualified_numeric_bound=qualified,accepted_without_qualified_numeric_bound=accepted-qualified,
                p95_lower_absolute_error_mps=float(lo) if lo is not None else None,
                p95_upper_absolute_error_mps=float(hi) if hi is not None else None,
                p95_upper_absolute_error_exact=str(hi) if hi is not None else None,
                possible_error_gt2_count=possible if complete else None,
                unavoidable_error_gt2_count=unavoidable if complete else None,
                qualified_possible_error_gt2_count=possible,qualified_unavoidable_error_gt2_count=unavoidable,
                conditional_numeric_gates_pass=bool(complete and 100*accepted>=95*requests and hi<=Fraction(1,2) and possible==0),
                physical_reference_appraised=False,physical_validation_complete=False,promotion_allowed=False,
                interpretation='Conditional on externally justified per-query instantaneous-speed bounds. Metadata and arithmetic do not validate calibration, timing, independence or endpoint protection.')
