"""Exact conditional current-query speed budget; no physical evidence appraisal."""
from fractions import Fraction

def uint(value):
    return type(value) is int and 0<=value<=2**63-1

def select(*,candidate_mmps,anchor_mmps,uncertainty_mmps,measured_us,now_us,
           clock_uncertainty_us,acceleration_mmps2,assumptions):
    result={"selection":"abstain","value_mmps":None,"reason":"assumptions_unknown",
            "radius_num":None,"radius_den":None,"conditional_budget_satisfied":False,
            "physical_validation_established":False,"production_policy_changed":False,
            "promotion_allowed":False}
    if assumptions is not True:return result
    if not all(uint(v) for v in (anchor_mmps,uncertainty_mmps,measured_us,now_us,
                                 clock_uncertainty_us,acceleration_mmps2)):
        return {**result,"reason":"invalid_or_missing_bound"}
    age=now_us-measured_us
    minimum_age=age-clock_uncertainty_us
    maximum_age=age+clock_uncertainty_us
    if minimum_age<0 or maximum_age>100000:
        return {**result,"reason":"uncertain_age_outside_window"}
    radius=Fraction(uncertainty_mmps)+Fraction(acceleration_mmps2*maximum_age,1000000)
    result.update(radius_num=radius.numerator,radius_den=radius.denominator,
                  minimum_age_us=minimum_age,maximum_age_us=maximum_age)
    if radius>500:return {**result,"reason":"budget_exhausted"}
    if uint(candidate_mmps) and abs(candidate_mmps-anchor_mmps)+radius<=500:
        return {**result,"selection":"candidate","value_mmps":candidate_mmps,
                "reason":"candidate_within_conditional_budget","conditional_budget_satisfied":True}
    return {**result,"selection":"anchor","value_mmps":anchor_mmps,
            "reason":"anchor_within_conditional_budget","conditional_budget_satisfied":True}
