"""Development-only point guard; base interval truth is an explicit premise."""
from fractions import Fraction as F
import math
def corroborate(selected,base):
    if selected is not None and (type(selected) is not int or selected<0):
        raise ValueError("selected speed must be a nonnegative integer mm/s or None")
    if selected is None:return None,"unavailable"
    fields=("low_n","low_d","high_n","high_d")
    if type(base) is not dict or base.get("conditional_budget_satisfied") is not True:
        return None,"base_unqualified"
    if any(type(base.get(k)) is not int for k in fields) or base["low_d"]<=0 or base["high_d"]<=0:
        return None,"base_unqualified"
    lo,hi=F(base["low_n"],base["low_d"]),F(base["high_n"],base["high_d"])
    if lo<0 or hi<lo:return None,"base_unqualified"
    lower,upper=max(0,math.ceil(hi-500)),math.floor(lo+500)
    if lower>upper:return None,"budget_exhausted"
    if lower<=selected<=upper:return selected,"corroborated"
    fallback=base.get("research_speed_mmps")
    if type(fallback) is int and lower<=fallback<=upper:return fallback,"base_fallback"
    return None,"inconsistent_base"
