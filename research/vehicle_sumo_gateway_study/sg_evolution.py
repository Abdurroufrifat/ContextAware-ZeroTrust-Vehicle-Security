"""Discrete SUMO speed-transition appraisal; FCD precision tolerance is numerical only."""
import math
def step_bound(before_mps,after_mps,command_mmps2,dt_us):
 if type(dt_us) is not int or dt_us<=0 or not all(type(x) in (int,float) and math.isfinite(x) for x in (before_mps,after_mps,command_mmps2)):raise ValueError("invalid transition")
 if before_mps<0 or after_mps<0 or not -4000<=command_mmps2<=2000:raise ValueError("invalid state/input")
 dt=dt_us/1000000
 lower=max(0.,before_mps+max(-4000,command_mmps2-1000)/1000*dt)
 upper=max(0.,before_mps+min(2000,command_mmps2+1000)/1000*dt)
 return lower-0.000002<=after_mps<=upper+0.000002
