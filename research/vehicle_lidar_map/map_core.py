"""Causal research interval motion helpers; no physical speed certificate."""
import numpy as np
def firing_mask(offsets,header_ns,receipt_ns):
 offsets=np.asarray(offsets,dtype=float)
 good=np.isfinite(offsets)&(offsets>=-.001)&(offsets<=.2)
 return good & (offsets <= (receipt_ns-header_ns)/1e9)
def interval_motion(previous,current,previous_end_ns,current_end_ns,mount):
 dt=(current_end_ns-previous_end_ns)/1e9
 if not .05<=dt<=.2: raise ValueError("invalid interval")
 delta=mount@np.linalg.inv(previous)@current@np.linalg.inv(mount)
 if not np.isfinite(delta).all(): raise ValueError("nonfinite pose")
 speed=float(np.linalg.norm(delta[:3,3])/dt)
 if speed>60: raise ValueError("outside numeric speed bound")
 return dict(speed_mps=speed,dt_seconds=dt,midpoint_ns=(previous_end_ns+current_end_ns)//2,relative_rear_pose=delta.tolist())
def eligible(midpoint_ns,completed_ns,query_ns):
 return completed_ns<=query_ns and 0<=query_ns-midpoint_ns<=100000000
def select_available(rows,query_ns):
 return next((r for r in reversed(rows) if r["completed_ns"]<=query_ns and r["receipt_ns"]<=query_ns),None)
