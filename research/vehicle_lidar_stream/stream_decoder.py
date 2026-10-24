"""Research VLS128 single-return decoder, based on retained BSD upstream reference."""
import math,struct
import numpy as np
BANKS=np.array([0xeeff,0xddff,0xccff,0xbbff]*3)
LASER=np.tile(np.arange(128),3)
GROUP=LASER//8
OFFSET_S=(2.665*GROUP-7.+5.333*(GROUP>7))*1e-6
def prepare(cal):
    if isinstance(cal,tuple): return cal
    ordered=sorted(cal["lasers"],key=lambda x:x["laser_id"])
    if [x["laser_id"] for x in ordered]!=list(range(128)): raise ValueError("Incomplete laser calibration")
    vert=np.array([x["vert_correction"] for x in ordered])
    rot=np.array([x["rot_correction"] for x in ordered])
    if not np.isfinite(vert).all() or not np.isfinite(rot).all(): raise ValueError("Invalid calibration")
    return np.tile(vert,3),np.tile(rot,3)
def decode(payload,stamp_ns,cal):
    payload=bytes(payload)
    if len(payload)!=1206: raise ValueError("Invalid VLS128 packet length")
    if payload[1205]!=161 or payload[1204] not in (55,56): raise ValueError("Unsupported sensor model/return mode")
    bank=np.ndarray((12,),dtype="<u2",buffer=payload,offset=0,strides=(100,))
    az=np.ndarray((12,),dtype="<u2",buffer=payload,offset=2,strides=(100,)).astype(float)
    if not np.array_equal(bank,BANKS) or np.any(az>=36000): raise ValueError("Invalid bank order or azimuth")
    if any(np.any(az[k:k+4]!=az[k]) for k in (0,4,8)): raise ValueError("Inconsistent firing-sequence azimuth")
    step=np.mod(np.diff(az[::4]),36000.)
    if np.any(step<=0) or np.any(step>100): raise ValueError("Unsupported within-packet rotation")
    bases=np.array([0.,step[0]/360000.,step.sum()/360000.])
    cover=np.array([step[0]/360000.,step[1]/360000.,55.241e-6])
    delta=np.array([step[0],step[1],step[1]])
    correction=np.repeat(delta/cover,128)*OFFSET_S
    base_az=np.repeat(az[::4],128)
    raw_angle=base_az+correction
    rounded=np.copysign(np.floor(np.abs(raw_angle)+.5),raw_angle).astype("int64")
    # Reproduce the upstream uint16 cast before modulo, including its wrap behavior.
    corrected=(rounded%65536)%36000
    vert,rot=prepare(cal)
    phi=corrected*np.pi/18000.-rot
    ranges=np.ndarray((12,32),dtype="<u2",buffer=payload,offset=4,strides=(100,3)).ravel().astype(float)*.004
    xyz=np.column_stack((ranges*np.cos(vert)*np.cos(phi),-ranges*np.cos(vert)*np.sin(phi),ranges*np.sin(vert)))
    times=np.int64(stamp_ns)+np.rint((np.repeat(bases,128)+OFFSET_S)*1e9).astype("int64")
    valid=ranges>0
    return dict(xyz=xyz[valid],time_ns=times[valid],distance=ranges[valid],laser=LASER[valid],point_index=np.flatnonzero(valid))
