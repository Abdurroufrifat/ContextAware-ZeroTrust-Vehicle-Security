"""Synthetic truth/witness factory. Never imported by the policy implementation."""
import json
import math
import random
from dataclasses import replace
from pathlib import Path

from coact_cps.vehicle.native_detectors import ModelConfig
from vehicle_native_data import NativeFrame
from vehicle_recovery_scenarios10b import perturb


def reference(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    return json.loads((root/'data/reference/vehicle_native_stage10b/stage10b_linux_frozen_transfer.json').read_bytes())


def trajectory(seed, frames=600, model=None):
    if type(seed) is not int or type(frames) is not int or not 1<=frames<=10000:
        raise ValueError('bounded_synthetic_trajectory_required')
    m = model or ModelConfig(**reference()['model'])
    rng = random.Random(seed)
    phase = rng.uniform(0,2*math.pi)
    def speed(t):
        return 15+2*math.sin(.08*t+phase)+.6*math.sin(.23*t+phase)
    rows=[]
    for i in range(frames):
        t=i*.2
        truth=speed(t)
        a=(truth-speed(t-.2))/.2
        f=NativeFrame(f'synthetic12_{seed}',i,1000+i*200,800+i*200,
            1000+i*200,1000+i*200,1000+i*200,
            truth+m.gnss_offset+rng.uniform(-.03,.03),truth+rng.uniform(-.02,.02),
            a+m.imu_offset,10)
        rows.append((f,truth,rng.uniform(-.1,.1)))
    return rows


def changed_stream(rows, case, model, session):
    frames=[r[0] for r in rows]
    n=len(frames);begin=n//3;end=2*n//3
    if case=='observer_aware':
        return perturb(frames,case,model,session)
    changed=[]
    for f in frames:
        on=begin<=f.frame<end
        shift=0.;a=0.
        if case=='joint_jump' and on:shift=5.
        if case=='persistent_joint_step' and f.frame>=begin:shift=5.
        if case in {'startup_common_bias','witness_compromised'}:shift=5.
        if case=='coherent_all_sources' and f.frame>=begin:
            shift=5+.05*(f.frame-begin)*.2;a=.05
        changed.append(replace(f,gnss_speed=f.gnss_speed+shift,wheel_speed=f.wheel_speed+shift,imu_accel=f.imu_accel+a))
    return changed


def period(frame,n,case):
    if case in {'startup_common_bias','witness_compromised'}:return 'active'
    if frame<n//3:return 'before'
    if frame<2*n//3 or case in {'persistent_joint_step','coherent_all_sources'}:return 'active'
    return 'after'
