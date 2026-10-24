"""Conditional recovery that never replaces an enrolled session reference."""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from .native_detectors import Observer, innovation_energy
from .native_policy import NativePolicyEngine
from .policy import digest
from .session_anchor import SessionAnchor


@dataclass
class _Session:
    anchor: SessionAnchor
    shadow: Observer
    quarantined: bool = False
    confirm_count: int = 0
    confirm_start: int = -1
    last_end: int = -1
    shadow_updates: int = 0
    recovery_resets: int = 0
    primary_alarm: bool = False
    anchor_alarm: bool = False
    anchor_score: float = 0.0
    released: bool = False


class SessionPolicyEngine(NativePolicyEngine):
    def __init__(
        self,
        source_keys,
        capability_key,
        permissions,
        model_config,
        thresholds,
        variant,
        anchor_config,
    ):
        if variant not in ("anchor_dual", "anchor_recovery"):
            raise ValueError("unknown session variant")
        probe = SessionAnchor(model_config.gnss_offset, **anchor_config)
        super().__init__(
            source_keys, capability_key, permissions, model_config, "gated_dual", thresholds
        )
        self.variant = variant
        self._anchor_config = dict(scale=probe.scale, threshold=probe.threshold)
        self._sessions = {}
        self._policy_digest = digest(
            dict(
                version="coact-session-anchor-v1",
                base=self.policy_digest,
                variant=variant,
                anchor=self._anchor_config,
                enrollment_ms=5000,
                minimum_samples=10,
                recovery_epochs=5,
                recovery_span_ms=1000,
                release_quorum=3,
                source_sha256={
                    p: hashlib.sha256(
                        Path(__file__).with_name(p).read_bytes().replace(b"\r\n", b"\n")
                    ).hexdigest()
                    for p in ["session_policy.py", "session_anchor.py"]
                },
            )
        )

    def _session(self, vehicle):
        if vehicle not in self._sessions:
            self._sessions[vehicle] = _Session(
                SessionAnchor(self._native_model.gnss_offset, **self._anchor_config),
                Observer(self._native_model),
            )
        return self._sessions[vehicle]

    def _ingest_imu(self, request, frames, now):
        if type(frames) is not dict or "imu" not in frames:
            return super()._ingest_imu(request, frames, now)
        body = self._read_source("imu", request, frames["imu"], now)
        s = body["signals"]
        state = self._session(request.vehicle_id)
        same = state.anchor.end == s["interval_end_ms"]
        super()._ingest_imu(request, frames, now)
        covered = state.anchor.ingest(
            s["acceleration_mps2"], s["interval_start_ms"], s["interval_end_ms"]
        )
        if not same:
            state.shadow.predict(
                s["acceleration_mps2"], s["interval_start_ms"], s["interval_end_ms"]
            )
            if not covered:
                state.confirm_count = 0
                state.confirm_start = -1

    def _motion(self, vehicle, bodies):
        ext = self._session(vehicle)
        p = self._states[vehicle]
        end = bodies["imu"]["signals"]["interval_end_ms"]
        if end == ext.last_end:
            return super()._motion(vehicle, bodies)
        g = float(bodies["gnss"]["signals"]["speed_mps"])
        w = float(bodies["wheel"]["signals"]["speed_mps"])
        anchor = ext.anchor.sample(g, w, end)
        if ext.shadow.x is None:
            ext.shadow.initialize(g, w, end)
        r, c = ext.shadow.innovation(g, w)
        quiet = innovation_energy(r, c) <= self._thresholds["dual_chi"]
        ext.shadow.update(g, w)
        ext.shadow_updates += 1
        if p.observer.x is None:
            p.observer.initialize(g, w, end)
        x, pc, n = p.observer.x.copy(), p.observer.p.copy(), p.updates
        alarm, cold, z, statistic = super()._motion(vehicle, bodies)
        if (anchor.alarm or ext.quarantined) and p.updates > n:
            p.observer.x = x
            p.observer.p = pc
            p.updates = n
        ext.primary_alarm = alarm
        ext.anchor_alarm = anchor.alarm
        ext.anchor_score = anchor.score
        ext.released = False
        if self.variant == "anchor_recovery":
            if alarm or anchor.alarm:
                ext.quarantined = True
            if ext.quarantined:
                gap = abs(float(p.observer.x[0] - ext.shadow.x[0]))
                bound = max(1.0, self._anchor_config["threshold"] * self._anchor_config["scale"])
                good = (
                    anchor.ready and not anchor.alarm and not anchor.lost and quiet and gap <= bound
                )
                if good:
                    if ext.confirm_count == 0:
                        ext.confirm_start = end
                    ext.confirm_count += 1
                else:
                    ext.confirm_count = 0
                    ext.confirm_start = -1
                if ext.confirm_count >= 5 and end - ext.confirm_start >= 1000:
                    p.observer.x = ext.shadow.x.copy()
                    p.observer.p = ext.shadow.p.copy()
                    p.observer.time_ms = ext.shadow.time_ms
                    for d in p.statistics.values():
                        d.reset()
                    p.quorum = 0
                    ext.quarantined = False
                    ext.confirm_count = 0
                    ext.confirm_start = -1
                    ext.recovery_resets += 1
                    ext.released = True
                    alarm = False
                    cold = True
                else:
                    alarm = True
        alarm = alarm or anchor.alarm
        output = (alarm, cold or p.quorum < 3, z, max(statistic, anchor.score))
        p.alarm = alarm
        p.statistic = output[3]
        p.cache_result = output
        ext.last_end = end
        return output

    def evaluate(self, request, frames, now_ms):
        with self._lock:
            d = super().evaluate(request, frames, now_ms)
            state = self._sessions.get(request.vehicle_id)
            if state is not None and d.outcome == "DENY":
                if d.reasons == ("native_motion_statistic_alarm",):
                    reason = (
                        "native_session_reference_lost"
                        if state.anchor.lost
                        else "native_session_anchor_alarm"
                        if state.anchor_alarm
                        else "native_session_quarantine"
                        if state.quarantined
                        else "native_motion_statistic_alarm"
                    )
                    d = self._remember(replace(d, reasons=(reason,)))
                else:
                    state.confirm_count = 0
                    state.confirm_start = -1
                    state.anchor.invalidate_context()
            return d

    def diagnostics(self, vehicle):
        with self._lock:
            out = super().diagnostics(vehicle)
            s = self._sessions.get(vehicle)
            out.update(
                quarantined=False,
                confirmation_epochs=0,
                recovery_resets=0,
                shadow_updates=0,
                anchor_ready=False,
                anchor_lost=False,
                anchor_bias=None,
                anchor_samples=0,
                anchor_alarm=False,
                anchor_score=0.0,
                primary_alarm=False,
                released=False,
            )
            if s is not None:
                out.update(
                    quarantined=s.quarantined,
                    confirmation_epochs=s.confirm_count,
                    recovery_resets=s.recovery_resets,
                    shadow_updates=s.shadow_updates,
                    anchor_ready=s.anchor.bias is not None,
                    anchor_lost=s.anchor.lost,
                    anchor_bias=s.anchor.bias,
                    anchor_samples=len(s.anchor.samples),
                    anchor_alarm=s.anchor_alarm,
                    anchor_score=s.anchor_score,
                    primary_alarm=s.primary_alarm,
                    released=s.released,
                )
            return out
