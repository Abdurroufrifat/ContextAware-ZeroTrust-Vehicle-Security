"""Conditional consistency recovery; does not attest physical sensor truth."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from .integrity_window import HorizonMonitor
from .native_detectors import Observer, innovation_energy
from .native_policy import NativePolicyEngine
from .policy import digest

VARIANTS = ("recovery_dual", "horizon_dual", "horizon_recovery")


@dataclass
class _Integrity:
    monitor: HorizonMonitor
    shadow: Observer
    quarantined: bool = False
    confirm_count: int = 0
    confirm_start: int = -1
    recovery_resets: int = 0
    shadow_updates: int = 0
    last_end: int = -1
    horizon_score: float = 0.0
    horizon_ready: bool = False
    horizon_alarm: bool = False
    primary_alarm: bool = False
    shadow_alarm: bool = False
    released: bool = False
    last_discontinuity: int = -1


class IntegrityPolicyEngine(NativePolicyEngine):
    def __init__(
        self,
        source_keys,
        capability_key,
        permissions,
        model_config,
        thresholds,
        variant,
        horizon_config,
    ):
        if variant not in VARIANTS:
            raise ValueError("unknown integrity variant")
        # Validate/freeze caller-supplied configuration before constructing state.
        probe = HorizonMonitor(
            model_config.imu_offset,
            tuple(horizon_config["horizons"]),
            tuple(horizon_config["scales"]),
            horizon_config["threshold"],
        )
        super().__init__(
            source_keys, capability_key, permissions, model_config, "gated_dual", thresholds
        )
        self.variant = variant
        self._has_horizon = variant != "recovery_dual"
        self._has_recovery = variant != "horizon_dual"
        self._horizon_config = dict(
            horizons=probe.horizons, scales=probe.scales, threshold=probe.threshold
        )
        self._integrity = {}
        self._policy_digest = digest(
            dict(
                version="native-integrity-recovery-v1",
                base=self.policy_digest,
                variant=variant,
                horizons=self._horizon_config,
                recovery_epochs=5,
                recovery_span_ms=1000,
                release_quorum=3,
                implementation_sha256=hashlib.sha256(
                    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
                window_implementation_sha256=hashlib.sha256(
                    Path(__file__)
                    .with_name("integrity_window.py")
                    .read_bytes()
                    .replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
        )

    def _extension(self, vehicle):
        if vehicle not in self._integrity:
            self._integrity[vehicle] = _Integrity(
                HorizonMonitor(self._native_model.imu_offset, **self._horizon_config),
                Observer(self._native_model),
            )
        return self._integrity[vehicle]

    def _ingest_imu(self, request, frames, now):
        if type(frames) is not dict or "imu" not in frames:
            return super()._ingest_imu(request, frames, now)
        body = self._read_source("imu", request, frames["imu"], now)
        state = self._extension(request.vehicle_id)
        signals = body["signals"]
        end = signals["interval_end_ms"]
        start = signals["interval_start_ms"]
        a = signals["acceleration_mps2"]
        same = end == state.monitor.end
        super()._ingest_imu(request, frames, now)
        covered = state.monitor.ingest(a, start, end)
        if not same:
            state.shadow.predict(a, start, end)
            if not covered:
                state.confirm_count = 0
                state.confirm_start = -1

    def _motion(self, vehicle, bodies):
        extension = self._extension(vehicle)
        primary = self._states[vehicle]
        end = bodies["imu"]["signals"]["interval_end_ms"]
        if end == extension.last_end:
            output = super()._motion(vehicle, bodies)
            return output[0], output[1] or primary.quorum < 3, output[2], output[3]
        g = float(bodies["gnss"]["signals"]["speed_mps"])
        w = float(bodies["wheel"]["signals"]["speed_mps"])
        horizon = extension.monitor.sample(g, w, end)
        speed_guard = max(
            1.0, self._horizon_config["threshold"] * self._horizon_config["scales"][0]
        )
        if horizon.instantaneous_error > speed_guard:
            extension.last_discontinuity = end
        if extension.shadow.x is None:
            extension.shadow.initialize(g, w, end)
        residual, covariance = extension.shadow.innovation(g, w)
        shadow_z = innovation_energy(residual, covariance)
        shadow_alarm = shadow_z > self._thresholds["dual_chi"]
        extension.shadow.update(g, w)
        extension.shadow_updates += 1
        if primary.observer.x is None:
            # Missing IMU continuity requires conditional fresh enrollment.
            # Normal window/quarantine confirmation still applies.
            primary.observer.initialize(g, w, end)
        previous_x = None if primary.observer.x is None else primary.observer.x.copy()
        previous_p = None if primary.observer.p is None else primary.observer.p.copy()
        previous_updates = primary.updates
        alarm, cold, z, statistic = super()._motion(vehicle, bodies)
        horizon_alarm = self._has_horizon and horizon.alarm
        if (horizon_alarm or extension.quarantined) and primary.updates > previous_updates:
            primary.observer.x = previous_x
            primary.observer.p = previous_p
            primary.updates = previous_updates
        extension.primary_alarm = alarm
        extension.shadow_alarm = shadow_alarm
        extension.horizon_alarm = horizon_alarm
        extension.horizon_ready = horizon.ready
        extension.horizon_score = horizon.score
        extension.released = False
        if self._has_recovery:
            if alarm or horizon_alarm:
                extension.quarantined = True
            if extension.quarantined:
                short_clear = (
                    horizon.ready
                    and horizon.residuals[0] / self._horizon_config["scales"][0]
                    <= self._horizon_config["threshold"]
                )
                anchor_close = (
                    abs(float(primary.observer.x[0] - extension.shadow.x[0])) <= speed_guard
                )
                stable = end - extension.last_discontinuity >= self._horizon_config["horizons"][0]
                good = (
                    not shadow_alarm
                    and anchor_close
                    and stable
                    and (not self._has_horizon or short_clear)
                )
                if good:
                    if extension.confirm_count == 0:
                        extension.confirm_start = end
                    extension.confirm_count += 1
                else:
                    extension.confirm_count = 0
                    extension.confirm_start = -1
                if extension.confirm_count >= 5 and end - extension.confirm_start >= 1000:
                    primary.observer.x = extension.shadow.x.copy()
                    primary.observer.p = extension.shadow.p.copy()
                    primary.observer.time_ms = extension.shadow.time_ms
                    for detector in primary.statistics.values():
                        detector.reset()
                    primary.quorum = 0
                    extension.quarantined = False
                    extension.confirm_count = 0
                    extension.confirm_start = -1
                    extension.recovery_resets += 1
                    extension.released = True
                    alarm = False
                    horizon_alarm = False
                    cold = True
                    while (
                        len(extension.monitor.history) > 1
                        and extension.monitor.history[1][0]
                        <= end - self._horizon_config["horizons"][0]
                    ):
                        extension.monitor.history.popleft()
                else:
                    alarm = True
        alarm = alarm or horizon_alarm
        cold = cold or (self._has_horizon and not horizon.ready) or primary.quorum < 3
        statistic = max(statistic, horizon.score) if self._has_horizon else statistic
        output = (alarm, cold, z, statistic)
        primary.alarm = alarm
        primary.statistic = statistic
        primary.cache_result = output
        extension.last_end = end
        return output

    def evaluate(self, request, frames, now_ms):
        with self._lock:
            decision = super().evaluate(request, frames, now_ms)
            state = self._integrity.get(request.vehicle_id)
            if state is not None:
                if decision.reasons == ("native_motion_statistic_alarm",):
                    reason = (
                        "native_integrity_horizon_alarm"
                        if state.horizon_alarm
                        else "native_integrity_quarantine"
                        if state.quarantined
                        else "native_motion_statistic_alarm"
                    )
                    decision = self._remember(replace(decision, reasons=(reason,)))
                elif decision.outcome == "DENY":
                    state.confirm_count = 0
                    state.confirm_start = -1
            return decision

    def diagnostics(self, vehicle):
        with self._lock:
            output = super().diagnostics(vehicle)
            state = self._integrity.get(vehicle)
            output.update(
                quarantined=False,
                confirmation_epochs=0,
                recovery_resets=0,
                shadow_updates=0,
                horizon_samples=0,
                horizon_score=0.0,
                horizon_ready=False,
                horizon_alarm=False,
                primary_alarm=False,
                shadow_alarm=False,
                released=False,
            )
            if state is not None:
                output.update(
                    quarantined=state.quarantined,
                    confirmation_epochs=state.confirm_count,
                    recovery_resets=state.recovery_resets,
                    shadow_updates=state.shadow_updates,
                    horizon_samples=len(state.monitor.history),
                    horizon_score=state.horizon_score,
                    horizon_ready=state.horizon_ready,
                    horizon_alarm=state.horizon_alarm,
                    primary_alarm=state.primary_alarm,
                    shadow_alarm=state.shadow_alarm,
                    released=state.released,
                )
            return output
