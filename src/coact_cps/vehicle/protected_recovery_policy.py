"""Protected-history conditional recovery; coordinated source forgery remains possible."""

import hashlib
from dataclasses import replace
from pathlib import Path

from .integrity_policy import IntegrityPolicyEngine
from .native_detectors import innovation_energy
from .native_policy import NativePolicyEngine
from .policy import VehicleRequest, digest
from .protected_reference import ProtectedReference


class ProtectedRecoveryPolicyEngine(IntegrityPolicyEngine):
    def __init__(
        self,
        source_keys,
        capability_key,
        permissions,
        model_config,
        thresholds,
        horizon_config,
        anchor_config,
    ):
        super().__init__(
            source_keys,
            capability_key,
            permissions,
            model_config,
            thresholds,
            "horizon_recovery",
            horizon_config,
        )
        self._protected_bound = max(1.0, anchor_config["scale"] * anchor_config["threshold"])
        ProtectedReference(self._protected_bound)
        self._protected_states = {}
        self._protected_checks = {}
        self._quarantine_started = {}
        self._recovery_paths = {}
        self._policy_digest = digest(
            dict(
                version="coact-protected-window-recovery-v1",
                protected_lag_ms=20000,
                protected_maximum_age_ms=90000,
                protected_error_bound=self._protected_bound,
                protected_implementation_sha256=hashlib.sha256(
                    Path(__file__)
                    .with_name("protected_reference.py")
                    .read_bytes()
                    .replace(b"\r\n", b"\n")
                ).hexdigest(),
                inherited=self.policy_digest,
                fallback_span_ms=self._horizon_config["horizons"][-1],
                confirmation_epochs=5,
                confirmation_span_ms=1000,
                release_quorum=3,
                implementation_sha256=hashlib.sha256(
                    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
        )

    def _protected(self, vehicle):
        if vehicle not in self._protected_states:
            self._protected_states[vehicle] = ProtectedReference(
                self._protected_bound, lag_ms=20000
            )
        return self._protected_states[vehicle]

    def _ingest_imu(self, request, frames, now):
        super()._ingest_imu(request, frames, now)
        if type(frames) is dict and "imu" in frames:
            body = self._read_source("imu", request, frames["imu"], now)
            s = body["signals"]
            self._protected(request.vehicle_id).ingest(
                s["acceleration_mps2"], s["interval_start_ms"], s["interval_end_ms"]
            )

    def _motion(self, vehicle, bodies):
        extension = self._extension(vehicle)
        primary = self._states[vehicle]
        end = bodies["imu"]["signals"]["interval_end_ms"]
        if end == extension.last_end:
            output = NativePolicyEngine._motion(self, vehicle, bodies)
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
        alarm, cold, z, statistic = NativePolicyEngine._motion(self, vehicle, bodies)
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
                if not extension.quarantined:
                    self._quarantine_started[vehicle] = end
                    self._protected(vehicle).freeze(end)
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
                fast = (
                    not shadow_alarm
                    and anchor_close
                    and stable
                    and short_clear
                    and not horizon_alarm
                )
                full_window = all(r is not None for r in horizon.residuals)
                longest = self._horizon_config["horizons"][-1]
                check = self._protected(vehicle).check(g - self._native_model.gnss_offset, w, end)
                self._protected_checks[vehicle] = check
                fresh = (
                    check.eligible
                    and full_window
                    and not horizon_alarm
                    and not shadow_alarm
                    and end - self._quarantine_started.get(vehicle, end) >= longest
                    and end - extension.last_discontinuity >= longest
                    and abs(g - self._native_model.gnss_offset - w) <= speed_guard
                )
                good = fast or fresh
                path = "close_reference" if fast else "fresh_window" if fresh else ""
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
                    self._protected(vehicle).release()
                    extension.confirm_count = 0
                    extension.confirm_start = -1
                    extension.recovery_resets += 1
                    extension.released = True
                    self._recovery_paths[vehicle] = path
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
        if (
            not extension.quarantined
            and not extension.primary_alarm
            and not extension.horizon_alarm
            and extension.horizon_ready
            and primary.quorum >= 3
            and not extension.released
        ):
            self._protected(vehicle).observe(
                float(primary.observer.x[0]), float(primary.observer.x[1]), end, quiet=True
            )
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
        if not isinstance(request, VehicleRequest):
            return super().evaluate(request, frames, now_ms)
        with self._lock:
            self._recovery_paths[request.vehicle_id] = ""
            state = self._integrity.get(request.vehicle_id)
            if state is not None:
                state.released = False
            decision = super().evaluate(request, frames, now_ms)
            if (
                self._recovery_paths.get(request.vehicle_id) == "fresh_window"
                and decision.outcome == "CHALLENGE"
            ):
                decision = self._remember(
                    replace(decision, reasons=("native_fresh_window_recovery_quorum",))
                )
            return decision

    def diagnostics(self, vehicle):
        with self._lock:
            out = super().diagnostics(vehicle)
            path = self._recovery_paths.get(vehicle, "")
            out.update(recovery_path=path, fresh_window_release=path == "fresh_window")
            state = self._protected_states.get(vehicle)
            extension = self._integrity.get(vehicle)
            current = (state is not None and state.active) or (
                extension is not None and extension.released
            )
            check = self._protected_checks.get(vehicle) if current else None
            out.update(
                primary_estimated_speed=float(self._states[vehicle].observer.x[0])
                if vehicle in self._states and self._states[vehicle].observer.x is not None
                else None,
                primary_estimated_bias=float(self._states[vehicle].observer.x[1])
                if vehicle in self._states and self._states[vehicle].observer.x is not None
                else None,
                primary_quorum=self._states[vehicle].quorum if vehicle in self._states else 0,
                protected_anchor_time_ms=check.anchor_time_ms if check else -1,
                protected_snapshot_speed=check.snapshot_speed if check else None,
                protected_snapshot_bias=check.snapshot_bias if check else None,
                protected_quarantine_start_ms=self._quarantine_started.get(vehicle, -1),
                protected_present=check.present if check else False,
                protected_lost=check.lost if check else False,
                protected_expired=check.expired if check else False,
                protected_age_ms=check.age_ms if check else -1,
                protected_prediction=check.prediction if check else None,
                protected_gnss_error=check.gnss_error if check else None,
                protected_wheel_error=check.wheel_error if check else None,
                protected_guard_pass=check.eligible if check else False,
                protected_window_release=path == "fresh_window",
                protected_error_bound=self._protected_bound,
            )
            return out
