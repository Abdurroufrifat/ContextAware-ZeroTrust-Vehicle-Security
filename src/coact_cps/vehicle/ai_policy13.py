"""The learned risk gate can add a denial but cannot clear native constraints."""

import hashlib
import math
from collections import deque
from pathlib import Path

import numpy as np

from .integrity_policy import IntegrityPolicyEngine
from .policy import digest


class LearnedRiskPolicyEngine(IntegrityPolicyEngine):
    def __init__(
        self,
        source_keys,
        capability_key,
        permissions,
        model_config,
        thresholds,
        horizon_config,
        scorer,
        score_threshold,
        model_digest,
        window=32,
    ):
        super().__init__(
            source_keys,
            capability_key,
            permissions,
            model_config,
            thresholds,
            "horizon_dual",
            horizon_config,
        )
        if (
            not callable(scorer)
            or type(window) is not int
            or not 2 <= window <= 256
            or type(score_threshold) not in (float, int)
            or not math.isfinite(score_threshold)
            or type(model_digest) is not str
            or not model_digest
        ):
            raise ValueError("frozen_scoring_contract_required")
        self._ai_scorer = scorer
        self._ai_threshold = float(score_threshold)
        self._ai_window = window
        self._ai_history = {}
        self._ai_end = {}
        self._ai_score = {}
        self._ai_alarm = {}
        self._policy_digest = digest(
            dict(
                version="coact-deny-only-ai-v13r1",
                inherited=self.policy_digest,
                model=model_digest,
                threshold=self._ai_threshold,
                window=window,
                source_sha256=hashlib.sha256(
                    Path(__file__).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
        )

    def _motion(self, vehicle, bodies):
        output = super()._motion(vehicle, bodies)
        end = bodies["imu"]["signals"]["interval_end_ms"]
        start = bodies["imu"]["signals"]["interval_start_ms"]
        if end != self._ai_end.get(vehicle):
            history = self._ai_history.setdefault(vehicle, deque(maxlen=self._ai_window))
            if vehicle in self._ai_end and start != self._ai_end[vehicle]:
                history.clear()
            history.append(
                (
                    bodies["gnss"]["signals"]["speed_mps"] - self._native_model.gnss_offset,
                    bodies["wheel"]["signals"]["speed_mps"],
                    bodies["imu"]["signals"]["acceleration_mps2"] - self._native_model.imu_offset,
                )
            )
            self._ai_end[vehicle] = end
            self._ai_score[vehicle] = None
            self._ai_alarm[vehicle] = False
            if len(history) == self._ai_window:
                self._ai_alarm[vehicle] = True
                try:
                    value = self._ai_scorer(np.asarray(history, dtype=np.float64))
                except Exception as exc:
                    raise ValueError("AI_scorer_unavailable") from exc
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError("invalid_AI_risk_score")
                self._ai_score[vehicle] = float(value)
                self._ai_alarm[vehicle] = value > self._ai_threshold
        result = (
            output[0] or self._ai_alarm.get(vehicle, False),
            output[1] or len(self._ai_history[vehicle]) < self._ai_window,
            output[2],
            output[3],
        )
        self._states[vehicle].cache_result = result
        return result

    def diagnostics(self, vehicle):
        with self._lock:
            out = super().diagnostics(vehicle)
            out.update(
                ai_score=self._ai_score.get(vehicle),
                ai_alarm=self._ai_alarm.get(vehicle, False),
                ai_window_epochs=len(self._ai_history.get(vehicle, ())),
            )
            return out
