import json
import unittest
from dataclasses import replace
from pathlib import Path

from vehicle_native_data import NativeFrame
from vehicle_session_scenarios import perturb as old_perturb

from coact_cps.vehicle.native_detectors import ModelConfig


class BoundedAttackTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        f = json.loads(
            (
                root / "data/reference/vehicle_native_stage9/stage9_linux_frozen_transfer.json"
            ).read_bytes()
        )
        return ModelConfig(**f["model"]), dict(
            f["session_config"], civic_thresholds=f["thresholds"]["candidate"]
        )

    def frames(self, speed):
        m, c = self.config()
        return [
            NativeFrame(
                "x",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                speed + m.gnss_offset if speed else 0.0,
                speed,
                m.imu_offset,
                10,
            )
            for i in range(600)
        ]

    def test_near_stop_attacker_never_updates_with_negative_speed(self):
        from vehicle_recovery_scenarios10b import perturb

        m, c = self.config()
        frames = [
            replace(f, imu_accel=m.imu_offset - 0.3) if i >= 200 else f
            for i, f in enumerate(self.frames(0.0))
        ]
        with self.assertRaises(ValueError):
            old_perturb(frames, "observer_aware", m, c)
        changed = perturb(frames, "observer_aware", m, c)
        self.assertTrue(
            all(0 <= f.gnss_speed <= 100 and 0 <= f.wheel_speed <= 100 for f in changed)
        )
        self.assertEqual(len(changed), len(frames))

    def test_valid_interior_legacy_stream_is_bit_identical(self):
        from vehicle_recovery_scenarios10b import perturb

        m, c = self.config()
        frames = self.frames(10.0)
        self.assertEqual(
            perturb(frames, "observer_aware", m, c), old_perturb(frames, "observer_aware", m, c)
        )
