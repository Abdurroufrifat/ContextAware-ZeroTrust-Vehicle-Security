import io
import json
import tempfile
import unittest
from pathlib import Path


class RecoveryRunnerTests(unittest.TestCase):
    def baseline(self):
        root = Path(__file__).resolve().parents[1]
        return json.loads(
            (
                root / "data/reference/vehicle_native_stage9/stage9_linux_frozen_transfer.json"
            ).read_bytes()
        )

    def test_calibration_never_loads_validation_or_test(self):
        import run_vehicle_native_stage10b as r

        seen = []
        manifest = dict(
            trips=[
                dict(id="c", phase="calibration"),
                dict(id="v", phase="validation"),
                dict(id="t", phase="test"),
            ]
        )
        trips = r.calibration_trips(
            Path("."), manifest, lambda root, e: seen.append(e["id"]) or [1]
        )
        self.assertEqual(seen, ["c"])
        self.assertEqual(trips[0][0]["id"], "c")

    def test_first_factor_uses_full_blocked_fraction_and_worst_route(self):
        import run_vehicle_native_stage10b as r

        f = self.baseline()
        original = dict(f["thresholds"]["candidate"])

        def measure(trips, thresholds):
            factor = thresholds["dual_chi"] / original["dual_chi"]
            return {
                "a": dict(blocked_fraction=0.04 if factor >= 2 else 0.11),
                "b": dict(blocked_fraction=0.01),
                "c": dict(blocked_fraction=0.01),
            }

        factor, thresholds, search = r.select_factor([], original, measure)
        self.assertEqual(factor, 2)
        self.assertEqual(len(search), 2)
        self.assertEqual(thresholds["chi"], original["chi"])
        self.assertEqual(original, f["thresholds"]["candidate"])

    def test_calibration_grid_exhaustion_stops_heldout_evaluation(self):
        import run_vehicle_native_stage10b as r

        with self.assertRaisesRegex(ValueError, "exhausted") as captured:
            r.select_factor(
                [],
                self.baseline()["thresholds"]["candidate"],
                lambda t, h: {"x": dict(blocked_fraction=0.11)},
            )
        self.assertEqual([x["factor"] for x in captured.exception.search], [1, 2, 4, 8, 16, 32])

    def test_frozen_factory_keeps_exact_settings(self):
        import run_vehicle_native_stage10b as r

        from coact_cps.vehicle.native_detectors import ModelConfig

        f = self.baseline()
        cfg = dict(
            f["session_config"],
            civic_thresholds=dict(f["thresholds"]["candidate"], dual_chi=1000.0, dual_cusum=2000.0),
        )
        e, k = r.new_engine(ModelConfig(**f["model"]), "frozen_matched", f["thresholds"], cfg)
        self.assertEqual(dict(e._thresholds), f["thresholds"]["candidate"])
        e, k = r.new_engine(ModelConfig(**f["model"]), "civic_matched", f["thresholds"], cfg)
        self.assertEqual(dict(e._thresholds), cfg["civic_thresholds"])

    def test_partial_phase_refuses_recomputation(self):
        import run_vehicle_native_stage10b as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/vehicle_native_stage10b/test").mkdir(parents=True)
            with self.assertRaises(ValueError):
                r.ensure_phase_unrun(root, "test")

    def test_persistent_step_and_cycling_labels_remain_outside_frames(self):
        from vehicle_native_data import NativeFrame
        from vehicle_recovery_scenarios10b import active, perturb

        from coact_cps.vehicle.native_detectors import ModelConfig

        f = self.baseline()
        frames = [
            NativeFrame(
                "x",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                10.0,
                10.0,
                0.0,
                10,
            )
            for i in range(600)
        ]
        cfg = dict(f["session_config"], civic_thresholds=f["thresholds"]["candidate"])
        changed = perturb(frames, "persistent_joint_step", ModelConfig(**f["model"]), cfg)
        self.assertTrue(active(frames[-1], frames, "persistent_joint_step"))
        self.assertEqual(changed[-1].gnss_speed, 15.0)
        self.assertEqual(set(changed[0].__dict__), set(frames[0].__dict__))
        pulse = perturb(frames, "recovery_cycling", ModelConfig(**f["model"]), cfg)
        self.assertTrue(any(x.gnss_speed == 15 for x in pulse))
        self.assertTrue(any(x.gnss_speed == 10 for x in pulse[220:]))

    def test_all_six_methods_share_one_normal_stream(self):
        import run_vehicle_native_stage10b as r
        from vehicle_native_data import NativeFrame

        from coact_cps.vehicle.native_detectors import ModelConfig

        f = self.baseline()
        frames = [
            NativeFrame(
                "x",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                10 + f["model"]["gnss_offset"],
                10.0,
                f["model"]["imu_offset"],
                10,
            )
            for i in range(120)
        ]
        cfg = dict(f["session_config"], civic_thresholds=f["thresholds"]["candidate"])
        context = io.StringIO()
        rows, events = r.run_case(
            dict(id="x"),
            frames,
            frames,
            "normal",
            ModelConfig(**f["model"]),
            f["thresholds"],
            context,
            cfg,
        )
        self.assertEqual(len(rows), 120 * 6)
        self.assertEqual(len(context.getvalue().splitlines()), 120)
        self.assertEqual({x["method"] for x in rows}, set(r.METHODS))
        self.assertTrue(all("recovery_path" in x and "fresh_window_release" in x for x in rows))

    def test_normal_bootstrap_pools_denominators_and_names_candidate(self):
        import run_vehicle_native_stage10b as r

        metrics = [
            dict(
                trip=f"t{i}",
                case="normal",
                method=m,
                period=p,
                requests=n,
                grants=n if m == "civic_recovery" else 0,
                grant_fraction=1.0 if m == "civic_recovery" else 0.0,
            )
            for i in range(5)
            for m in r.METHODS
            for p, n in [("before", 100), ("after", 10)]
        ]
        rows = [x for x in r.paired_intervals(metrics, 10142026) if x["period"] == "overall"]
        self.assertEqual(len(rows), 5)
        self.assertTrue(
            all(
                x["candidate"] == "civic_recovery"
                and x["route_clusters"] == 5
                and x["candidate_minus_reference"] == 1.0
                for x in rows
            )
        )

    def test_whitebox_scenario_is_bounded_before_any_heldout_run(self):
        from vehicle_native_data import NativeFrame
        from vehicle_recovery_scenarios10b import perturb

        from coact_cps.vehicle.native_detectors import ModelConfig

        f = self.baseline()
        frames = [
            NativeFrame(
                "x",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                10.0,
                10.0,
                0.0,
                10,
            )
            for i in range(600)
        ]
        cfg = dict(
            f["session_config"], civic_thresholds=dict(f["thresholds"]["candidate"], dual_chi=1e6)
        )
        changed = perturb(frames, "v2_observer_aware", ModelConfig(**f["model"]), cfg)
        self.assertTrue(
            all(10 <= x.gnss_speed <= 18 and 10 <= x.wheel_speed <= 18 for x in changed)
        )

    def test_review_bundle_preserves_prior_date_selection_bytes(self):
        import run_vehicle_native_stage10b as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "review"
            destination.mkdir()
            raw = b'{"trips":[{"day":"2018-05-01"}]}\n'
            (root / "VEHICLE_STAGE9_SELECTION.json").write_bytes(raw)
            r.copy_prior_selection(root, destination)
            self.assertEqual((destination / "stage9_selection_plan.json").read_bytes(), raw)
