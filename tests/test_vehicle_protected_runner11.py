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
        import run_vehicle_native_stage11 as r

        seen = []
        manifest = dict(
            trips=[
                dict(id="c", phase="calibration"),
                dict(id="v", phase="validation"),
                dict(id="t", phase="test"),
            ]
        )
        trips = r.pilot_trips(Path("."), manifest, lambda root, e: seen.append(e["id"]) or [1])
        self.assertEqual(seen, ["c"])
        self.assertEqual(trips[0][0]["id"], "c")

    def test_frozen_factory_keeps_exact_settings(self):
        import run_vehicle_native_stage11 as r

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
        import run_vehicle_native_stage11 as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/vehicle_native_stage11/test").mkdir(parents=True)
            with self.assertRaises(ValueError):
                r.ensure_phase_unrun(root, "test")

    def test_persistent_step_and_cycling_labels_remain_outside_frames(self):
        from vehicle_native_data import NativeFrame
        from vehicle_protected_scenarios11 import active, perturb

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

    def test_all_seven_methods_share_one_normal_stream(self):
        import run_vehicle_native_stage11 as r
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
        self.assertEqual(len(rows), 120 * 7)
        self.assertEqual(len(context.getvalue().splitlines()), 120)
        self.assertEqual({x["method"] for x in rows}, set(r.METHODS))
        self.assertTrue(all("recovery_path" in x and "fresh_window_release" in x for x in rows))

    def test_normal_bootstrap_pools_denominators_and_names_candidate(self):
        import run_vehicle_native_stage11 as r

        metrics = [
            dict(
                trip=f"t{i}",
                case="normal",
                method=m,
                period=p,
                requests=n,
                grants=n if m == "protected_recovery" else 0,
                grant_fraction=1.0 if m == "protected_recovery" else 0.0,
            )
            for i in range(5)
            for m in r.METHODS
            for p, n in [("before", 100), ("after", 10)]
        ]
        rows = [x for x in r.paired_intervals(metrics, 10142026) if x["period"] == "overall"]
        self.assertEqual(len(rows), 6)
        self.assertTrue(
            all(
                x["candidate"] == "protected_recovery"
                and x["route_clusters"] == 5
                and x["candidate_minus_reference"] == 1.0
                for x in rows
            )
        )

    def test_whitebox_scenario_is_bounded_before_any_heldout_run(self):
        from vehicle_native_data import NativeFrame
        from vehicle_protected_scenarios11 import perturb

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

    def test_prior_selection_bundle_retains_all_three_protocols(self):
        import run_vehicle_native_stage11 as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "review"
            destination.mkdir()
            for stage in ["9", "10", "10B"]:
                raw = json.dumps(dict(stage=stage, trips=[])).encode()
                (root / f"VEHICLE_STAGE{stage}_SELECTION.json").write_bytes(raw)
            r.copy_prior_selection(root, destination)
            for stage, target in [("9", "stage9"), ("10", "stage10"), ("10B", "stage10b")]:
                self.assertEqual(
                    (root / f"VEHICLE_STAGE{stage}_SELECTION.json").read_bytes(),
                    (destination / f"{target}_selection_plan.json").read_bytes(),
                )

    def test_new_freeze_preserves_prior_parameters_without_fitting(self):
        import run_vehicle_native_stage11 as r

        f = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "data/reference/vehicle_native_stage10b/stage10b_linux_frozen_transfer.json"
            ).read_bytes()
        )
        settings = r.inherited_settings(f)
        for key in [
            "model",
            "thresholds",
            "session_config",
            "selected_factor",
            "calibration_search",
        ]:
            self.assertEqual(settings[key], f[key])
        settings["model"]["gnss_offset"] = 999
        self.assertNotEqual(settings["model"], f["model"])

    def test_pilot_directory_is_write_once(self):
        import run_vehicle_native_stage11 as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts/vehicle_native_stage11/calibration").mkdir(parents=True)
            with self.assertRaises(ValueError):
                r.ensure_phase_unrun(root, "calibration")

    def test_test_requires_audit_of_the_matching_validation_archive(self):
        import run_vehicle_native_stage11 as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "artifacts/vehicle_native_stage11"
            out.mkdir(parents=True)
            with self.assertRaises(ValueError):
                r.require_validation_audit(root, "frozen-contract")
            (out / "latest_validation_review.zip").write_bytes(b"archive")
            (out / "independent_validation_audit.json").write_text(
                json.dumps(
                    dict(
                        status="PASS",
                        phase="validation",
                        contract_sha256="wrong",
                        archive_sha256=r.sha(b"archive"),
                    )
                )
            )
            with self.assertRaises(ValueError):
                r.require_validation_audit(root, "frozen-contract")
            (out / "independent_validation_audit.json").write_text(
                json.dumps(
                    dict(
                        status="PASS",
                        phase="validation",
                        contract_sha256="frozen-contract",
                        archive_sha256=r.sha(b"archive"),
                    )
                )
            )
            r.require_validation_audit(root, "frozen-contract")

    def test_postissue_revocation_exports_the_consumption_quorum_reset(self):
        import run_vehicle_native_stage11 as r
        from vehicle_native_data import NativeFrame

        from coact_cps.vehicle.native_detectors import ModelConfig

        f = self.baseline()
        model = ModelConfig(**f["model"])
        frames = [
            NativeFrame(
                "x",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                10 + model.gnss_offset,
                10,
                model.imu_offset,
                10,
            )
            for i in range(120)
        ]
        cfg = dict(f["session_config"], civic_thresholds=f["thresholds"]["candidate"])
        rows, events = r.run_case(
            dict(id="x"),
            frames,
            frames,
            "postissue_revoked",
            model,
            f["thresholds"],
            io.StringIO(),
            cfg,
        )
        affected = [x for x in rows if x["perturbed"] and x["issued_outcome"] == "ALLOW"]
        self.assertTrue(affected)
        self.assertTrue(
            all(
                x["issued_motion_quorum"] == 3
                and x["consumed_motion_quorum"] == 0
                and not x["final_grant"]
                for x in affected
            )
        )
