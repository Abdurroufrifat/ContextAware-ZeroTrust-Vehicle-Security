import copy
import json
import tempfile
import unittest
from pathlib import Path


class ProtectedAuditTests(unittest.TestCase):
    def row(self, time=1000, **overrides):
        r = dict(
            trip="synthetic",
            case="normal",
            method="protected_recovery",
            time_ms=str(time),
            perturbed="False",
            issued_outcome="CHALLENGE",
            final_grant="False",
            capability_authorized="False",
            motion_scored="True",
            recovery_release="False",
            issued_motion_quorum="1",
            consumed_motion_quorum="1",
            primary_quorum="1",
        )
        r.update(overrides)
        return r

    def test_revoked_identity_and_postissue_grants_are_rejected(self):
        from vehicle_protected_audit11 import AccessAudit

        for case in ["identity_revoked", "postissue_revoked"]:
            with self.assertRaises(AssertionError):
                AccessAudit().observe(
                    self.row(
                        case=case,
                        perturbed="True",
                        issued_outcome="ALLOW",
                        capability_authorized="True",
                        final_grant="True",
                        issued_motion_quorum="3",
                        consumed_motion_quorum="3",
                        primary_quorum="3",
                    )
                )

    def test_authorization_without_an_allow_decision_is_rejected(self):
        from vehicle_protected_audit11 import AccessAudit

        with self.assertRaises(AssertionError):
            AccessAudit().observe(self.row(capability_authorized="True"))

    def test_premature_release_or_denial_quorum_is_rejected(self):
        from vehicle_protected_audit11 import AccessAudit

        for initial in [
            self.row(
                recovery_release="True",
                issued_motion_quorum="0",
                consumed_motion_quorum="0",
                primary_quorum="0",
            ),
            self.row(
                issued_outcome="DENY",
                motion_scored="False",
                issued_motion_quorum="0",
                consumed_motion_quorum="0",
                primary_quorum="0",
            ),
        ]:
            a = AccessAudit()
            a.observe(initial)
            with self.assertRaises(AssertionError):
                a.observe(
                    self.row(
                        1200,
                        issued_outcome="ALLOW",
                        capability_authorized="True",
                        final_grant="True",
                        issued_motion_quorum="3",
                        consumed_motion_quorum="3",
                        primary_quorum="3",
                    )
                )

    def test_three_fresh_epochs_after_release_are_sufficient(self):
        from vehicle_protected_audit11 import AccessAudit

        a = AccessAudit()
        a.observe(
            self.row(
                recovery_release="True",
                issued_motion_quorum="0",
                consumed_motion_quorum="0",
                primary_quorum="0",
            )
        )
        for i in [1, 2]:
            a.observe(
                self.row(
                    1000 + 200 * i,
                    issued_motion_quorum=str(i),
                    consumed_motion_quorum=str(i),
                    primary_quorum=str(i),
                )
            )
        a.observe(
            self.row(
                1600,
                issued_outcome="ALLOW",
                capability_authorized="True",
                final_grant="True",
                issued_motion_quorum="3",
                consumed_motion_quorum="3",
                primary_quorum="3",
            )
        )
        self.assertTrue(a.passed)

    def fixture(self):
        rows = {}
        contexts = {}
        prefix = 0.0
        anchor_prefix = None
        begin = 40200
        endgap = 42200
        for end in range(200, 120201, 200):
            present = not (begin <= end < endgap)
            if present:
                prefix -= 0.12
            if end == 10000:
                anchor_prefix = prefix
            frozen = end >= 30000
            lost = end >= endgap
            age = end - 10000
            prediction = (
                10
                + prefix
                - (anchor_prefix if anchor_prefix is not None else prefix)
                + 0.6 * age / 1000
            )
            error = abs(prediction - 10)
            r = dict(
                trip="synthetic",
                case="imu_dropout",
                method="protected_recovery",
                time_ms=str(end),
                frame=str(end // 200),
                motion_scored=str(present),
                quarantined=str(frozen),
                primary_alarm="False",
                horizon_alarm="False",
                horizon_ready="True",
                primary_quorum="3",
                recovery_release="False",
                primary_estimated_speed="10.0",
                primary_estimated_bias="-0.6",
                protected_present=str(frozen),
                protected_anchor_time_ms=str(10000 if frozen else -1),
                protected_snapshot_speed="10.0" if frozen else "",
                protected_snapshot_bias="-0.6" if frozen else "",
                protected_quarantine_start_ms=str(30000 if frozen else -1),
                protected_age_ms=str(age if frozen else -1),
                protected_prediction=str(prediction) if frozen else "",
                protected_gnss_error=str(error) if frozen else "",
                protected_wheel_error=str(error) if frozen else "",
                protected_lost=str(lost if frozen else False),
                protected_expired=str(age > 90000 if frozen else False),
                protected_error_bound="2.0",
                protected_guard_pass=str(
                    frozen and not lost and age <= 90000 and error <= 2.0 + 1e-12
                ),
                protected_window_release="False",
            )
            key = ("synthetic", "imu_dropout", end)
            rows[key] = r
            contexts[key] = dict(
                observed=dict(
                    time_ms=end,
                    interval_start_ms=end - 200,
                    gnss_speed=10.1,
                    wheel_speed=10.0,
                    imu_accel=-0.6,
                )
            )
        f = dict(
            model=dict(gnss_offset=0.1),
            session_config=dict(anchor=dict(scale=1.0, threshold=2.0)),
            protected_parameters=dict(lag_ms=20000, maximum_age_ms=90000),
        )
        return rows, contexts, f

    def test_missing_imu_is_not_integrated_and_loss_is_latched(self):
        from vehicle_protected_audit11 import audit_protected_traces

        rows, c, f = self.fixture()
        result = audit_protected_traces(rows, c, f)
        self.assertGreater(result["lost_rows"], 0)
        rows[("synthetic", "imu_dropout", 43000)]["protected_lost"] = "False"
        with self.assertRaises(AssertionError):
            audit_protected_traces(rows, c, f)

    def test_inflated_bound_false_expiry_and_old_snapshot_are_rejected(self):
        from vehicle_protected_audit11 import audit_protected_traces

        for time, field, value in [
            (50000, "protected_error_bound", "3.0"),
            (110000, "protected_expired", "False"),
            (30000, "protected_anchor_time_ms", "9800"),
        ]:
            rows, c, f = self.fixture()
            rows[("synthetic", "imu_dropout", time)][field] = value
            with self.assertRaises(AssertionError):
                audit_protected_traces(rows, c, f)

    def test_missing_interval_prediction_cannot_be_hidden_by_loss_flag(self):
        from vehicle_protected_audit11 import audit_protected_traces

        rows, c, f = self.fixture()
        rows[("synthetic", "imu_dropout", 43000)]["protected_prediction"] = "10.0"
        with self.assertRaises(AssertionError):
            audit_protected_traces(rows, c, f)

    def test_auditor_and_criteria_are_bound_to_freeze(self):
        import run_vehicle_native_stage11 as r

        self.assertIn("scripts/audit_vehicle_native_stage11.py", r.SOURCE_FILES)
        self.assertIn("scripts/vehicle_protected_audit11.py", r.SOURCE_FILES)
        origin = Path(__file__).resolve().parents[1]
        cfg = json.loads((origin / r.CONFIG).read_bytes())
        names = set(
            r.SOURCE_FILES
            + [r.CONFIG, r.DATA]
            + list(cfg["metadata_sha256"])
            + list(cfg["reference_sha256"])
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((origin / name).read_bytes())
            f = dict(
                source_sha256=r.fingerprints(root, r.SOURCE_FILES),
                config_sha256=r.sha((root / r.CONFIG).read_bytes().replace(b"\r\n", b"\n")),
                manifest_sha256=r.sha((root / r.DATA).read_bytes()),
                reference_sha256=cfg["reference_sha256"],
                metadata_sha256=cfg["metadata_sha256"],
            )
            path = root / r.CONTRACT
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(f))
            (root / r.RECEIPT).write_text(
                json.dumps(dict(contract_sha256=r.sha(path.read_bytes())))
            )
            r.check_frozen(root)
            auditor = root / "scripts/audit_vehicle_native_stage11.py"
            original = auditor.read_bytes()
            auditor.write_bytes(original + b"\n# Changed criterion\n")
            with self.assertRaises(ValueError):
                r.check_frozen(root)
            auditor.write_bytes(original)
            changed = copy.deepcopy(cfg)
            changed["promotion_gates"]["normal_minimum"] = 0.0
            (root / r.CONFIG).write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                r.check_frozen(root)

    def test_manifest_describes_no_stage11_threshold_fitting(self):
        from acquire_vehicle_native_stage11 import build_manifest

        text = " ".join(build_manifest(dict(rule="fixed"), [])["limitations"])
        self.assertNotIn("thresholds may use only3new calibration", text)

    def test_fresh_stream_cannot_claim_initial_allow_quorum(self):
        from vehicle_protected_audit11 import AccessAudit

        with self.assertRaises(AssertionError):
            AccessAudit().observe(
                self.row(
                    issued_outcome="ALLOW",
                    capability_authorized="True",
                    final_grant="True",
                    issued_motion_quorum="3",
                    consumed_motion_quorum="3",
                    primary_quorum="3",
                )
            )

    def test_context_cannot_substitute_a_different_native_recording(self):
        from vehicle_protected_audit11 import check_native_context

        expected = dict(
            frame=0, time_ms=1000, trip_id="native-route", gnss_speed=10.0, imu_accel=-0.6
        )
        check_native_context(dict(expected), expected)
        wrong = dict(expected, gnss_speed=18.0)
        with self.assertRaises(AssertionError):
            check_native_context(wrong, expected)
        wrong = dict(expected, time_ms=999)
        with self.assertRaises(AssertionError):
            check_native_context(wrong, expected)
