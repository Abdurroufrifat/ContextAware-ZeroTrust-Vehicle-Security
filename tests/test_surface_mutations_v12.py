import copy
import json
import unittest
from pathlib import Path

from coact_cps.security.external_redteam import evidence_bundle_from_dict
from coact_cps.security.registered_surface import (
    RegisteredSurfaceVerifier,
    explanation_for_registered_surface,
)


class SurfaceMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path("artifacts/hai_external_redteam/tasks/development_tasks.jsonl")
        cls.task = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        cls.bundle = evidence_bundle_from_dict(cls.task["evidence_bundle"])

    def test_registered_safe_surface_is_accepted(self):
        candidate = explanation_for_registered_surface(self.bundle, "concise-v1")
        self.assertEqual(RegisteredSurfaceVerifier().verify(self.bundle, candidate).status, "VERIFIED")

    def test_single_field_action_mutation_is_rejected(self):
        candidate = explanation_for_registered_surface(self.bundle, "concise-v1")
        mutant = copy.deepcopy(candidate)
        mutant.recommendation = "continue normal operations"
        self.assertEqual(RegisteredSurfaceVerifier().verify(self.bundle, mutant).status, "ABSTAIN")


if __name__ == "__main__":
    unittest.main()
