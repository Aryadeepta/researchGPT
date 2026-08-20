"""Durable research-trial coverage; intentionally outside generic ProofBench."""
import unittest

from src.mvp_erdos791_trial import TrialSpecification, trial_attempt


class Erdos791TrialTests(unittest.TestCase):
    def test_frozen_spec_hash_excludes_retry(self):
        spec = TrialSpecification("control", 791, "url", 18, None, (), "definition")
        digest = spec.digest()
        first = trial_attempt(digest, strategy="one", revision="x", dirty=True, resource_settings={})
        second = trial_attempt(digest, strategy="two", revision="y", dirty=False, resource_settings={})
        self.assertEqual(first["trial_specification_sha256"], second["trial_specification_sha256"])
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])


if __name__ == "__main__":
    unittest.main()
