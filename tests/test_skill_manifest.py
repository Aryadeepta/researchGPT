import tempfile
import unittest

from src.capabilities import SkillBuilder, SkillRegistry, SkillValidator, capability_requirement


class SkillManifestTests(unittest.TestCase):
    def test_saved_manifest_has_stable_version_identity(self):
        spec = SkillBuilder().build(capability_requirement("bounded parser", "parse", required_outputs=["out.json"]))
        with tempfile.TemporaryDirectory() as root:
            registry = SkillRegistry(root)
            first = registry.save(spec)
            second = registry.save(first)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(first["immutable_version_id"], second["immutable_version_id"])
        self.assertEqual(first["interface"]["output_schema"], ["out.json"])

    def test_permissions_and_interface_are_validated(self):
        spec = SkillBuilder().build(capability_requirement("bounded parser", "parse", required_outputs=["out.json"]))
        spec["permissions"]["filesystem"] = "all_host"
        spec["interface"] = {"input_schema": "not-a-list", "output_schema": []}
        result = SkillValidator().validate(spec)
        self.assertFalse(result["valid"])
        self.assertIn("invalid filesystem permission", result["errors"])
        self.assertIn("skill interface schemas must be lists", result["errors"])
