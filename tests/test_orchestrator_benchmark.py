import unittest

from src.orchestrator_benchmark import BenchmarkMode, BenchmarkTask, benchmark_record, benchmark_schema, public_demo_seed


class BenchmarkIsolationTests(unittest.TestCase):
    def test_oblivious_baseline_cannot_create_trusted_package(self):
        task = BenchmarkTask("t1", "algorithm", "bounded property", ("pytest",))
        record = benchmark_record(task, BenchmarkMode.VERIFICATION_OBLIVIOUS_BASELINE)
        self.assertFalse(record["normal_research_package_allowed"])
        self.assertEqual(record["trust_status"], "UNTRUSTED_BENCHMARK_ONLY")

    def test_schema_and_demo_are_declared_without_results(self):
        self.assertIn("FULL", benchmark_schema()["modes"])
        self.assertIn("test_failure", public_demo_seed()["expected_workflow"])
