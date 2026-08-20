import random
import tempfile
import unittest
from pathlib import Path

from tools.proofbench import proof_gym as g


class ProofGymTests(unittest.TestCase):
    def test_generates_unique_cases(self):
        rng = random.Random(1234)
        cases = []

        for level in g.LEVELS:
            batch = g.make_cases(level, 5, rng)
            self.assertEqual(len(batch), 5)
            self.assertTrue(all(c.level == level for c in batch))
            cases.extend(batch)

        self.assertEqual(
            len({c.case_id for c in cases}),
            len(cases),
        )
        self.assertEqual(
            len({c.theorem for c in cases}),
            len(cases),
        )

    def test_initial_candidate_is_not_forbidden(self):
        rng = random.Random(7)

        for level in g.LEVELS:
            for case in g.make_cases(level, 5, rng):
                self.assertIsNone(
                    g.FORBIDDEN.search(g.initial_source(case))
                )

    def test_prompt_is_local_edit_only(self):
        case = g.make_cases(
            "L1",
            1,
            random.Random(9),
        )[0]

        state = g.Validation(
            False,
            1,
            "LEAN_COMPILATION_FAILURE",
            "example compiler error",
        )

        prompt = g.make_prompt(
            case,
            g.initial_source(case),
            state,
            1,
        )

        self.assertIn(
            "PUBLIC_LOCAL_ONLY_PROOF_GYM",
            prompt,
        )
        self.assertIn(
            "complete exact replacement",
            prompt,
        )
        self.assertNotIn("REMOTE_PROGRESS", prompt)

    def test_forbidden_rejected_before_compiler(self):
        case = g.make_cases(
            "L1",
            1,
            random.Random(11),
        )[0]

        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            (ws / "Solution.lean").write_text(
                g.source(case, "by sorry")
            )

            result = g.validate(
                "/this/lean/must/not/be/run",
                ws,
                case,
            )

            self.assertFalse(result.ok)
            self.assertEqual(
                result.code,
                "PROOF_INTEGRITY_FAILURE",
            )


if __name__ == "__main__":
    unittest.main()
