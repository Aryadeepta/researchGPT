import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.public_release_audit import candidate_files, redact, scan_history, scan_text, scan_worktree

AUDITOR = Path(__file__).parents[1] / "tools" / "public_release_audit.py"


class PublicReleaseAuditTests(unittest.TestCase):
    def test_detects_fake_google_key_and_redacts_it(self):
        fake = "AIza" + "A" * 32
        findings = scan_text(f"KEY={fake}", "test")
        self.assertEqual(findings[0].category, "GOOGLE_API_KEY")
        self.assertNotIn(fake, findings[0].redacted)
        self.assertEqual(findings[0].redacted, "AIza...REDACTED")

    def test_detects_fake_github_token_and_private_key(self):
        text = "ghp_" + "a" * 36 + "\n-----BEGIN" + " PRIVATE KEY-----"
        categories = {item.category for item in scan_text(text, "test")}
        self.assertEqual(categories, {"GITHUB_TOKEN", "PRIVATE_KEY"})

    def test_secret_reference_is_not_a_leak(self):
        self.assertEqual(scan_text("KEY: ${{ secrets.GEMINI_API_KEY }}", "workflow"), [])

    def test_excludes_local_state_and_models_but_not_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".codex-logs").mkdir()
            (root / ".research-artifacts").mkdir()
            (root / "model.gguf").write_text("ignored")
            (root / "src.py").write_text("x = 1")
            paths = {path.relative_to(root).as_posix() for path in candidate_files(root)}
            self.assertIn("src.py", paths)
            self.assertNotIn("model.gguf", paths)
            self.assertNotIn(".codex-logs/x", paths)
            self.assertEqual(scan_worktree(root), [])

    def test_history_detects_fake_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            fake = "AIza" + "B" * 32
            (root / "old.env").write_text(fake)
            subprocess.run(["git", "add", "old.env"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "old"], cwd=root, check=True)
            findings = scan_history(root)
            self.assertEqual(len(findings), 1)
            self.assertNotIn(fake, findings[0].redacted)

    def test_history_scan_does_not_mutate_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "source.py").write_text("value = 1\n")
            subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
            scan_history(root)
            self.assertEqual(before, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root))
            self.assertEqual(subprocess.check_output(["git", "status", "--porcelain"], cwd=root), b"")

    def test_clean_repository_passes_local_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "source.py").write_text("value = 1\n")
            subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            result = subprocess.run([sys.executable, str(AUDITOR), "--local-only"], cwd=root,
                                    check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PUBLIC_RELEASE_READY=PASS", result.stdout)

    def test_redact_never_returns_full_value(self):
        self.assertEqual(redact("sk-" + "x" * 40), "sk-x...REDACTED")
