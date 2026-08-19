#!/usr/bin/env python3
"""Read-only source and Git history checks used before a public release."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXCLUDED_DIRS = {
    ".git", ".codex-logs", ".codex-prompts", ".research-artifacts",
    ".research-runs", ".literature-cache", ".local-model-cache",
    ".paper-runs", ".pytest_cache", "__pycache__", ".mypy_cache", "venv",
    ".researchgpt-codegraph",
}
EXCLUDED_SUFFIXES = (".gguf", ".safetensors", ".Zone.Identifier")
PATTERNS = {
    "GOOGLE_API_KEY": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "GITHUB_TOKEN": re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "OPENAI_KEY": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "ANTHROPIC_KEY": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "HUGGINGFACE_TOKEN": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "AWS_ACCESS_KEY": re.compile(r"AKIA[0-9A-Z]{16}"),
    "PRIVATE_KEY": re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----"),
    "BEARER_CREDENTIAL": re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    "ASSIGNED_SECRET": re.compile(r"(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^\s'\"$]{12,}['\"]"),
}


@dataclass(frozen=True)
class Finding:
    category: str
    location: str
    redacted: str


def redact(value: str) -> str:
    return value[:4] + "...REDACTED" if value else "REDACTED"


def scan_text(text: str, location: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        # A GitHub expression is a reference, never a secret value by itself.
        if "${{ secrets." in line:
            continue
        for category, pattern in PATTERNS.items():
            for match in pattern.finditer(line):
                findings.append(Finding(category, f"{location}:{line_number}", redact(match.group(0))))
    return findings


def candidate_files(root: Path) -> Iterable[Path]:
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
        for name in files:
            path = Path(directory) / name
            if path.suffix in EXCLUDED_SUFFIXES or path.name.endswith(EXCLUDED_SUFFIXES):
                continue
            yield path


def scan_worktree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in candidate_files(root):
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            findings.extend(scan_text(path.read_text(encoding="utf-8", errors="replace"), str(path.relative_to(root))))
        except OSError:
            continue
    return findings


def git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root, input=input, stderr=subprocess.DEVNULL)


def scan_history(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    seen_blobs: set[str] = set()
    commits = git(root, "rev-list", "--all").decode().splitlines()
    for commit in commits:
        rows = git(root, "ls-tree", "-r", "-z", commit).split(b"\0")
        for row in rows:
            if not row:
                continue
            metadata, path = row.split(b"\t", 1)
            _, kind, object_id = metadata.decode().split()
            if kind != "blob" or object_id in seen_blobs:
                continue
            seen_blobs.add(object_id)
            try:
                content = git(root, "cat-file", "blob", object_id)
            except subprocess.CalledProcessError:
                continue
            if len(content) > 8 * 1024 * 1024:
                continue
            findings.extend(scan_text(content.decode("utf-8", errors="replace"), f"{commit[:12]}:{path.decode(errors='replace')}"))
    return findings


def audit_workflows(root: Path) -> list[str]:
    failures: list[str] = []
    workflow_dir = root / ".github" / "workflows"
    for path in sorted(workflow_dir.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8", errors="replace")
        has_secret = "secrets." in text
        untrusted_trigger = any(trigger in text for trigger in ("pull_request_target:", "issue_comment:", "workflow_run:", "repository_dispatch:"))
        if has_secret and untrusted_trigger:
            failures.append(f"UNSAFE_SECRET_WORKFLOW:{path.relative_to(root)}")
        if has_secret and re.search(r"(?m)^\s*(?:-|)\s*(?:run:\s*)?.*(?:printenv|set -x|env\s*$)", text):
            failures.append(f"SECRET_LOGGING_RISK:{path.relative_to(root)}")
    return failures


def github_audit(root: Path, reasons: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    if shutil.which("gh") is None:
        reasons.append("GITHUB_AUDIT_BLOCKED: gh is unavailable")
        return findings
    try:
        repo = git(root, "config", "--get", "remote.origin.url").decode().strip()
        match = re.search(r"github\.com[/:]([^/]+/[^/.]+)(?:\.git)?$", repo)
        if not match:
            reasons.append("GITHUB_AUDIT_BLOCKED: origin is not a GitHub repository")
            return findings
        name = match.group(1)
        subprocess.check_output(["gh", "api", f"repos/{name}"], stderr=subprocess.DEVNULL, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        reasons.append("GITHUB_AUDIT_BLOCKED: gh authentication or GitHub API access failed")
        return findings
    responses: dict[str, dict] = {}
    for endpoint, label in ((f"repos/{name}/actions/secrets", "secrets metadata"),
                            (f"repos/{name}/actions/runs?per_page=100", "Actions runs"),
                            (f"repos/{name}/actions/artifacts?per_page=100", "Actions artifacts")):
        try:
            responses[label] = json.loads(subprocess.check_output(
                ["gh", "api", endpoint], stderr=subprocess.DEVNULL, timeout=45))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            reasons.append(f"GITHUB_AUDIT_BLOCKED: cannot enumerate {label}")
    if "Actions runs" not in responses or "Actions artifacts" not in responses:
        return findings
    for run in responses["Actions runs"].get("workflow_runs", []):
        run_id = str(run.get("id", "unknown"))
        try:
            log = subprocess.check_output(["gh", "run", "view", run_id, "--repo", name, "--log"],
                                          stderr=subprocess.DEVNULL, timeout=120).decode("utf-8", "replace")
            findings.extend(scan_text(log, f"github-actions-log:{run_id}"))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            reasons.append(f"GITHUB_AUDIT_BLOCKED: cannot inspect Actions log for run {run_id}")
    with tempfile.TemporaryDirectory(prefix="researchgpt-release-audit-") as temp:
        for artifact in responses["Actions artifacts"].get("artifacts", []):
            artifact_id = str(artifact.get("id", "unknown"))
            size = int(artifact.get("size_in_bytes") or 0)
            if size > 100 * 1024 * 1024:
                reasons.append(f"REMOTE_ARTIFACT_AUDIT_INCOMPLETE:{artifact_id} exceeds 100 MiB")
                continue
            archive = Path(temp) / f"{artifact_id}.zip"
            try:
                with archive.open("wb") as output:
                    subprocess.run(["gh", "api", f"repos/{name}/actions/artifacts/{artifact_id}/zip"],
                                   stdout=output, stderr=subprocess.DEVNULL, timeout=120, check=True)
                with zipfile.ZipFile(archive) as contents:
                    for member in contents.infolist():
                        if member.is_dir() or member.file_size > 8 * 1024 * 1024:
                            continue
                        findings.extend(scan_text(contents.read(member).decode("utf-8", "replace"),
                                                  f"github-artifact:{artifact_id}:{member.filename}"))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, zipfile.BadZipFile):
                reasons.append(f"GITHUB_AUDIT_BLOCKED: cannot inspect Actions artifact {artifact_id}")
    return findings


def render(status: str, reasons: list[str], findings: list[Finding]) -> str:
    lines = ["# Public Release Audit", "", "This report never includes full credential values.", ""]
    lines += [f"- Findings: {len(findings)}", f"- Status: `{status}`", ""]
    if reasons:
        lines += ["## Reasons", ""] + [f"- {reason}" for reason in reasons] + [""]
    if findings:
        lines += ["## Redacted findings", ""] + [f"- `{f.category}` at `{f.location}`: `{f.redacted}`" for f in findings] + [""]
    lines.append(f"PUBLIC_RELEASE_READY={status}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only public-release secret audit")
    parser.add_argument("--local-only", action="store_true", help="do not contact GitHub")
    parser.add_argument("--github", action="store_true", help="require GitHub-side audit")
    parser.add_argument("--report", type=Path, help="write a redacted Markdown report")
    args = parser.parse_args(argv)
    root = Path.cwd()
    findings = scan_worktree(root) + scan_history(root)
    reasons = audit_workflows(root)
    if findings:
        reasons.append("SECRET_MATERIAL_FOUND")
    if not args.local_only:
        findings.extend(github_audit(root, reasons))
    status = "FAIL" if any(not r.startswith("GITHUB_AUDIT_BLOCKED") for r in reasons) else ("BLOCKED" if reasons else "PASS")
    report = render(status, reasons, findings)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
