"""Bounded, public-only Kimina-Prover helpers for ProofBench.

This module deliberately knows nothing about the generic local-model router.
Kimina output is untrusted text: callers must submit every extracted candidate
to Lean before using it as proof-search state.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = Path.home() / ".cache/researchgpt-models/kimina/Kimina-Prover-RL-1.7B.Q4_K_M.gguf"
DEFAULT_LLAMA = Path("/home/aryad/llama.cpp-direct/build/bin/llama-cli")
EXPECTED_Q4_K_M_SHA256 = "3ee90dd18b132f536aec0c024e834ebb56ea109f07b291e2f9198fb4e6d594e5"
HIDDEN_MARKERS = ("hidden", "oracle", "held-out", "heldout", "controller secret", "private")
MAX_RAW_OUTPUT = 12000
MAX_CANDIDATE_CHARS = 2000
MAX_CANDIDATES = 32
MAX_CANDIDATE_LINES = 12

TACTIC_START = re.compile(r"^\s*(?:rfl\b|assumption\b|constructor\b|intro\b|intros\b|exact\b|apply\b|refine\b|rw\b|simp\b|simpa\b|have\b|show\b|change\b|cases\b|induction\b|decide\b|omega\b|norm_num\b|aesop\b|calc\b)")


def bounded(text, limit):
    return str(text or "")[-limit:]


def assert_public(value):
    """Reject any prompt capsule containing private/held-out material."""
    raw = json.dumps(value, sort_keys=True).lower()
    if any(marker in raw for marker in HIDDEN_MARKERS):
        raise ValueError("KIMINA_PROMPT_PRIVATE_MARKER")
    return value


def public_kimina_prompt(*, declaration, goal, prefix, rejected):
    capsule = assert_public({
        "declaration": declaration,
        "goal": goal,
        "prefix": prefix,
        "rejected": rejected,
    })
    return """Continue this PUBLIC Lean 4 proof state. Return Lean proof/tactic text after any reasoning.\n\n"
        "Current exact Lean goal/context (authoritative):\n{goal}\n\n"
        "Validated tactic prefix (ALREADY EXECUTED successfully; continue after it, do NOT restart the theorem or repeat it):\n{prefix}\n\n"
        "Original theorem declaration (context only):\n{declaration}\n\n"
        "Recent rejected public tactics and Lean diagnostics (do not repeat diagnostic prose):\n{rejected}\n\n"
        "Provide a valid Lean proof continuation. Do not discuss private tests or use placeholders.""".format(
            declaration=capsule["declaration"], goal=capsule["goal"], prefix=capsule["prefix"],
            rejected=json.dumps(capsule["rejected"], sort_keys=True),
        )


def clean_block(block):
    lines = [line.rstrip() for line in block.strip().splitlines()]
    for index, line in enumerate(lines):
        if ":= by" in line:
            tail = line.split(":= by", 1)[1].strip()
            return "\n".join(([tail] if tail else []) + lines[index + 1:]).strip()
    return "\n".join(lines).strip()


DIAGNOSTIC_START = re.compile(r"^\s*(?:Tactic `.*?` failed:|[Ee]rror:|unknown tactic\b|unsolved goals\b|[^\n]+:\d+:\d+:\s*error:)")


def formal_region(text):
    # The first formal boundary is authoritative; pre-boundary thinking is not
    # candidate material even if it happens to contain tactic-looking prose.
    if "</think>" in text:
        return text.split("</think>", 1)[1], "CLOSED_THINK_FORMAL_REGION"
    if "[End thinking]" in text:
        return text.split("[End thinking]", 1)[1], "END_THINKING_FORMAL_REGION"
    if "<think>" in text:
        # A truncated reasoning response is not a formal response.  In
        # particular, never fall back to scanning it (or the echoed prompt).
        return "", "REASONING_TRUNCATED"
    return text, "NO_REASONING_MARKERS"


def _formal_region(text):
    """Compatibility helper for callers that only need the text region."""
    return formal_region(text)[0]


def _is_diagnostic(candidate):
    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    if not lines:
        return True
    diagnostic_lines = sum(bool(DIAGNOSTIC_START.match(line)) for line in lines)
    return bool(DIAGNOSTIC_START.match(lines[0])) or diagnostic_lines * 2 >= len(lines)


def extract_candidates(text, *, max_candidates=MAX_CANDIDATES, max_candidate_chars=MAX_CANDIDATE_CHARS,
                       max_lines=MAX_CANDIDATE_LINES):
    """Deterministically extract bounded, plausible Lean chunks from native output."""
    formal, _ = formal_region(bounded(text, MAX_RAW_OUTPUT))
    candidates = []
    for match in re.finditer(r"```(?:lean4?|tactics?)?\s*\n(.*?)(?:```|\Z)", formal, flags=re.S | re.I):
        body = clean_block(match.group(1)).split("[ Prompt:", 1)[0].split("Exiting...", 1)[0].strip()
        if body:
            candidates.append(body)
    # Kimina sometimes prints a complete declaration without a Markdown fence.
    # Surface only its existing proof body; do not synthesize any tactics.
    formal_lines = formal.splitlines()
    for index, line in enumerate(formal_lines):
        if ":= by" in line:
            body = clean_block("\n".join(formal_lines[index:]))
            if body:
                candidates.append(body)
        elif line.strip() == "by":
            body = "\n".join(formal_lines[index + 1:]).strip()
            if body:
                candidates.append(body)
    tactic_lines = [line.strip().strip("`") for line in formal_lines if TACTIC_START.match(line.strip().strip("`"))]
    candidates.extend(tactic_lines)
    # Include contiguous sequences, retaining indentation only where it was
    # present in a code block; standalone lines are intentional flat tactics.
    for start in range(len(tactic_lines)):
        for length in range(2, min(6, len(tactic_lines) - start) + 1):
            candidates.append("\n".join(tactic_lines[start:start + length]))
    unique, seen = [], set()
    for candidate in candidates:
        candidate = candidate.strip()
        if (not candidate or candidate in seen or _is_diagnostic(candidate) or len(candidate) > max_candidate_chars
                or len(candidate.splitlines()) > max_lines):
            continue
        seen.add(candidate)
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class KiminaGeneration:
    status: str
    output: str = ""
    diagnostic: str = ""
    returncode: int | None = None
    transcript: str = ""
    response: str = ""
    formal_region_classification: str = ""
    transcript_sha256: str = ""
    response_sha256: str = ""


def isolate_model_response(transcript, prompt):
    """Remove the exact echoed input prompt before any candidate extraction.

    llama.cpp versions differ in how much banner/prompt text they print.  The
    caller knows the exact prompt, so an echoed copy is an unambiguous boundary.
    When no echo is present, stdout is retained as the bounded response; this is
    still safe because the prompt itself is never supplied to the extractor.
    """
    transcript = str(transcript or "")
    marker = str(prompt or "")
    index = transcript.rfind(marker) if marker else -1
    return transcript[index + len(marker):] if index >= 0 else transcript


class KiminaMicroProofSolver:
    """Direct llama.cpp adapter; no grammar/schema is ever supplied."""
    def __init__(self, model=None, llama=None, attempts_per_goal=1, reasoning_budget=96,
                 max_output_tokens=640, timeout=180, threads=2, runner=subprocess.run):
        self.model = Path(model or os.environ.get("KIMINA_MODEL", DEFAULT_MODEL))
        self.llama = Path(llama or os.environ.get("LLAMA_CPP_CLI_BIN", DEFAULT_LLAMA))
        self.attempts_per_goal = max(1, min(int(attempts_per_goal), 4))
        self.reasoning_budget = max(1, min(int(reasoning_budget), 512))
        self.max_output_tokens = max(32, min(int(max_output_tokens), 2048))
        self.timeout, self.threads, self.runner = max(1, int(timeout)), max(1, int(threads)), runner
        self.invocations = 0
        self.failures = []
        self.model_sha256 = ""

    def metadata(self):
        if self.model.is_file() and not self.model_sha256:
            self.model_sha256 = sha256_file(self.model)
        return {"model": self.model.name, "model_sha256": self.model_sha256,
                "expected_model_sha256": EXPECTED_Q4_K_M_SHA256,
                "model_path": str(self.model), "llama": str(self.llama), "invocations": self.invocations,
                "generation_failures": len(self.failures)}

    def generate(self, prompt, *, max_output_tokens=None):
        if not self.llama.is_file():
            return KiminaGeneration("MISSING_BINARY", diagnostic=str(self.llama))
        if not self.model.is_file():
            return KiminaGeneration("MISSING_MODEL", diagnostic=str(self.model))
        self.invocations += 1
        output_budget = self.max_output_tokens if max_output_tokens is None else max(32, min(int(max_output_tokens), 2048))
        cmd = [str(self.llama), "-m", str(self.model), "-c", "2048", "-t", str(self.threads), "-n", str(output_budget),
               "-st", "--simple-io", "--skip-chat-parsing", "--no-display-prompt", "-rea", "on", "--reasoning-budget", str(self.reasoning_budget),
               "--temp", "0.4", "--top-p", "0.95", "-p", prompt]
        try:
            cp = self.runner(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            result = KiminaGeneration("TIMEOUT", diagnostic=bounded(getattr(exc, "stdout", "") or str(exc), 800))
        except OSError as exc:
            result = KiminaGeneration("PROCESS_FAILURE", diagnostic=str(exc))
        else:
            transcript = bounded(cp.stdout, MAX_RAW_OUTPUT)
            output = bounded(isolate_model_response(transcript, prompt), MAX_RAW_OUTPUT)
            _, classification = formal_region(output)
            transcript_sha256 = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            response_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
            if cp.returncode:
                result = KiminaGeneration("PROCESS_FAILURE", output, bounded(output, 800), cp.returncode,
                                          transcript, output, classification, transcript_sha256, response_sha256)
            elif not output.strip():
                result = KiminaGeneration("OUTPUT_EMPTY", "", "empty stdout", cp.returncode,
                                          transcript, output, classification, transcript_sha256, response_sha256)
            else:
                return KiminaGeneration("OK", output, returncode=cp.returncode, transcript=transcript,
                                        response=output, formal_region_classification=classification,
                                        transcript_sha256=transcript_sha256, response_sha256=response_sha256)
        self.failures.append(result.status)
        return result
