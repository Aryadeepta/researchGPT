"""Reusable, public Lean proof-task input and source rendering.

The task deliberately contains a declaration, never a mutable theorem body.
Search may supply only the proof following ``:=``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProofTask:
    task_id: str
    theorem: str
    declaration: str
    expected_type: str
    imports: tuple[str, ...] = ("Std",)
    preamble: str = "set_option autoImplicit false"
    namespace: str = "ProofTask"
    initial_prefix: tuple[str, ...] = ()
    provenance: str = ""
    output_path: str | None = None
    level: str = "generic"
    description: str = "Generic Lean proof task."

    @property
    def source_hash(self) -> str:
        value = {"imports": self.imports, "preamble": self.preamble,
                 "theorem": self.theorem, "declaration": self.declaration,
                 "namespace": self.namespace, "provenance": self.provenance}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def render_source(self, proof_body: str) -> str:
        imports = "".join(f"import {item}\n" for item in self.imports)
        preamble = (self.preamble.rstrip() + "\n") if self.preamble.strip() else ""
        namespace_open = f"namespace {self.namespace}\n\n" if self.namespace else ""
        namespace_close = f"\nend {self.namespace}\n" if self.namespace else ""
        return f"{imports}{preamble}{namespace_open}theorem {self.theorem} {self.declaration} := {proof_body}\n{namespace_close}"

    @property
    def qualified_theorem(self) -> str:
        return f"{self.namespace}.{self.theorem}" if self.namespace else self.theorem

    @classmethod
    def from_json(cls, path: str | Path) -> "ProofTask":
        source = Path(path)
        raw = json.loads(source.read_text())
        required = ("task_id", "theorem", "declaration", "expected_type")
        missing = [key for key in required if not isinstance(raw.get(key), str) or not raw[key].strip()]
        if missing:
            raise ValueError("PROOF_TASK_MISSING:" + ",".join(missing))
        imports = raw.get("imports", ["Std"])
        if not isinstance(imports, list) or not all(isinstance(x, str) and x.strip() for x in imports):
            raise ValueError("PROOF_TASK_IMPORTS")
        initial = raw.get("initial_prefix", [])
        if not isinstance(initial, list) or not all(isinstance(x, str) for x in initial):
            raise ValueError("PROOF_TASK_INITIAL_PREFIX")
        return cls(**{key: raw[key] for key in required}, imports=tuple(imports),
                   preamble=str(raw.get("preamble", "set_option autoImplicit false")),
                   namespace=str(raw.get("namespace", "ProofTask")), initial_prefix=tuple(initial),
                   provenance=str(raw.get("provenance", str(source))),
                   output_path=raw.get("output_path"), level="generic",
                   description=str(raw.get("description", "Generic Lean proof task.")))
