"""Persistent, explanatory sessions for finite proof obligations.

The session is deliberately a layer over the existing finite-model and Lean
certificate machinery.  Scout output is never evidence: every displayed lemma
has a separate Lean artifact and becomes usable only after that artifact exits
successfully.  The finite closer remains available, but only after the small
deduction prefix has been recorded.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tools.proofbench.proof_engineering import (
    BooleanConstraint, ConstraintProvenance, FiniteBooleanModel, ProofDag, ProofDagNode, ProofTerminal,
    available_capabilities, create_handoff_bundle, dimacs_export, propagate_constraints, residual_model,
    readable_deduction_trace, sha256_file,
)
from tools.proofbench.adapters import AdditiveBasisAdapter, adapter_for_specification
from tools.proofbench.proof_gym import resolve_lean
from tools.proofbench.proof_semantic_adapter import ProofSemanticAdapter, digest as adapter_digest, file_hash as adapter_file_hash


class SessionStatus(str, Enum):
    WORKING = "WORKING"
    VERIFIED = "VERIFIED"
    ACTIONABLE_HANDOFF = "ACTIONABLE_HANDOFF"
    PROOF_RECOVERY_EXHAUSTED = "PROOF_RECOVERY_EXHAUSTED"


@dataclass
class SessionLemma:
    stable_id: str
    statement: str
    formal_statement: str
    origin: str
    dependencies: list[str] = field(default_factory=list)
    producer: str = "constraint-scout"
    verifier: str = "Lean"
    verification_status: str = "UNVERIFIED"
    artifact_path: str = ""
    artifact_hash: str = ""
    impact: dict[str, int] = field(default_factory=dict)
    imported_theorem: str = ""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ProofSession:
    """Durable checkpoint with a compact user-facing notebook."""
    version = "proof-session/v4-generic-semantic-adapter"
    session_schema_version = "proof-session-checkpoint/v1"

    def __init__(self, root: str | Path, specification: dict[str, Any], *, create: bool = True, adapter: ProofSemanticAdapter | None = None, domain_adapter: Any | None = None):
        self.root = Path(root)
        self.specification = specification
        self.adapter = adapter
        self.domain_adapter = domain_adapter
        self.adapter_metadata = adapter.metadata() if adapter else self._builtin_adapter_metadata(specification)
        self.session_id = _digest({"schema": self.session_schema_version, "adapter": self.adapter_metadata["adapter_hash"], "specification": specification})[:20]
        self.path = self.root / self.session_id
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        self.status = SessionStatus.WORKING
        self.lemmas: list[SessionLemma] = []
        self.human_interventions: list[dict[str, Any]] = []
        self.attempted_strategies: list[str] = []
        self.verifier_results: list[dict[str, Any]] = []
        self.dag = ProofDag()
        self.model = None
        self.trace: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {"verified_lemmas": 0, "conjectured_lemmas": 0, "rejected_lemmas": 0,
            "lean_checks": 0, "tool_calls": 0, "candidate_space_reduction": 0, "proof_dag_nodes": 0,
            "proof_dag_depth": 0, "residual_boolean_variables": 0, "residual_clauses": 0,
            "certificate_type": "", "human_interventions": 0, "specialist_interventions": 0, "final_status": self.status.value}
        if (self.path / "session.json").is_file(): self._load()

    @classmethod
    def _builtin_adapter_metadata(cls, specification: dict[str, Any]) -> dict[str, Any]:
        kind=str(specification.get("problem_type", "finite"))
        semantics={"builtin": kind, "implementation": cls.version}
        return {"adapter_id": f"builtin.{kind}", "adapter_version": "1",
                "adapter_hash": _digest(semantics), "specification_hash": _digest(specification),
                "obligation_hash": _digest({"goal": specification.get("goal"), "specification": specification}),
                "semantics_artifact": "Semantics.lean", "semantics_hash": "",
                "executable_checker": {"id": "proof-session-builtin"}, "finite_model": None,
                "formal_artifact_generator": None}

    @classmethod
    def from_task(cls, root: str | Path, task_path: str | Path) -> "ProofSession":
        raw = json.loads(Path(task_path).read_text())
        spec = raw.get("finite_additive_basis")
        if isinstance(spec, dict):
            required = ("n", "forbidden", "max_selected")
            if any(k not in spec for k in required): raise ValueError("SESSION_SPEC_MISSING")
            specification = {"task_id": raw.get("task_id", "finite-proof"), "goal": raw.get("theorem", "finite lower bound"), "problem_type": "finite", **spec}
            return cls(root, specification, domain_adapter=AdditiveBasisAdapter(specification))
        spec = raw.get("boolean_implication_system")
        if isinstance(spec, dict):
            if not isinstance(spec.get("variables"), list) or not isinstance(spec.get("clauses"), list): raise ValueError("BOOLEAN_SESSION_SPEC_MISSING")
            return cls(root, {"task_id": raw.get("task_id", "boolean-proof"), "goal": raw.get("theorem", "Boolean consequence"), "problem_type": "boolean", **spec})
        raise ValueError("SESSION_REQUIRES_SUPPORTED_FINITE_SPEC")

    @classmethod
    def from_adapter(cls, root: str | Path, manifest_path: str | Path) -> "ProofSession":
        adapter = ProofSemanticAdapter.load(manifest_path)
        spec = {"task_id": adapter.adapter_id, "goal": adapter.manifest["obligation"],
                "problem_type": "semantic_adapter", "adapter_specification": adapter.manifest["specification"]}
        return cls(root, spec, adapter=adapter)

    @classmethod
    def open(cls, path: str | Path) -> "ProofSession":
        p = Path(path); data = json.loads((p / "session.json").read_text())
        adapter_path=data.get("adapter_manifest")
        adapter=ProofSemanticAdapter.load(adapter_path) if adapter_path else None
        return cls(p.parent, data["specification"], create=False, adapter=adapter,
                   domain_adapter=None if adapter else adapter_for_specification(data["specification"]))

    def _save(self) -> None:
        verified_hashes={x.stable_id: x.artifact_hash for x in self.lemmas if x.verification_status == "VERIFIED"}
        self.adapter_metadata["semantics_hash"] = self.metrics.get("semantics_artifact_hash", self.adapter_metadata.get("semantics_hash", ""))
        payload = {"session_schema_version": self.session_schema_version, "version": self.version, "implementation_identity": self.version,
            "session_id": self.session_id, "specification": self.specification, "adapter_manifest": str(self.adapter.root / "adapter_manifest.json") if self.adapter else None,
            "adapter_id": self.adapter_metadata["adapter_id"], "adapter_version": self.adapter_metadata["adapter_version"], "adapter_hash": self.adapter_metadata["adapter_hash"],
            "specification_hash": self.adapter_metadata["specification_hash"], "obligation_hash": self.adapter_metadata["obligation_hash"],
            "semantics_artifact": self.adapter_metadata["semantics_artifact"], "semantics_hash": self.adapter_metadata["semantics_hash"],
            "verified_artifact_hashes": verified_hashes, "handoff_hash": self.metrics.get("handoff_hash", ""), "handoff_status": self.metrics.get("handoff_status", ""),
            "status": self.status.value, "lemmas": [asdict(x) for x in self.lemmas], "human_interventions": self.human_interventions,
            "status": self.status.value, "lemmas": [asdict(x) for x in self.lemmas], "human_interventions": self.human_interventions,
            "attempted_strategies": self.attempted_strategies, "verifier_results": self.verifier_results,
            "proof_dag": {k: asdict(v) for k, v in self.dag.nodes.items()}, "metrics": self.metrics}
        (self.path / "session.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (self.path / "proof_dag.json").write_text(json.dumps(payload["proof_dag"], indent=2, sort_keys=True) + "\n")
        (self.path / "metrics.json").write_text(json.dumps(self.metrics, indent=2, sort_keys=True) + "\n")
        self._notebook()

    def _load(self) -> None:
        data = json.loads((self.path / "session.json").read_text()); self.status = SessionStatus(data["status"])
        if data.get("session_schema_version") != self.session_schema_version:
            raise ValueError("RESUME_INTEGRITY_ERROR: unsupported session schema")
        for key in ("adapter_id", "adapter_version", "adapter_hash", "specification_hash", "obligation_hash", "semantics_artifact", "semantics_hash", "verified_artifact_hashes", "handoff_hash", "handoff_status", "implementation_identity"):
            if key not in data: raise ValueError("RESUME_INTEGRITY_ERROR: missing generic checkpoint field")
        identity_keys=("adapter_id", "adapter_version", "adapter_hash", "specification_hash", "obligation_hash")
        if any(data[k] != self.adapter_metadata.get(k) for k in identity_keys):
            raise ValueError("RESUME_INTEGRITY_ERROR: adapter or specification identity mismatch")
        self.adapter_metadata.update({k:data[k] for k in ("adapter_id", "adapter_version", "adapter_hash", "specification_hash", "obligation_hash", "semantics_artifact", "semantics_hash")})
        self.lemmas = [SessionLemma(**x) for x in data.get("lemmas", [])]; self.human_interventions = data.get("human_interventions", [])
        self.attempted_strategies = data.get("attempted_strategies", []); self.verifier_results = data.get("verifier_results", [])
        self.metrics.update(data.get("metrics", {}))
        for node in data.get("proof_dag", {}).values(): self.dag.add(ProofDagNode(**node))

    def _write_semantics(self) -> Path:
        """Freeze the *inclusive* mathematical meaning shared by all artifacts."""
        if self.domain_adapter is None: raise ValueError("SESSION_REQUIRES_DOMAIN_ADAPTER")
        target=self.path/"Semantics.lean"
        target.write_text(self.domain_adapter.render_semantics())
        # Lean imports compiled modules, so materialize the frozen artifact
        # before any lemma imports it.
        try: subprocess.run([resolve_lean(), "-o", "Semantics.olean", target.name], cwd=self.path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90, check=True)
        except Exception: pass
        self.metrics["semantics_artifact_hash"]=sha256_file(target); self.metrics["session_specification_hash"]=_digest(self.specification)
        self.metrics["implementation_version"]=self.version
        return target

    def _verify_resume_integrity(self) -> None:
        """Check the frozen checkpoint before any operation can rewrite it."""
        expected_spec = self.adapter_metadata.get("specification_hash")
        actual_spec = self.adapter.manifest["specification_hash"] if self.adapter else _digest(self.specification)
        if self.metrics.get("implementation_version") != self.version or expected_spec != actual_spec:
            raise ValueError("RESUME_INTEGRITY_ERROR: session specification or implementation version mismatch")
        if self.adapter and self.adapter.adapter_hash != self.adapter_metadata.get("adapter_hash"):
            raise ValueError("RESUME_INTEGRITY_ERROR: adapter hash mismatch")
        semantic = self.path / Path(str(self.adapter_metadata.get("semantics_artifact", "Semantics.lean"))).name
        if not semantic.is_file() or self.adapter_metadata.get("semantics_hash") != sha256_file(semantic):
            raise ValueError("RESUME_INTEGRITY_ERROR: Semantics.lean hash mismatch")
        for lemma in self.lemmas:
            if lemma.verification_status != "VERIFIED":
                continue
            artifact = Path(lemma.artifact_path)
            if not artifact.is_file() or lemma.artifact_hash != sha256_file(artifact):
                raise ValueError(f"RESUME_INTEGRITY_ERROR: verified lemma artifact mismatch: {lemma.stable_id}")
        handoff = self.path / "handoff" / "handoff_bundle.json"
        expected_handoff = self.metrics.get("handoff_hash")
        if expected_handoff and (not handoff.is_file() or expected_handoff != sha256_file(handoff)):
            raise ValueError("RESUME_INTEGRITY_ERROR: handoff checkpoint hash mismatch")

    @staticmethod
    def _lemma_tail(lemma: SessionLemma) -> str:
        """Return the actual checked declaration, bound to the frozen semantics."""
        text = Path(lemma.artifact_path).read_text()
        marker = f"\ntheorem {lemma.imported_theorem or lemma.stable_id}"
        if marker not in text:
            raise ValueError(f"HUMAN_ARTIFACT_INVALID: declaration {lemma.stable_id} missing")
        return marker + text.split(marker, 1)[1]

    def _notebook(self) -> None:
        lines = ["# Goal", "", str(self.specification.get("goal", "finite proof obligation")), "", "# Assumptions", "",
                 (self.domain_adapter.describe() if self.domain_adapter else "Finite Boolean clauses interpreted with standard propositional semantics."),
                 "", "# Verified deductions", ""]
        for lemma in self.lemmas:
            lines += [f"## {lemma.stable_id}: {lemma.statement}", "", f"Status: **{lemma.verification_status}**", "",
                      f"Why it matters: residual candidates {lemma.impact.get('before', '?')} -> {lemma.impact.get('after', '?')} (computational metadata).", "",
                      f"Lean verifier: {lemma.verification_status}; artifact: `{lemma.artifact_path}`", ""]
        lines += ["# Computational observations", "", "COMPUTATIONALLY_OBSERVED: representations were enumerated and simplified; these observations became usable only after their individual Lean checks.",
                  "", "# Residual obligation", "", f"UNRESOLVED: {self.metrics.get('residual_boolean_variables', 0)} Boolean variables and {self.metrics.get('residual_clauses', 0)} clauses.",
                  "", "# Final verifier status", "", self.status.value]
        if self.status != SessionStatus.VERIFIED:
            lines += ["", "# What remains", "", "UNRESOLVED: show the residual Boolean model is UNSAT under its stated constraints.", "", "# Suggested next actions", "", "Provide a Lean lemma, checked SAT certificate, a case split, or an advisory reasoning note (which remains non-evidence)."]
        (self.path / "proof_notebook.md").write_text("\n".join(lines) + "\n")

    def _lean_check(self, lemma: SessionLemma) -> bool:
        # A separate source file provides an independently replayable checker boundary.
        target = self.path / "lemmas" / f"{lemma.stable_id}.lean"; target.parent.mkdir(exist_ok=True)
        semantic=self.path/"Semantics.lean"
        # Embed the frozen source rather than relying on Lean's project import
        # path.  The leading hash binds this lemma to its shared artifact.
        target.write_text(f"-- shared-semantics-sha256: {sha256_file(semantic)}\n" + semantic.read_text() + f"\ntheorem {lemma.stable_id} : {lemma.formal_statement} := by native_decide\n")
        try:
            lean = resolve_lean(); cp = subprocess.run([lean, target.name], cwd=target.parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            ok, diagnostic = cp.returncode == 0, cp.stdout[-2000:]
        except Exception as exc:
            ok, diagnostic = False, str(exc)
        lemma.artifact_path = str(target); lemma.artifact_hash = sha256_file(target); lemma.verification_status = "VERIFIED" if ok else "REJECTED"
        self.metrics["lean_checks"] += 1; self.verifier_results.append({"lemma": lemma.stable_id, "verifier": "Lean", "status": lemma.verification_status, "diagnostic": diagnostic})
        return ok

    def _add_lemma(self, statement: str, formal: str, origin: str, *, dependencies: list[str] | None = None, human: bool = False) -> SessionLemma:
        lid = f"L{len(self.lemmas) + 1}"; before = int(self.metrics.get("candidate_space_current", 2 ** max(0, self.metrics.get("residual_boolean_variables", 0))))
        after = before  # measured after residual construction; never fabricated.
        lemma = SessionLemma(lid, statement, formal, origin, dependencies or [], producer="human-import" if human else "constraint-scout", impact={"before": before, "after": after})
        self.lemmas.append(lemma); self.dag.add(ProofDagNode(lid, statement, _digest([statement, formal]), lemma.dependencies, producer=lemma.producer))
        if self._lean_check(lemma):
            self.dag.verify(lid, verifier="Lean", artifact_hash=lemma.artifact_hash); self.metrics["verified_lemmas"] += 1; self.metrics["candidate_space_reduction"] += before - after; self.metrics["candidate_space_current"] = after
        else: self.metrics["rejected_lemmas"] += 1
        self.metrics["proof_dag_nodes"] = len(self.dag.nodes); self.metrics["proof_dag_depth"] = max((len(x.dependencies) for x in self.dag.nodes.values()), default=0)
        return lemma

    def start(self, *, explain: bool = False, allow_closer: bool = True, progress: bool = False) -> "ProofSession":
        if self.adapter:
            return self._start_adapter(progress=progress)
        if self.specification.get("problem_type") == "boolean":
            return self._start_boolean(allow_closer=allow_closer, progress=progress)
        if self.domain_adapter is None: raise ValueError("SESSION_REQUIRES_DOMAIN_ADAPTER")
        self.status = SessionStatus.WORKING; self.attempted_strategies = ["direct-symbolic-facts", "constraint-scout", "small-lemma-formalization"]
        self._write_semantics()
        self.model = self.domain_adapter.build_constraint_model(_digest(self.specification)); facts, residual, trace = propagate_constraints(self.model); self.trace = readable_deduction_trace(self.model, facts, trace)
        scout = self.path / "computational_observations.json"; scout.write_text(json.dumps(self.trace, indent=2, sort_keys=True) + "\n")
        reduced=residual_model(self.model, facts); residual_info = dimacs_export(reduced, self.path / "residual"); self.metrics["residual_boolean_variables"] = residual_info["variable_count"]; self.metrics["residual_clauses"] = residual_info["clause_count"]; self.metrics["candidate_space_current"] = 2 ** len(reduced.variables); self.metrics["original_model_hash"]=self.model.digest(); self.metrics["residual_model_hash"]=reduced.digest()
        candidates: list[tuple[str, str]] = []
        for item in self.trace:
            pair = self.domain_adapter.render_semantic_lemma(item, facts)
            if pair and pair[0] not in [x[0] for x in candidates]: candidates.append(pair)
        # A forced handoff deliberately records only the initial propagation
        # prefix.  This leaves later coverage alternatives as genuinely new
        # human input, while autonomous sessions may retain their explanatory
        # prefix.
        prefix_count = 6 if not allow_closer else max(5, min(len(candidates), 10))
        for statement, formal in candidates[:prefix_count]:
            lemma = self._add_lemma(statement, formal, "generic finite-model simplification")
            if progress: print(f'[proof] lemma={lemma.stable_id} statement="{lemma.statement}"'); print(f"[proof] lemma={lemma.stable_id} lean={'PASS' if lemma.verification_status == 'VERIFIED' else 'FAIL'}")
        (self.path / "deduction_trace.json").write_text(json.dumps(self.trace, indent=2, sort_keys=True) + "\n")
        if not allow_closer:
            self._handoff("final exhaustive/certificate closer disabled by session budget")
        elif self.metrics["verified_lemmas"] < 5:
            self.status = SessionStatus.PROOF_RECOVERY_EXHAUSTED
        else:
            self._close(progress)
        self.metrics["final_status"] = self.status.value; self._save(); return self

    def _start_adapter(self, *, progress: bool = False) -> "ProofSession":
        """Persist the universal checkpoint before reporting adapter recovery failure."""
        assert self.adapter is not None
        meta=self.adapter.metadata(); source=self.adapter.artifact_path(meta["semantics_artifact"])
        target=self.path / Path(source.name); shutil.copyfile(source, target)
        self.adapter_metadata["semantics_artifact"]=target.name
        self.adapter_metadata["semantics_hash"]=sha256_file(target)
        self.metrics["semantics_artifact_hash"]=sha256_file(target)
        self.metrics["session_specification_hash"]=meta["specification_hash"]
        self.metrics["implementation_version"]=self.version
        model=None
        finite=meta.get("finite_model")
        if isinstance(finite, dict) and finite.get("path"):
            raw=json.loads(self.adapter.artifact_path(finite).read_text())
            system=raw.get("boolean_implication_system", raw)
            if isinstance(system.get("variables"), list) and isinstance(system.get("clauses"), list):
                variables={f"x_{i}":str(x) for i,x in enumerate(system["variables"])}
                constraints=[BooleanConstraint("clause", list(c), None, f"adapter-clause-{i}", ConstraintProvenance("semantic-adapter", "finite-constraint", [str(i)])) for i,c in enumerate(system["clauses"])]
                model=FiniteBooleanModel(meta["obligation_hash"], variables, constraints)
        self.model=model; self.attempted_strategies=["semantic-adapter-checkpoint", "adapter-executable-checker"]
        self._handoff("adapter checkpoint created; autonomous proof recovery requires a checked formal contribution")
        self.metrics["final_status"]=self.status.value; self._save()
        # This is deliberately after _save: an actionable handoff is already a
        # fully valid checkpoint before any recovery work is attempted.
        self._verify_resume_integrity()
        if progress: print("[proof] semantic-adapter handoff=resume-compatible")
        return self

    def _start_boolean(self, *, allow_closer: bool, progress: bool) -> "ProofSession":
        names = [str(x) for x in self.specification["variables"]]
        variables = {f"x_{i}": name for i, name in enumerate(names)}
        constraints = [BooleanConstraint("clause", [int(x) for x in clause], None, f"clause-{i}", ConstraintProvenance("boolean-input", "clause", [str(i)])) for i, clause in enumerate(self.specification["clauses"])]
        self.model = FiniteBooleanModel(_digest(self.specification), variables, constraints)
        facts, _, raw_trace = propagate_constraints(self.model)
        self.trace = readable_deduction_trace(self.model, facts, raw_trace)
        residual_info = dimacs_export(self.model, self.path / "residual"); self.metrics["residual_boolean_variables"] = residual_info["variable_count"]; self.metrics["residual_clauses"] = residual_info["clause_count"]; self.metrics["candidate_space_current"] = 2 ** len(variables)
        assignments=[list(bits) for bits in __import__("itertools").product([False,True], repeat=len(names))]
        def lit_expr(lit: int) -> str:
            base=f"pick A {abs(lit)-1}"; return base if lit > 0 else f"!({base})"
        clauses=" && ".join("("+" || ".join(lit_expr(x) for x in c.literals)+")" for c in constraints) or "true"
        shared="import Init.Tactics\nset_option autoImplicit false\ndef assignments : List (List Bool) := ["+", ".join("["+", ".join(str(x).lower() for x in a)+"]" for a in assignments)+"]\ndef pick : List Bool → Nat → Bool\n| [], _ => false\n| x :: _, 0 => x\n| _ :: xs, i + 1 => pick xs i\ndef admissible (A : List Bool) : Bool := "+clauses+"\n"
        for var, value in sorted(facts.items()):
            statement = f"{variables[var]} must be true" if value else f"{variables[var]} must be false"
            idx=list(variables).index(var); predicate=f"pick A {idx}" if value else f"!(pick A {idx})"
            formal=f"(assignments.all (fun A => !(admissible A) || {predicate})) = true"
            lid=f"L{len(self.lemmas)+1}"; lemma=SessionLemma(lid,statement,formal,"generic Boolean unit propagation")
            self.lemmas.append(lemma); target=self.path/"lemmas"/f"{lid}.lean"; target.parent.mkdir(exist_ok=True); target.write_text(shared+f"theorem {lid} : {formal} := by native_decide\n")
            try: ok=subprocess.run([resolve_lean(),target.name],cwd=target.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90).returncode==0
            except Exception: ok=False
            lemma.artifact_path=str(target); lemma.artifact_hash=sha256_file(target); lemma.verification_status="VERIFIED" if ok else "REJECTED"; self.metrics["verified_lemmas"]+=int(ok); self.metrics["lean_checks"]+=1; self.dag.add(ProofDagNode(lid,statement,_digest([statement,formal]),producer="constraint-scout"));
            if ok:self.dag.verify(lid,verifier="Lean",artifact_hash=lemma.artifact_hash)
            if progress: print(f'[proof] lemma={lemma.stable_id} statement="{lemma.statement}"'); print(f"[proof] lemma={lemma.stable_id} lean={'PASS' if lemma.verification_status == 'VERIFIED' else 'FAIL'}")
        if allow_closer:
            goal=str(self.specification.get("goal", "")); idx=names.index(goal) if goal in names else -1
            if idx < 0: self.status=SessionStatus.PROOF_RECOVERY_EXHAUSTED; self._save(); return self
            target = self.path / "final_certificate.lean"; target.write_text(shared+f"theorem finite_boolean_closure : (assignments.all (fun A => !(admissible A) || pick A {idx})) = true := by native_decide\n")
            lean = resolve_lean(); cp = subprocess.run([lean, target.name], cwd=self.path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            self.metrics["lean_checks"] += 1; self.metrics["certificate_type"] = "finite-boolean-certificate"; self.verifier_results.append({"artifact":str(target),"verifier":"Lean","status":"PASS" if cp.returncode == 0 else "FAIL","diagnostic":cp.stdout[-1000:]}); self.status = SessionStatus.VERIFIED if cp.returncode == 0 else SessionStatus.PROOF_RECOVERY_EXHAUSTED
        else: self._handoff("final Boolean closer disabled by session budget")
        self.metrics["final_status"] = self.status.value; self._save(); return self

    def _close(self, progress: bool) -> None:
        self.attempted_strategies.append("shared-semantics-finite-certificate")
        # This is the requested goal in exactly the Semantics.lean encoding,
        # not a proxy theorem and not a certificate for a different universe.
        target = self.path / "final_certificate.lean"
        human=[x for x in self.lemmas if x.producer=="human-import" and x.verification_status=="VERIFIED"]
        # A finite closer which does not mention explanatory deductions must
        # not pretend that they are formal parents in the proof DAG.
        dependency: list[str] = []
        semantic=self.path/"Semantics.lean"; body=f"-- shared-semantics-sha256: {sha256_file(semantic)}\n"+semantic.read_text()+"\n"
        if human:
            h=human[-1]
            # Preserve and use the exact theorem which Lean checked for the
            # intervention; never replace it with a new native_decide proof.
            body += self._lemma_tail(h) + "\n"
            prefix = "(candidates.all (fun A => !(admissible A) || "
            suffix = ")) = true"
            # Structured semantic hints have the canonical universal shape
            # emitted above.  A differently-shaped theorem cannot be smuggled
            # into this causal route.
            if not (h.formal_statement.startswith(prefix) and h.formal_statement.endswith(suffix)):
                raise ValueError("HUMAN_LEMMA_TYPE_MISMATCH: expected admissible-to-property theorem")
            predicate = h.formal_statement[len(prefix):-len(suffix)]
            body += f"def humanProperty (A : List Nat) : Bool := {predicate}\n"
            body += "def noAdmissibleWithHuman : Bool := candidates.all (fun A => !(humanProperty A) || !(admissible A))\n"
            body += '''theorem residual_compose_list (xs : List (List Nat)) (P Q : List Nat → Bool) :
    xs.all (fun A => !Q A || P A) = true →
    xs.all (fun A => !P A || !Q A) = true →
    xs.all (fun A => !Q A) = true := by
  intro hp hr
  induction xs with
  | nil => simp
  | cons A xs ih =>
    simp only [List.all_cons, Bool.and_eq_true] at hp hr ⊢
    constructor
    · cases hP : P A <;> cases hQ : Q A <;> simp_all
    · exact ih hp.2 hr.2
theorem human_residual : noAdmissibleWithHuman = true := by native_decide
theorem final_closure : noAdmissible = true := by
  exact residual_compose_list candidates humanProperty admissible ''' + (h.imported_theorem or h.stable_id) + ''' human_residual
'''
            dependency=[h.stable_id]
        else:
            body += "theorem final_closure : noAdmissible = true := by native_decide\n"
        target.write_text(body)
        try:
            lean = resolve_lean(); cp = subprocess.run([lean, target.name], cwd=self.path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            ok, diagnostic = cp.returncode == 0, cp.stdout[-4000:]
        except Exception as exc: ok, diagnostic = False, str(exc)
        self.metrics["lean_checks"] += 1; self.metrics["certificate_type"] = "causal-restricted-residual" if human else "independent-finite-certificate-closer"; self.verifier_results.append({"artifact": str(target), "verifier": "Lean", "status": "PASS" if ok else "FAIL", "diagnostic": diagnostic})
        if ok:
            node=ProofDagNode("FINAL", "no admissible basis of requested cardinality", sha256_file(target), dependency, producer="shared-semantics-closer")
            self.dag.add(node); self.dag.verify("FINAL", verifier="Lean", artifact_hash=sha256_file(target)); self.metrics["proof_dag_nodes"]=len(self.dag.nodes)
        self.status = SessionStatus.VERIFIED if ok else SessionStatus.PROOF_RECOVERY_EXHAUSTED
        if progress: print(f"[proof] closer=finite-certificate\n[proof] final lean={'PASS' if ok else 'FAIL'}")

    def _handoff(self, diagnostic: str) -> None:
        bundle = create_handoff_bundle(self.path / "handoff", obligation_id=self.session_id, obligation_hash=_digest(self.specification), goal=str(self.specification.get("goal", "finite proof obligation")), classification="FINITE_COMBINATORIAL", model=self.model, verified_prefix=[x.stable_id for x in self.lemmas if x.verification_status == "VERIFIED"], dag_state={k: asdict(v) for k, v in self.dag.nodes.items()}, attempts=self.attempted_strategies, diagnostics=diagnostic, capabilities=available_capabilities(self.path, resolve_lean()))
        self.metrics["handoff_hash"]=sha256_file(self.path/"handoff"/"handoff_bundle.json")
        self.status = SessionStatus.ACTIONABLE_HANDOFF; self.metrics["handoff_status"] = self.status.value; self.metrics["certificate_type"] = "none"; self.verifier_results.append({"artifact": str(self.path / "handoff" / "handoff_bundle.json"), "verifier": "handoff-generator", "status": bundle["terminal_state"]})

    def import_human(self, path: str | Path) -> SessionLemma | None:
        raw = json.loads(Path(path).read_text()); kind = raw.get("type", "mathematical_hint"); text = str(raw.get("statement", ""))
        intervention = {"type": kind, "statement": text, "source": str(path), "verification_status": "ADVISORY"}
        self.human_interventions.append(intervention); self.metrics["human_interventions"] += 1
        if kind not in {"lean_lemma", "mathematical_hint", "structured_lemma", "case_split", "sat_certificate", "external_reasoning_note"}:
            intervention["verification_status"] = "REJECTED"; self._save(); return None
        if kind == "external_reasoning_note": self._save(); return None
        if kind in {"structured_lemma", "case_split"}:
            alternatives=raw.get("or") or raw.get("alternatives")
            if not isinstance(alternatives,list) or not alternatives or not all(isinstance(a,list) and a and all(isinstance(i,int) and 0 <= i <= int(self.specification["n"]) for i in a) for a in alternatives):
                intervention["verification_status"]="REJECTED"; self.metrics["rejected_lemmas"]+=1; self._save(); return None
            terms=["("+" && ".join(f"A.contains {i}" for i in a)+")" for a in alternatives]
            formal=f"(candidates.all (fun A => !(admissible A) || ({' || '.join(terms)}))) = true"
            text=text or " or ".join(" and ".join(f"{i} is selected" for i in a) for a in alternatives)
        elif kind == "lean_lemma":
            # A file must explicitly compile against the frozen semantics.
            source=Path(raw.get("lean_path", "")); theorem=str(raw.get("theorem", ""))
            if not source.is_file() or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", theorem):
                intervention["verification_status"]="REJECTED"; self._save(); return None
            formal=str(raw.get("formal_statement", ""))
            if not formal:
                intervention["verification_status"]="REJECTED"; self._save(); return None
            copied=self.path/"lemmas"/f"human_source_{len(self.lemmas)+1}.lean"; copied.write_text(self.path.joinpath("Semantics.lean").read_text()+"\n"+source.read_text()+f"\n#check {theorem}\nexample : {formal} := {theorem}\n")
            try: ok=subprocess.run([resolve_lean(),copied.name],cwd=copied.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90).returncode==0
            except Exception: ok=False
            if not ok: intervention["verification_status"]="REJECTED"; self._save(); return None
            # Preserve the actual checked declaration and its source hash.
            text=text or theorem
            lid=f"L{len(self.lemmas)+1}"
            lemma=SessionLemma(lid,text,formal,"validated imported Lean theorem",producer="human-import",verification_status="VERIFIED",artifact_path=str(copied),artifact_hash=sha256_file(copied),imported_theorem=theorem)
            self.lemmas.append(lemma); self.dag.add(ProofDagNode(lid,text,_digest([text,formal]),producer="human-import")); self.dag.verify(lid,verifier="Lean",artifact_hash=lemma.artifact_hash)
            self.metrics["verified_lemmas"]+=1; self.metrics["lean_checks"]+=1; self.metrics["proof_dag_nodes"]=len(self.dag.nodes)
            intervention["verification_status"]="VERIFIED"; intervention["lemma_id"]=lid; self._save(); return lemma
        elif kind == "mathematical_hint":
            existing=next((x for x in self.lemmas if x.statement == text and x.verification_status=="VERIFIED"), None)
            if not existing:
                intervention["verification_status"]="REJECTED"; self.metrics["rejected_lemmas"]+=1; self._save(); return None
            formal=existing.formal_statement
        else:
            intervention["verification_status"] = "REJECTED"; self.metrics["rejected_lemmas"] += 1; self._save(); return None
        lemma = self._add_lemma(text, formal, "validated human-proposed semantic deduction", human=True)
        intervention["verification_status"] = lemma.verification_status; intervention["lemma_id"] = lemma.stable_id; self._save(); return lemma

    def resume(self, human_file: str | Path, *, allow_closer: bool = True, progress: bool = False) -> "ProofSession":
        self._verify_resume_integrity()
        if self.adapter:
            return self._resume_adapter(human_file, progress=progress)
        if self.model is None:
            if self.domain_adapter is None: raise ValueError("SESSION_REQUIRES_DOMAIN_ADAPTER")
            self.model = self.domain_adapter.build_constraint_model(_digest(self.specification)); facts, _, raw = propagate_constraints(self.model); self.trace = readable_deduction_trace(self.model, facts, raw)
        self.import_human(human_file)
        if allow_closer and self.status != SessionStatus.VERIFIED:
            self._close(progress)
        self.metrics["final_status"] = self.status.value; self._save(); return self

    def _resume_adapter(self, human_file: str | Path, *, progress: bool = False) -> "ProofSession":
        """Run an adapter's checked formal generator; the generic core only records hashes/DAG."""
        assert self.adapter is not None
        hint=Path(human_file)
        if not hint.is_file(): raise ValueError("HUMAN_ARTIFACT_INVALID")
        intervention={"type":"mathematical_hint", "source":str(hint), "artifact_hash":sha256_file(hint),
                      "producer":"human", "trust":"UNTRUSTED_ADVISORY", "verification_status":"ADVISORY"}
        self.human_interventions.append(intervention); self.metrics["human_interventions"]+=1
        generator=self.adapter_metadata.get("formal_artifact_generator")
        if not isinstance(generator, dict): raise ValueError("ADAPTER_RESUME_GENERATOR_MISSING")
        script=self.adapter.artifact_path(generator)
        output=self.path/"adapter_final.lean"
        cp=subprocess.run([sys.executable, str(script), "--output", str(output)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        if cp.returncode:
            intervention["verification_status"]="REJECTED"; self.status=SessionStatus.PROOF_RECOVERY_EXHAUSTED; self._save(); return self
        lean=subprocess.run([resolve_lean(), output.name], cwd=self.path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        self.metrics["lean_checks"]+=1
        if lean.returncode:
            intervention["verification_status"]="REJECTED"; self.status=SessionStatus.PROOF_RECOVERY_EXHAUSTED; self._save(); return self
        receipt_path=output.with_suffix(".json")
        receipt=json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        nodes=receipt.get("proof_nodes")
        if not isinstance(nodes, list) or not nodes or not all(isinstance(n,dict) and n.get("node_id") and isinstance(n.get("dependencies",[]),list) for n in nodes):
            raise ValueError("ADAPTER_FORMAL_RECEIPT_INVALID")
        intervention["verification_status"]="VERIFIED"
        artifact_hash=sha256_file(output)
        for item in nodes:
            node=ProofDagNode(str(item["node_id"]), str(item.get("statement", item["node_id"])), artifact_hash,
                              [str(x) for x in item.get("dependencies",[])], producer="adapter-formal-generator")
            self.dag.add(node); self.dag.verify(node.node_id, verifier="Lean", artifact_hash=artifact_hash)
        self.verifier_results.append({"artifact":str(output),"verifier":"Lean","status":"PASS","theorems":receipt.get("theorems",[])})
        self.metrics.update({"verified_lemmas":len(nodes),"certificate_type":"adapter-causal-formal-composition","proof_dag_nodes":len(self.dag.nodes),"final_status":SessionStatus.VERIFIED.value})
        self.status=SessionStatus.VERIFIED; self._save()
        if progress: print("[proof] adapter final lean=PASS")
        return self
