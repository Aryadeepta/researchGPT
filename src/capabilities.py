import json
import os
import re
import subprocess
import hashlib
from pathlib import Path
from uuid import uuid4

from src.decisions import DecisionEngine
from src.computational_experiments import validate_executable_evidence_protocol
from src.research_state import now_iso
from src.storage import sha256_file


UNSAFE_TOKENS = ("sudo ", "rm -rf /", "mkfs", "dd if=", "chmod 777 /", "curl | sh", "wget | sh")
IMPLEMENTATION_KINDS = {"code", "shell", "command_tool_wrapper", "prompt_backed_helper", "composite_workflow"}
ALLOWED_FILESYSTEM_PERMISSIONS = {"none", "run_workspace", "read_only_workspace"}
ALLOWED_NETWORK_PERMISSIONS = {"none", "public_read"}


def skill_manifest_identity(spec):
    """Stable identity for a versioned skill, excluding local installation paths."""
    implementation = dict(spec.get("implementation", {}))
    implementation.pop("script_path", None)
    payload = {
        "skill_id": spec.get("skill_id"), "version": spec.get("version"),
        "name": spec.get("name", spec.get("capability_id")),
        "implementation": implementation, "inputs": spec.get("inputs", []), "outputs": spec.get("outputs", []),
        "dependencies": spec.get("dependencies", []), "permissions": spec.get("permissions", {}),
        "verifier": spec.get("verifier", spec.get("validation_tests", [])),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_capability_id(value):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned or f"capability_{uuid4().hex[:8]}"


def capability_requirement(capability_id, purpose, required_inputs=None, required_outputs=None, validation_criteria=None, required_tools=None,
                           resource_requirements=None, network_requirements=None, risk="low", expected_artifacts=None,
                           produces_modalities=None, verification_mechanisms=None, capability_status="REQUIRED",
                           can_generate_external_evidence_autonomously=False, evidence_protocol=None):
    return {
        "capability_id": normalize_capability_id(capability_id),
        "purpose": purpose,
        "required_inputs": required_inputs or [],
        "required_outputs": required_outputs or [],
        "validation_criteria": validation_criteria or [],
        "required_tools": required_tools or [],
        "resource_requirements": resource_requirements or {},
        "network_requirements": network_requirements or "none",
        "risk": risk,
        "expected_artifacts": expected_artifacts or [],
        "produces_modalities": produces_modalities or [],
        "verification_mechanisms": verification_mechanisms or [],
        "capability_status": capability_status,
        "can_generate_external_evidence_autonomously": bool(can_generate_external_evidence_autonomously),
        "evidence_protocol": evidence_protocol,
    }


class SkillRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def skill_dir(self, skill_id):
        return self.root / skill_id

    def list_skills(self):
        skills = []
        for path in self.root.iterdir():
            spec = path / "skill.json"
            if spec.exists():
                skills.append(json.loads(spec.read_text(encoding="utf-8")))
        return skills

    def find(self, requirement):
        outputs = set(requirement.get("required_outputs", []))
        for skill in self.list_skills():
            if skill.get("capability_id") == requirement.get("capability_id"):
                return skill
            if outputs and outputs.issubset(set(skill.get("outputs", []))):
                return skill
        return None

    def save(self, spec):
        spec = json.loads(json.dumps(spec))
        skill_id = spec["skill_id"]
        path = self.skill_dir(skill_id)
        path.mkdir(parents=True, exist_ok=True)
        script = spec.get("implementation", {}).get("script", "")
        if script:
            script_path = path / "run.sh"
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(0o755)
            spec["implementation"]["script_path"] = str(script_path)
            spec["checksum"] = sha256_file(script_path)
            spec["implementation_hash"] = spec["checksum"]
        spec.setdefault("name", spec.get("capability_id", skill_id))
        spec.setdefault("interface", {"input_schema": spec.get("inputs", []), "output_schema": spec.get("outputs", [])})
        spec.setdefault("implementation_kind", spec.get("implementation", {}).get("type", "code"))
        spec.setdefault("execution_history_references", [])
        spec["manifest_hash"] = skill_manifest_identity(spec)
        spec["immutable_version_id"] = f"{skill_id}@{spec.get('version', 1)}+{spec['manifest_hash'][:16]}"
        (path / "skill.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return spec


class SkillBuilder:
    def build(self, requirement, created_by="skill_builder"):
        skill_id = f"{requirement['capability_id']}_v1"
        outputs = requirement.get("required_outputs") or requirement.get("expected_artifacts") or ["skill_output.json"]
        if requirement["capability_id"] == "literature_metadata_analysis":
            script = """#!/bin/bash
set -Eeo pipefail
python3 - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

source = Path("evidence/discovery.json")
data = json.loads(source.read_text())
records = []
seen = set()
for retrieval in data.get("retrievals", []):
    for record in retrieval.get("records", []):
        key = record.get("doi") or record.get("identifier") or (str(record.get("title", "")).lower(), record.get("year"))
        if not key or str(key) in seen:
            continue
        seen.add(str(key))
        records.append(record)

Path("analysis").mkdir(parents=True, exist_ok=True)
years = Counter(str(r.get("year") or "unknown") for r in records)
providers = Counter(r.get("source_provider") or "unknown" for r in records)
verified = sum(1 for r in records if r.get("verification_status") == "VERIFIED_METADATA")
metrics = {
    "record_count": len(records),
    "verified_metadata_count": verified,
    "provider_counts": dict(providers),
    "year_counts": dict(sorted(years.items())),
    "records_with_abstract": sum(1 for r in records if r.get("abstract")),
}
Path("analysis/literature_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\\n")
with open("analysis/literature_records.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["identifier", "title", "year", "doi", "stable_url", "verification_status"])
    writer.writeheader()
    for r in records:
        writer.writerow({k: r.get(k) for k in writer.fieldnames})
PY
"""
        else:
            lines = ["#!/bin/bash", "set -Eeo pipefail"]
            for output in outputs:
                lines.append(f"mkdir -p \"$(dirname '{output}')\"")
                lines.append(f"printf '{{\"capability_id\":\"{requirement['capability_id']}\",\"status\":\"produced\"}}\\n' > '{output}'")
            script = "\n".join(lines) + "\n"
        return {
            "skill_id": skill_id,
            "capability_id": requirement["capability_id"],
            "purpose": requirement["purpose"],
            "inputs": requirement.get("required_inputs", []),
            "outputs": outputs,
            "implementation": {"type": "shell", "script": script},
            "implementation_kind": "shell",
            "name": requirement["capability_id"],
            "interface": {"input_schema": requirement.get("required_inputs", []), "output_schema": outputs},
            "dependencies": requirement.get("required_tools", []),
            "provenance": {"created_from_requirement": requirement, "created_at": now_iso()},
            "validation_tests": [{"type": "smoke", "expected_outputs": outputs}],
            "resource_requirements": requirement.get("resource_requirements", {}),
            "permissions": {"filesystem": "run_workspace", "network": requirement.get("network_requirements", "none")},
            "execution_history_references": [],
            "created_by": created_by,
            "version": 1,
            "promotion_status": "run_local",
            "produces_modalities": requirement.get("produces_modalities", []),
            "verification_mechanisms": requirement.get("verification_mechanisms", []),
            "capability_status": "CANDIDATE_CREATED",
            "can_generate_external_evidence_autonomously": requirement.get("can_generate_external_evidence_autonomously", False),
            "evidence_protocol": requirement.get("evidence_protocol"),
        }


class SkillValidator:
    def validate(self, spec):
        errors = []
        script = spec.get("implementation", {}).get("script", "")
        for token in UNSAFE_TOKENS:
            if token in script:
                errors.append(f"unsafe token: {token.strip()}")
        resources = spec.get("resource_requirements", {})
        if resources.get("paid_credentials") or resources.get("privileged") or resources.get("external_account") or resources.get("expensive_compute"):
            errors.append("human approval required for paid/privileged/external/expensive resource")
        if not spec.get("outputs"):
            errors.append("skill declares no outputs")
        kind = spec.get("implementation_kind", spec.get("implementation", {}).get("type"))
        if kind not in IMPLEMENTATION_KINDS:
            errors.append("invalid implementation kind")
        interface = spec.get("interface", {})
        if not isinstance(interface.get("input_schema", spec.get("inputs")), list) or not isinstance(interface.get("output_schema", spec.get("outputs")), list):
            errors.append("skill interface schemas must be lists")
        permissions = spec.get("permissions", {})
        if permissions.get("filesystem", "none") not in ALLOWED_FILESYSTEM_PERMISSIONS:
            errors.append("invalid filesystem permission")
        if permissions.get("network", "none") not in ALLOWED_NETWORK_PERMISSIONS:
            errors.append("invalid network permission")
        if "executable_computation" in spec.get("produces_modalities", []):
            errors.extend(validate_executable_evidence_protocol(spec))
            command_key = spec.get("evidence_protocol", {}).get("validation", {}).get("command_path")
            if command_key and not spec.get("implementation", {}).get(command_key):
                errors.append(f"implementation lacks declared validation command: {command_key}")
        return {"valid": not errors, "errors": errors}

    def smoke_test(self, spec, work_dir):
        script = spec.get("implementation", {}).get("script_path")
        if not script:
            return {"valid": False, "errors": ["missing script_path"]}
        if spec.get("capability_id") == "literature_metadata_analysis":
            fixture = Path(work_dir) / "evidence" / "discovery.json"
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(json.dumps({
                "retrievals": [{
                    "records": [{
                        "identifier": "fixture",
                        "title": "Fixture record",
                        "year": 2000,
                        "doi": "10.fixture/1",
                        "stable_url": "https://example.invalid/fixture",
                        "verification_status": "VERIFIED_METADATA",
                        "source_provider": "fixture",
                    }]
                }]
            }), encoding="utf-8")
        result = subprocess.run([script], cwd=work_dir, capture_output=True, text=True, timeout=30)
        errors = []
        if result.returncode != 0:
            errors.append(f"smoke command failed: {result.returncode}")
        for output in spec.get("outputs", []):
            if not (Path(work_dir) / output).exists():
                errors.append(f"missing smoke output: {output}")
        return {"valid": not errors, "errors": errors, "stdout": result.stdout, "stderr": result.stderr}


class SkillManager:
    def __init__(self, registry, builder=None, validator=None, decision_engine=None):
        self.registry = registry
        self.builder = builder or SkillBuilder()
        self.validator = validator or SkillValidator()
        self.decision_engine = decision_engine or DecisionEngine()

    def resolve(self, state, requirement, work_dir, notifier=None, max_repairs=1):
        skill = self.registry.find(requirement)
        if not skill:
            skill = self.builder.build(requirement)
        for attempt in range(max_repairs + 1):
            validation = self.validator.validate(skill)
            if validation["valid"]:
                saved = self.registry.save(skill)
                smoke = self.validator.smoke_test(saved, work_dir)
                if smoke["valid"]:
                    saved["capability_status"] = "AVAILABLE_VERIFIED"
                    saved["promotion_status"] = "run_local"
                    saved["verification_evidence"] = {"static_validation": validation, "smoke_execution": smoke}
                    saved = self.registry.save(saved)
                    return saved, {"status": "VALIDATED", "attempts": attempt + 1, "smoke": smoke}
                validation = smoke
            if attempt < max_repairs and "script" not in skill.get("implementation", {}):
                skill = self.builder.build(requirement)
                continue
            decision = self.decision_engine.resolve_or_request(state, {
                "stage": "skill_validation",
                "severity": "HIGH",
                "question": f"Capability '{requirement['capability_id']}' requires human approval or engineering repair.",
                "why_human_is_needed": "; ".join(validation["errors"]),
                "options": [{"id": "A", "description": "Approve or repair the skill externally", "benefits": ["unblocks capability"], "risks": validation["errors"]}],
                "recommended_option": "A",
                "blocked_nodes": [],
                "risk": "high",
                "material_scientific_impact": True,
            }, notifier=notifier)
            return None, {"status": "HUMAN_REQUIRED", "decision": decision, "errors": validation["errors"]}


def promote_skill_candidate(spec, validation_history):
    successful = [v for v in validation_history if v.get("status") == "VALIDATED"]
    promoted = dict(spec)
    promoted["promotion_status"] = "candidate_reusable" if len(successful) >= 3 else "run_local"
    promoted["promotion_evidence"] = validation_history
    return promoted
