"""Domain-neutral, immutable semantic adapter manifests for proof sessions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ProofSemanticAdapter:
    """Validated manifest for task-local semantics, without domain assumptions.

    The manifest is intentionally data-only.  Optional executable/model and
    formal-generator entries are paths plus hashes, never answer fields.
    """
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ProofSemanticAdapter":
        source = Path(path).resolve()
        raw = json.loads(source.read_text())
        required = ("adapter_id", "adapter_version", "specification", "obligation",
                    "semantics_artifact", "executable_checker")
        if any(not raw.get(k) for k in required):
            raise ValueError("ADAPTER_MANIFEST_MISSING_REQUIRED_FIELD")
        root = source.parent
        for key in ("semantics_artifact", "executable_checker", "finite_model", "formal_artifact_generator"):
            entry = raw.get(key)
            if entry is None:
                continue
            if not isinstance(entry, dict) or not entry.get("path"):
                raise ValueError("ADAPTER_MANIFEST_INVALID_ARTIFACT")
            target = (root / entry["path"]).resolve()
            if root not in target.parents and target != root:
                raise ValueError("ADAPTER_MANIFEST_PATH_ESCAPE")
            if not target.is_file():
                raise ValueError("ADAPTER_MANIFEST_ARTIFACT_MISSING")
            actual = file_hash(target)
            if entry.get("sha256") and entry["sha256"] != actual:
                raise ValueError("ADAPTER_MANIFEST_ARTIFACT_HASH_MISMATCH")
            entry["sha256"] = actual
        raw.setdefault("specification_hash", digest(raw["specification"]))
        raw.setdefault("obligation_hash", digest(raw["obligation"]))
        identity = {k: v for k, v in raw.items() if k != "adapter_hash"}
        raw.setdefault("adapter_hash", digest(identity))
        if raw["adapter_hash"] != digest(identity):
            raise ValueError("ADAPTER_MANIFEST_HASH_MISMATCH")
        return cls(root, raw)

    @property
    def adapter_id(self) -> str: return str(self.manifest["adapter_id"])
    @property
    def adapter_version(self) -> str: return str(self.manifest["adapter_version"])
    @property
    def adapter_hash(self) -> str: return str(self.manifest["adapter_hash"])

    def metadata(self) -> dict[str, Any]:
        m = self.manifest
        return {"adapter_id": self.adapter_id, "adapter_version": self.adapter_version,
                "adapter_hash": self.adapter_hash, "specification_hash": m["specification_hash"],
                "obligation_hash": m["obligation_hash"], "semantics_artifact": m["semantics_artifact"],
                "semantics_hash": m["semantics_artifact"]["sha256"],
                "executable_checker": m["executable_checker"],
                "finite_model": m.get("finite_model"),
                "formal_artifact_generator": m.get("formal_artifact_generator")}

    def artifact_path(self, entry: dict[str, Any]) -> Path:
        return (self.root / entry["path"]).resolve()
