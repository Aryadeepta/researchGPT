import json
import os
from datetime import datetime, timezone


CLAIM_STATUSES = {
    "HYPOTHESIS",
    "ANALYTICAL_ESTIMATE",
    "SUPPORTED_BY_CITATION",
    "VERIFIED_TOOL_OUTPUT",
    "VERIFIED_MEASUREMENT",
    "UNVERIFIED",
    "CONTRADICTED",
}


def _now():
    return datetime.now(timezone.utc).isoformat()

class EvidenceLedger:
    def __init__(self, project_dir):
        self.ledger_path = os.path.join(project_dir, "evidence_ledger.json")
        self.data = self._load_ledger()

    def _load_ledger(self):
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r") as f:
                data = json.load(f)
                return self._migrate(data)
        return self._migrate({"project": "Unknown", "status": "BLOCKED_MISSING_EVIDENCE", "claims": []})

    def _migrate(self, data):
        data.setdefault("schema_version", 2)
        data.setdefault("project", "Unknown")
        data.setdefault("status", "BLOCKED_MISSING_EVIDENCE")
        data.setdefault("claims", [])
        for idx, claim in enumerate(data["claims"], start=1):
            claim.setdefault("claim_id", f"C{idx:03d}")
            claim.setdefault("claim", "")
            claim.setdefault("status", "UNVERIFIED")
            if claim["status"] not in CLAIM_STATUSES:
                claim["status"] = "UNVERIFIED"
            claim.setdefault("origin", "legacy")
            claim.setdefault("producer", claim.get("origin", "legacy"))
            claim.setdefault("artifacts", [])
            claim.setdefault("validator_artifacts", [])
            claim.setdefault("validated_by", [])
            claim.setdefault("counterevidence", [])
            claim.setdefault("assumptions", [])
            claim.setdefault("limitations", [])
            claim.setdefault("replication_status", "NOT_ATTEMPTED")
            claim.setdefault("allowed_paper_language", "This remains an unverified hypothesis.")
            claim.setdefault("citation_ids", [])
            claim.setdefault("fatal_adversarial_findings", [])
            claim.setdefault("paper_role", "main")
            claim.setdefault("updated_at", _now())
        return data

    def save(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_claim(self, claim_id):
        return next((c for c in self.data["claims"] if c["claim_id"] == claim_id), None)

    def add_claim(self, claim):
        claim = dict(claim)
        claim.setdefault("claim_id", f"C{len(self.data['claims']) + 1:03d}")
        if self.get_claim(claim["claim_id"]):
            raise ValueError(f"duplicate claim_id: {claim['claim_id']}")
        claim.setdefault("status", "UNVERIFIED")
        claim.setdefault("origin", "unknown")
        claim.setdefault("producer", claim["origin"])
        claim.setdefault("artifacts", [])
        claim.setdefault("validator_artifacts", [])
        claim.setdefault("validated_by", [])
        claim.setdefault("counterevidence", [])
        claim.setdefault("assumptions", [])
        claim.setdefault("limitations", [])
        claim.setdefault("replication_status", "NOT_ATTEMPTED")
        claim.setdefault("allowed_paper_language", "This remains an unverified hypothesis.")
        claim.setdefault("citation_ids", [])
        claim.setdefault("fatal_adversarial_findings", [])
        claim.setdefault("paper_role", "main")
        claim["updated_at"] = _now()
        self._validate_claim(claim)
        self.data["claims"].append(claim)
        self.save()
        return claim

    def update_claim(self, claim_id, **kwargs):
        claim = self.get_claim(claim_id)
        if claim:
            before_status = claim.get("status")
            claim.update(kwargs)
            claim["updated_at"] = _now()
            self._validate_claim(claim, previous_status=before_status)
            self.save()
            return True
        return False

    def _validate_claim(self, claim, previous_status=None):
        status = claim.get("status")
        if status not in CLAIM_STATUSES:
            raise ValueError(f"invalid claim status: {status}")
        producer = claim.get("producer")
        if producer and producer in claim.get("validated_by", []):
            raise ValueError("no component may validate evidence that it generated itself")
        if status == "VERIFIED_MEASUREMENT":
            if claim.get("origin") == "planner" or producer == "planner":
                raise ValueError("planner cannot register measured results")
            if not claim.get("artifacts"):
                raise ValueError("verified measurements require raw artifacts")
            if claim.get("hard_coded_results"):
                raise ValueError("hard-coded results cannot be verified measurements")
        if status == "VERIFIED_TOOL_OUTPUT" and not claim.get("artifacts"):
            raise ValueError("verified tool output requires artifacts")
        if status == "SUPPORTED_BY_CITATION" and not claim.get("citation_ids"):
            raise ValueError("citation-supported claims require citation ids")
        if previous_status and previous_status != status and claim.get("updated_by") == "paper_writer":
            raise ValueError("paper writers cannot upgrade claim evidence")

    def is_paper_ready(self):
        if self.data.get("status") != "RESEARCH_COMPLETE":
            return False
        for claim in self.data["claims"]:
            if claim.get("paper_role", "main") == "main" and claim["status"] not in ["VERIFIED_MEASUREMENT", "VERIFIED_TOOL_OUTPUT", "SUPPORTED_BY_CITATION"]:
                return False
            if claim.get("replication_status") == "FAILED":
                return False
            if claim.get("fatal_adversarial_findings"):
                return False
        return True

    def get_unverified_claims(self):
        return [c for c in self.data["claims"] if c["status"] == "UNVERIFIED"]
