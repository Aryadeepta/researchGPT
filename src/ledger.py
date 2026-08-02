import json
import os

class EvidenceLedger:
    def __init__(self, project_dir):
        self.ledger_path = os.path.join(project_dir, "evidence_ledger.json")
        self.data = self._load_ledger()

    def _load_ledger(self):
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r") as f:
                data = json.load(f)
                # Ensure claims list exists
                if "claims" not in data:
                    data["claims"] = []
                return data
        return {"project": "Unknown", "status": "BLOCKED_MISSING_EVIDENCE", "claims": []}

    def save(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_claim(self, claim_id):
        return next((c for c in self.data["claims"] if c["claim_id"] == claim_id), None)

    def update_claim(self, claim_id, **kwargs):
        claim = self.get_claim(claim_id)
        if claim:
            claim.update(kwargs)
            self.save()
            return True
        return False

    def is_paper_ready(self):
        # All claims must be verified
        for claim in self.data["claims"]:
            if claim["status"] not in ["VERIFIED_MEASUREMENT", "VERIFIED_TOOL_OUTPUT", "SUPPORTED_BY_CITATION"]:
                return False
        return True

    def get_unverified_claims(self):
        return [c for c in self.data["claims"] if c["status"] == "UNVERIFIED"]
