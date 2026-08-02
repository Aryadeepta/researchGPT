import json
import os

class EvidenceLedger:
    def __init__(self, project_dir):
        self.ledger_path = os.path.join(project_dir, "research_ledger.json")
        self.data = self._load_ledger()

    def _load_ledger(self):
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, "r") as f:
                return json.load(f)
        return {"claims": [], "status": "PARTIAL_RESEARCH"}

    def save(self):
        with open(self.ledger_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def update_claim(self, claim_id, **kwargs):
        for claim in self.data["claims"]:
            if claim["claim_id"] == claim_id:
                claim.update(kwargs)
                self.save()
                return True
        return False

    def is_paper_ready(self):
        # Hard gate: No UNVERIFIED claims allowed
        for claim in self.data["claims"]:
            if claim["status"] == "UNVERIFIED":
                return False
        return True
