import json
import tempfile
import unittest
from pathlib import Path

from src.capabilities import SkillBuilder, SkillManager, SkillRegistry, SkillValidator, capability_requirement, promote_skill_candidate
from src.decisions import AGENTIC_RESOLUTION, AUTO, HUMAN_REQUIRED, DecisionEngine, list_open_decisions, submit_decision
from src.engineering import create_engineering_request
from src.notifier import ConsoleNotifier, GitHubNotifier
from src.paper_pipeline import PaperPipeline, PaperStore, verify_paper
from src.research_package import load_research_package, write_research_package
from src.research_state import complete_node, create_run_state
from src.storage import LocalArtifactStore
from src.verification import verify_research_run


def completed_research_fixture(root, run_id="run1"):
    store = LocalArtifactStore(Path(root) / "runs")
    state = create_run_state(run_id, "input topic", dag=[
        {"node_id": "execute", "kind": "execution", "dependencies": [], "contract": {"requires_execution": True, "raw_outputs": ["experiments/raw.csv"]}},
    ])
    raw = Path(root) / "raw.csv"
    raw.write_text("value\n1\n", encoding="utf-8")
    entry = store.put_artifact(run_id, raw, "experiments/raw.csv", "executor")
    state["artifact_manifest"]["artifacts"].append(entry)
    state["claim_evidence_ledger"]["claims"].append({
        "claim_id": "C001",
        "claim": "A measured result exists.",
        "status": "VERIFIED_MEASUREMENT",
        "producer": "executor",
        "validated_by": ["independent_validator"],
        "artifacts": ["experiments/raw.csv"],
        "replication_status": "PASSED",
        "paper_role": "main",
        "objective_relation": "DIRECT_ANSWER",
    })
    state["selected_question"] = "What measured result was produced?"
    state["selected_objective_coverage"] = {"status": "SUFFICIENT", "claim_to_objective_mapping": [{"claim_id": "C001", "relation": "DIRECT_ANSWER"}]}
    complete_node(state, "execute", [entry])
    state["replication_status"] = "PASSED"
    store.atomic_update_state(run_id, state)
    package, report = write_research_package(store, state)
    store.atomic_update_state(run_id, state)
    assert package, report
    return store, state, package


class GeneralKernelTests(unittest.TestCase):
    def test_research_can_complete_without_invoking_paper_generation(self):
        with tempfile.TemporaryDirectory() as d:
            store, state, package = completed_research_fixture(d)
            self.assertEqual(state["status"], "RESEARCH_COMPLETE")
            self.assertFalse((Path(d) / "papers").exists())
            self.assertEqual(package["version"], 1)

    def test_paper_generation_consumes_completed_research_package(self):
        with tempfile.TemporaryDirectory() as d:
            store, _, package = completed_research_fixture(d)
            paper_store = PaperStore(Path(d) / "papers")
            pipeline = PaperPipeline(store, paper_store)
            state = pipeline.start("paper1", "run1")
            self.assertEqual(state["research_package_ref"]["package_hash"], package["package_hash"])
            state = pipeline.resume("paper1")
            self.assertEqual(state["status"], "PAPER_COMPLETE")
            self.assertEqual(verify_paper(paper_store, "paper1")["status"], "PASS")

    def test_paper_generation_refuses_incomplete_research_package(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            store.atomic_update_state("run1", create_run_state("run1", "topic"))
            state = PaperPipeline(store, PaperStore(Path(d) / "papers")).start("paper1", "run1")
            self.assertEqual(state["status"], "BLOCKED_RESEARCH_NOT_READY")

    def test_paper_generation_cannot_mutate_research_claim_status(self):
        with tempfile.TemporaryDirectory() as d:
            store, before, _ = completed_research_fixture(d)
            PaperPipeline(store, PaperStore(Path(d) / "papers")).start("paper1", "run1")
            after = store.load_state("run1")
            self.assertEqual(before["claim_evidence_ledger"], after["claim_evidence_ledger"])

    def test_research_revision_produces_new_package_version(self):
        with tempfile.TemporaryDirectory() as d:
            store, state, first = completed_research_fixture(d)
            state = store.load_state("run1")
            state["known_limitations"].append("additional limitation")
            second, _ = write_research_package(store, state)
            self.assertEqual(first["version"], 1)
            self.assertEqual(second["version"], 2)
            self.assertNotEqual(first["package_hash"], second["package_hash"])

    def test_existing_paper_remains_tied_to_original_package(self):
        with tempfile.TemporaryDirectory() as d:
            store, state, first = completed_research_fixture(d)
            paper_store = PaperStore(Path(d) / "papers")
            pipeline = PaperPipeline(store, paper_store)
            pipeline.start("paper1", "run1")
            state = store.load_state("run1")
            state["known_limitations"].append("new later limitation")
            write_research_package(store, state)
            pipeline.resume("paper1")
            paper = json.loads((paper_store.paper_root("paper1") / "paper_package.json").read_text())
            self.assertEqual(paper["research_package_ref"]["package_hash"], first["package_hash"])

    def test_both_pipelines_can_create_human_decision_requests(self):
        engine = DecisionEngine()
        state = create_run_state("run1", "topic")
        request = engine.resolve_or_request(state, {
            "stage": "scope",
            "question": "Choose materially different direction?",
            "options": [{"id": "A", "description": "A"}],
            "risk": "high",
            "material_scientific_impact": True,
        })
        self.assertEqual(request["status"], "WAITING_FOR_HUMAN")
        paper_state = {"run_id": "paper1", "status": "PLANNED", "decisions": [], "decision_history": []}
        request2 = engine.resolve_or_request(paper_state, {
            "stage": "narrative",
            "question": "Exclude validated central result?",
            "options": [{"id": "A", "description": "A"}],
            "risk": "high",
            "material_scientific_impact": True,
        })
        self.assertEqual(request2["status"], "WAITING_FOR_HUMAN")

    def test_responding_to_decision_resumes_correct_pipeline(self):
        state = create_run_state("run1", "topic")
        engine = DecisionEngine()
        req = engine.resolve_or_request(state, {"stage": "x", "question": "Q", "options": [{"id": "A", "description": "A"}], "risk": "high", "material_scientific_impact": True})
        submit_decision(state, req["decision_id"], "A")
        self.assertEqual(list_open_decisions(state), [])
        self.assertEqual(state["decision_history"][0]["selected_option"], "A")

    def test_core_works_without_github(self):
        with tempfile.TemporaryDirectory() as d:
            store, _, _ = completed_research_fixture(d)
            self.assertEqual(verify_research_run(store, "run1")["status"], "PASS")

    def test_optional_github_notifications_do_not_alter_research_semantics(self):
        state = create_run_state("run1", "topic")
        before = json.dumps(state["dag"], sort_keys=True)
        with tempfile.TemporaryDirectory() as d:
            notifier = GitHubNotifier(str(Path(d) / "issue.txt"))
            notifier.notify_transition(state, "WAITING_FOR_HUMAN", {"decision_id": "D1", "question": "Q"})
        self.assertEqual(before, json.dumps(state["dag"], sort_keys=True))

    def test_no_core_conditional_branches_keyed_on_field_names(self):
        src = "\n".join(p.read_text(encoding="utf-8") for p in Path("src").glob("*.py"))
        forbidden_conditionals = [
            "if topic ==",
            "if domain ==",
            "elif topic ==",
            "elif domain ==",
            "if field ==",
            "elif field ==",
        ]
        self.assertFalse([word for word in forbidden_conditionals if word in src.lower()])

    def test_capability_requirements_are_data_driven(self):
        req = capability_requirement("new evidence parser", "parse project evidence", required_outputs=["out.json"])
        self.assertEqual(req["capability_id"], "new_evidence_parser")
        self.assertEqual(req["required_outputs"], ["out.json"])

    def test_run_local_skills_add_capabilities_without_kernel_change(self):
        with tempfile.TemporaryDirectory() as d:
            registry = SkillRegistry(Path(d) / "skills")
            req = capability_requirement("arbitrary capability", "produce arbitrary artifact", required_outputs=["out.json"])
            skill, result = SkillManager(registry).resolve(create_run_state("r", "t"), req, d)
            self.assertEqual(result["status"], "VALIDATED")
            self.assertTrue((Path(d) / "out.json").exists())
            self.assertEqual(skill["promotion_status"], "run_local")

    def test_pipelines_can_use_arbitrary_skill_specifications(self):
        spec = SkillBuilder().build(capability_requirement("paper format x", "format paper", required_outputs=["paper.txt"]))
        self.assertEqual(spec["outputs"], ["paper.txt"])

    def test_absence_of_domain_builtin_does_not_block_if_skill_constructed(self):
        with tempfile.TemporaryDirectory() as d:
            req = capability_requirement("previously unseen method", "generic capability", required_outputs=["artifact.json"])
            skill, result = SkillManager(SkillRegistry(Path(d) / "skills")).resolve(create_run_state("r", "t"), req, d)
            self.assertIsNotNone(skill)
            self.assertEqual(result["status"], "VALIDATED")

    def test_skill_builder_validation_failure_bounded_repair_successful_execution(self):
        with tempfile.TemporaryDirectory() as d:
            req = capability_requirement("missing implementation", "repair", required_outputs=["out.json"])
            skill = {"skill_id": "bad_v1", "capability_id": req["capability_id"], "purpose": "bad", "outputs": ["out.json"], "implementation": {}, "resource_requirements": {}, "permissions": {}}
            registry = SkillRegistry(Path(d) / "skills")
            registry.save(skill)
            resolved, result = SkillManager(registry).resolve(create_run_state("r", "t"), req, d, max_repairs=1)
            self.assertEqual(result["status"], "VALIDATED")
            self.assertTrue((Path(d) / "out.json").exists())

    def test_skill_requires_privileged_paid_resource_human_required(self):
        with tempfile.TemporaryDirectory() as d:
            req = capability_requirement("paid tool", "needs paid tool", required_outputs=["out.json"], resource_requirements={"paid_credentials": True}, risk="high")
            skill, result = SkillManager(SkillRegistry(Path(d) / "skills")).resolve(create_run_state("r", "t"), req, d)
            self.assertIsNone(skill)
            self.assertEqual(result["status"], "HUMAN_REQUIRED")

    def test_human_automation_levels(self):
        engine = DecisionEngine()
        self.assertEqual(engine.classify({"risk": "low", "reversible": True, "estimated_cost_usd": 0}), AUTO)
        self.assertEqual(engine.classify({"risk": "medium", "reversible": True, "candidate_analyses": [{"choice": "A", "confidence": 0.8}, {"choice": "A", "confidence": 0.9}]}), AGENTIC_RESOLUTION)
        self.assertEqual(engine.classify({"risk": "high", "material_scientific_impact": True}), HUMAN_REQUIRED)

    def test_notification_deduplication(self):
        state = create_run_state("run1", "topic")
        notifier = ConsoleNotifier()
        payload = {"decision_id": "D1", "question": "Q"}
        self.assertTrue(notifier.notify_transition(state, "HUMAN_REQUIRED", payload))
        self.assertFalse(notifier.notify_transition(state, "HUMAN_REQUIRED", payload))

    def test_engineering_request_blocks_with_codex_task(self):
        state = create_run_state("run1", "topic")
        request = create_engineering_request(state, "executor cannot run required trusted operation", "requires core change", ["src/executor.py"], "add adapter", ["unit test"])
        self.assertEqual(state["status"], "BLOCKED_ENGINEERING_REQUIRED")
        self.assertIn("generated_codex_prompt", request)

    def test_skill_promotion_requires_repeated_validation(self):
        spec = SkillBuilder().build(capability_requirement("repeat", "repeat", required_outputs=["out.json"]))
        promoted = promote_skill_candidate(spec, [{"status": "VALIDATED"}, {"status": "VALIDATED"}, {"status": "VALIDATED"}])
        self.assertEqual(promoted["promotion_status"], "candidate_reusable")


if __name__ == "__main__":
    unittest.main()
