import argparse
import json
import os

from src.decisions import list_open_decisions
from src.paper_pipeline import PaperPipeline, PaperStore, verify_paper
from src.storage import artifact_store_from_env


def print_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


def paper_store_from_env():
    root = os.environ.get("PAPER_ARTIFACT_ROOT") or os.path.join(os.environ.get("RESEARCH_ARTIFACT_ROOT", "research_runs"), "papers")
    return PaperStore(root)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--paper-id", required=True)
    start.add_argument("--research-run-id", required=True)
    start.add_argument("--package-version", type=int)
    for name in ("resume", "status", "decisions", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--paper-id", required=True)
        p.add_argument("--max-nodes", type=int, default=10)
    decide = sub.add_parser("decide")
    decide.add_argument("--paper-id", required=True)
    decide.add_argument("--decision-id", required=True)
    decide.add_argument("--option")
    decide.add_argument("--text")
    args = parser.parse_args()
    pipeline = PaperPipeline(artifact_store_from_env(), paper_store_from_env())

    if args.command == "start":
        state = pipeline.start(args.paper_id, args.research_run_id, args.package_version)
        print(state.get("status"))
        return 0
    if args.command == "resume":
        state = pipeline.resume(args.paper_id, args.max_nodes)
        print(state.get("status"))
        return 0
    if args.command == "status":
        state = paper_store_from_env().load(args.paper_id)
        print(state.get("status", "missing") if state else "missing")
        return 0
    if args.command == "decisions":
        state = paper_store_from_env().load(args.paper_id) or {}
        print_json(list_open_decisions(state))
        return 0
    if args.command == "decide":
        print_json(pipeline.decide(args.paper_id, args.decision_id, args.option, args.text))
        return 0
    if args.command == "verify":
        print_json(verify_paper(paper_store_from_env(), args.paper_id))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
