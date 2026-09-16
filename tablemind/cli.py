from __future__ import annotations
import argparse, json
from pathlib import Path
from tablemind.evaluation.evaluator import evaluate
from tablemind.evaluation.robustness import evaluate_perturbations
from tablemind.inference.router import PolicyRouter

def main() -> None:
    parser = argparse.ArgumentParser(prog="tablemind")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate"); sim.add_argument("--seed", type=int, default=0)
    ev = sub.add_parser("evaluate"); ev.add_argument("--seeds", type=int, default=10); ev.add_argument("--output", default="outputs/evaluation.json")
    rb = sub.add_parser("robustness"); rb.add_argument("--output", default="outputs/robustness.json")
    sub.add_parser("policy")
    args = parser.parse_args()
    if args.command == "simulate":
        report = evaluate(1)
        print(json.dumps(report, indent=2))
    elif args.command == "evaluate":
        report = evaluate(args.seeds)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "robustness":
        report = evaluate_perturbations()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "policy":
        print(json.dumps([s.__dict__ for s in PolicyRouter().status()], indent=2))


if __name__ == "__main__":
    main()
