#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
from tablemind.execution.simulation import MuJoCoAdapter
from tablemind.planning.task_graph import BimanualTaskPlanner

def main():
    p = argparse.ArgumentParser(); p.add_argument("--episodes", type=int, default=50); p.add_argument("--output", default="data/demos/reference.jsonl")
    a = p.parse_args(); path = Path(a.output)
    if path.suffix.lower() != ".jsonl":
        path = path / "reference.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seed in range(a.episodes):
            sim = MuJoCoAdapter(seed); planner = BimanualTaskPlanner(); steps=[]
            while not planner.is_complete():
                obs = sim.observe("Set the dinner table with the utensils, plate, and cup.")
                action = planner.next_action(obs)
                if action is None: break
                sim.execute(action); planner.verify_current(True)
                steps.append({"arm": action.arm, "verb": action.verb, "target": action.target, "target_xy": action.target_xy})
            f.write(json.dumps({"seed": seed, "steps": steps}) + "\n")
    print(f"wrote {a.episodes} reference episodes to {path}")

if __name__ == "__main__": main()
