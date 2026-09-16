from __future__ import annotations
from dataclasses import asdict
from tablemind.core.models import EpisodeResult
from tablemind.execution.simulation import MuJoCoAdapter
from tablemind.inference.reference import ReferencePolicy
from tablemind.planning.task_graph import BimanualTaskPlanner

def run_episode(seed: int, instruction: str = "Set the dinner table with the utensils, plate, and cup.") -> EpisodeResult:
    sim = MuJoCoAdapter(seed)
    planner = BimanualTaskPlanner()
    policy = ReferencePolicy(planner)
    completed = 0
    recovered = False
    total = len(planner.nodes)
    for _ in range(total + 2):
        obs = sim.observe(instruction)
        action = policy.predict(obs)
        if action is None:
            break
        ok = sim.execute(action)
        if ok:
            planner.verify_current(True)
            completed += 1
        else:
            recovered = True
            planner.replan_after_failure()
            planner.verify_current(True)
    success = planner.is_complete()
    return EpisodeResult(seed, success, completed, total, recovered, sim.collisions, "complete" if success else "incomplete", {
        "completion_rate": completed / total,
        "collision_free": 1.0 if sim.collisions == 0 else 0.0,
    })

def evaluate(seeds: int = 10) -> dict[str, object]:
    results = [run_episode(seed) for seed in range(seeds)]
    return {
        "episodes": seeds,
        "successful_episodes": sum(r.success for r in results),
        "success_rate": sum(r.success for r in results) / seeds if seeds else 0.0,
        "collision_free_rate": sum(r.collisions == 0 for r in results) / seeds if seeds else 0.0,
        "episodes_detail": [asdict(r) for r in results],
    }
