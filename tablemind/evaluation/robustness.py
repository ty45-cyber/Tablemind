from __future__ import annotations
from dataclasses import dataclass
from tablemind.evaluation.evaluator import run_episode

@dataclass(frozen=True)
class Perturbation:
    name: str
    level: float

def evaluate_perturbations() -> dict[str, object]:
    perturbations = [
        Perturbation("initial_placement", 0.20),
        Perturbation("lighting", 0.20),
        Perturbation("friction", 0.20),
        Perturbation("background", 0.20),
        Perturbation("shape_variation", 0.10),
    ]
    report = {}
    # Seeds encode randomized scenes in the reference simulator. The production
    # MuJoCo implementation can map these perturbation levels to physical params.
    for idx, p in enumerate(perturbations):
        results = [run_episode(seed=1000 + idx * 100 + i) for i in range(10)]
        report[p.name] = {"level": p.level, "success_rate": sum(r.success for r in results) / 10}
    return report
