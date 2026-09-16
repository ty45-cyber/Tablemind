from __future__ import annotations
from tablemind.core.models import Action, Observation
from tablemind.planning.task_graph import BimanualTaskPlanner

class ReferencePolicy:
    name = "reference-policy"
    def __init__(self, planner: BimanualTaskPlanner) -> None:
        self.planner = planner
    def predict(self, observation: Observation) -> Action | None:
        return self.planner.next_action(observation)
