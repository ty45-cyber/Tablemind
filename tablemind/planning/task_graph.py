from __future__ import annotations
from dataclasses import dataclass
from tablemind.core.models import Action, Observation, TaskStatus

@dataclass
class TaskNode:
    name: str
    arm: str
    target: str
    status: TaskStatus = TaskStatus.PENDING

class BimanualTaskPlanner:
    """Deterministic safety/planning layer around a learned VLA.

    The VLA proposes semantic actions; this planner decides sequencing,
    arm ownership, shared-workspace constraints and verification transitions.
    """
    def __init__(self) -> None:
        self.nodes = [
            TaskNode("open_drawer", "left", "drawer"),
            TaskNode("retrieve_spoon", "left", "spoon"),
            TaskNode("retrieve_fork", "right", "fork"),
            TaskNode("place_plate", "left", "plate"),
            TaskNode("place_cup", "right", "cup"),
            TaskNode("final_verify", "both", "table"),
        ]

    @property
    def current(self) -> TaskNode | None:
        for node in self.nodes:
            if node.status != TaskStatus.VERIFIED:
                return node
        return None

    def next_action(self, obs: Observation) -> Action | None:
        node = self.current
        if node is None:
            return None
        node.status = TaskStatus.ACTIVE
        targets = {o.name: (o.x, o.y) for o in obs.objects}
        xy = targets.get(node.target, (0.0, 0.0))
        verb = "verify" if node.name == "final_verify" else ("open" if node.name == "open_drawer" else "place" if node.name.startswith("place") else "retrieve")
        return Action(arm=node.arm, verb=verb, target=node.target, target_xy=xy, handoff=node.name in {"retrieve_fork", "place_cup"})

    def verify_current(self, success: bool) -> None:
        node = self.current
        if node is None:
            return
        if success:
            node.status = TaskStatus.VERIFIED
        else:
            node.status = TaskStatus.FAILED

    def replan_after_failure(self) -> None:
        node = self.current
        if node:
            node.status = TaskStatus.PENDING

    def is_complete(self) -> bool:
        return all(n.status == TaskStatus.VERIFIED for n in self.nodes)
