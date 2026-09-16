from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    VERIFIED = "verified"
    FAILED = "failed"
    COMPLETE = "complete"

@dataclass(frozen=True)
class SceneObject:
    name: str
    category: str
    x: float
    y: float
    z: float
    graspable: bool = True

@dataclass
class Observation:
    seed: int
    instruction: str
    objects: list[SceneObject]
    joint_state: list[float]
    camera_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Action:
    arm: str
    verb: str
    target: str
    target_xy: tuple[float, float]
    duration_s: float = 0.8
    handoff: bool = False

@dataclass
class EpisodeResult:
    seed: int
    success: bool
    completed_steps: int
    total_steps: int
    recovered: bool
    collisions: int
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
