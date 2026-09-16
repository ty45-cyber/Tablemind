from __future__ import annotations
import importlib.util
from dataclasses import dataclass

@dataclass(frozen=True)
class BackendStatus:
    backend: str
    available: bool
    reason: str

class PolicyRouter:
    def status(self, export_dir: str = "exports/policy") -> list[BackendStatus]:
        has_ov = importlib.util.find_spec("physicalai") is not None
        has_lerobot = importlib.util.find_spec("lerobot") is not None
        return [
            BackendStatus("openvino-physicalai", has_ov, "physicalai installed" if has_ov else "install physicalai to load exported policy"),
            BackendStatus("smolvla", has_lerobot, "lerobot installed" if has_lerobot else "install lerobot[smolvla] to load SmolVLA"),
            BackendStatus("reference", True, "deterministic simulation policy always available"),
        ]
