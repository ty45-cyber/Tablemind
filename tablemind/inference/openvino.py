from __future__ import annotations
from pathlib import Path
from typing import Any

class OpenVINOPolicy:
    def __init__(self, export_dir: str, device: str = "AUTO") -> None:
        from physicalai.inference import InferenceModel
        self.export_dir = Path(export_dir)
        self.device = device
        self.model = InferenceModel(str(self.export_dir), backend="openvino", device=device)
    def reset(self) -> None:
        self.model.reset()
    def select_action(self, observation: dict[str, Any]) -> Any:
        return self.model.select_action(observation)
    def predict_action_chunk(self, observation: dict[str, Any]) -> Any:
        return self.model.predict_action_chunk(observation)
    def close(self) -> None:
        self.model.close()
