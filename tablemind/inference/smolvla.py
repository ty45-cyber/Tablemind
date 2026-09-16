from __future__ import annotations

from typing import Any


class SmolVLAAdapter:
    """Lazy LeRobot SmolVLA adapter.

    SmolVLA is a 450M-parameter policy intended for task-specific fine-tuning.
    The adapter deliberately keeps model loading optional so the simulation and
    CI path can run without downloading weights.
    """

    def __init__(self, model_path: str = "lerobot/smolvla_base", device: str = "cpu") -> None:
        try:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot SmolVLA dependencies are missing. Install the vla extra: "
                "pip install -e '.[vla]'"
            ) from exc

        self.model_path = model_path
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(model_path).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(self.policy.config)

    def predict(self, observation: dict[str, Any]) -> Any:
        model_input = self.preprocess(observation)
        action = self.policy.select_action(model_input)
        return self.postprocess(action)
