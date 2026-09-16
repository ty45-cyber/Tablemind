from __future__ import annotations
from tablemind.core.models import Observation

KEYWORDS = {
    "spoon": "spoons",
    "fork": "forks",
    "plate": "plate",
    "cup": "cup",
    "mug": "cup",
    "dinner": "dinner",
}

def ground_instruction(instruction: str, observation: Observation) -> dict[str, object]:
    text = instruction.lower()
    requested = [label for word, label in KEYWORDS.items() if word in text]
    visible = [obj.name for obj in observation.objects]
    return {
        "instruction": instruction,
        "intent": "set_dinner_table" if "dinner" in text or "table" in text else "manipulate_objects",
        "requested_categories": sorted(set(requested)),
        "visible_objects": visible,
    }
