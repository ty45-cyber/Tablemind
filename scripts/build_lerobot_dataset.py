#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tablemind.execution.simulation import MuJoCoAdapter


TASK = "Set the dinner table with the utensils, plate, and cup."
IMAGE_SIZE = 256
CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
VERB_IDS = {"open": 0.0, "retrieve": 1.0, "place": 2.0, "verify": 3.0}
ARM_IDS = {"left": -1.0, "right": 1.0, "both": 0.0}


def scene_image(sim: MuJoCoAdapter, step: int, view: int) -> np.ndarray:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (238, 235, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 36, 236, 220), fill=(183, 145, 93), outline=(73, 55, 39), width=3)
    draw.ellipse((92, 92, 164, 164), fill=(228, 228, 220), outline=(70, 70, 65), width=2)

    x_scale = 150.0
    y_scale = 170.0
    for obj in sim.objects:
        x = int(128 + obj.x * x_scale)
        y = int(132 - obj.y * y_scale)
        color = {"spoon": (210, 210, 215), "fork": (170, 170, 180), "plate": (245, 245, 240), "cup": (105, 155, 200), "drawer": (110, 80, 55)}.get(obj.name, (80, 80, 80))
        radius = 11 if obj.name != "drawer" else 18
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(35, 35, 35))
        draw.text((x - 17, y + radius + 2), obj.name[:7], fill=(20, 20, 20))

    if view == 1:
        draw.line((35, 215, 105, 175), fill=(55, 95, 160), width=4)
    elif view == 2:
        draw.line((221, 215, 151, 175), fill=(160, 75, 55), width=4)
    draw.text((8, 8), f"TABLEMIND step {step}", fill=(20, 20, 20))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TABLEMIND reference episodes to a local LeRobot dataset.")
    parser.add_argument("--input", default="data/demos/reference.jsonl")
    parser.add_argument("--output", default="outputs/datasets/tablemind_smolvla")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.episodes is not None:
        records = records[: args.episodes]
    if not records:
        raise ValueError(f"No episodes found in {input_path}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        key: {"dtype": "image", "shape": (3, IMAGE_SIZE, IMAGE_SIZE), "names": ["channels", "height", "width"]}
        for key in CAMERA_KEYS
    }
    features.update({
        "observation.state": {"dtype": "float32", "shape": (6,), "names": ["state"] * 6},
        "action": {"dtype": "float32", "shape": (6,), "names": ["action"] * 6},
    })
    dataset = LeRobotDataset.create(
        repo_id="tablemind_smolvla",
        root=output_path,
        fps=5,
        features=features,
        robot_type="tablemind_dual_so101",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    for record in records:
        sim = MuJoCoAdapter(int(record["seed"]))
        for step, item in enumerate(record["steps"]):
            target_x, target_y = item["target_xy"]
            verb = VERB_IDS[item["verb"]]
            arm = ARM_IDS[item["arm"]]
            handoff = 1.0 if item["arm"] == "both" else 0.0
            progress = step / max(len(record["steps"]) - 1, 1)
            state = np.array([progress, target_x, target_y, arm, verb, handoff], dtype=np.float32)
            action = np.array([target_x, target_y, verb, arm, item.get("duration_s", 0.8), handoff], dtype=np.float32)
            frame = {key: scene_image(sim, step, index).transpose(2, 0, 1) for index, key in enumerate(CAMERA_KEYS)}
            frame.update({"observation.state": state, "action": action, "task": TASK})
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)

    print(f"wrote {len(records)} episodes to {output_path}")


if __name__ == "__main__":
    main()