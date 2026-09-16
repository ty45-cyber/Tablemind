#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tablemind.evaluation.evaluator import evaluate
from tablemind.execution.simulation import MuJoCoAdapter
from tablemind.planning.task_graph import BimanualTaskPlanner


WIDTH, HEIGHT, FPS = 1280, 720, 24
BG = (12, 20, 29)
INK = (235, 241, 244)
MUTED = (155, 172, 183)
CYAN = (67, 205, 220)
ORANGE = (246, 157, 80)
GREEN = (95, 218, 145)


def font(size: int, bold: bool = False):
    name = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    fallback = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name if Path(name).exists() else fallback, size)


def text(draw, xy, value, size=24, fill=INK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def base(title: str, eyebrow: str = "TABLEMIND  /  SIMULATION DEMO") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 10), fill=CYAN)
    text(draw, (64, 48), eyebrow, 18, CYAN, True)
    text(draw, (64, 84), title, 42, INK, True)
    draw.line((64, 150, WIDTH - 64, 150), fill=(43, 61, 73), width=2)
    return image, draw


def title_frame() -> Image.Image:
    image, draw = base("Bimanual dinner-table setup")
    text(draw, (64, 220), "MuJoCo simulation  |  Dual SO-101 arms", 30, INK)
    text(draw, (64, 270), "Natural-language instruction to coordinated action", 24, MUTED)
    draw.rounded_rectangle((64, 380, 1216, 490), radius=10, fill=(22, 39, 51), outline=(55, 82, 94), width=2)
    text(draw, (96, 414), '"Set the dinner table with the utensils, plate, and cup."', 25, CYAN)
    text(draw, (64, 630), "Simulation-only evidence package", 20, GREEN, True)
    return image


def scene_frame(seed: int, step: int, action_text: str, status: str) -> Image.Image:
    image, draw = base("Randomized scene and action execution")
    sim = MuJoCoAdapter(seed)
    left, top, right, bottom = 90, 210, 850, 640
    draw.rounded_rectangle((left, top, right, bottom), radius=12, fill=(163, 122, 78), outline=(223, 187, 126), width=3)
    draw.ellipse((390, 350, 550, 510), fill=(228, 228, 219), outline=(72, 72, 70), width=3)
    for obj in sim.objects:
        x = int(470 + obj.x * 310)
        y = int(415 - obj.y * 300)
        color = {"spoon": (216, 218, 223), "fork": (174, 178, 185), "plate": (245, 245, 240), "cup": (92, 157, 211), "drawer": (103, 71, 46)}.get(obj.name, (60, 60, 60))
        radius = 22 if obj.name != "drawer" else 34
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(35, 35, 35), width=2)
        text(draw, (x, y + radius + 8), obj.name, 15, (30, 30, 30), True, "ma")
    if action_text:
        color = ORANGE if "right" in action_text else CYAN
        draw.line((470, 595, 470 + (step % 2) * 170 - 85, 530), fill=color, width=7)
    draw.rounded_rectangle((900, 220, 1216, 430), radius=10, fill=(22, 39, 51))
    text(draw, (932, 250), f"SEED {seed:02d}", 18, CYAN, True)
    text(draw, (932, 300), f"STEP {step + 1} / 6", 22, INK, True)
    text(draw, (932, 350), status, 20, GREEN, True)
    draw.rounded_rectangle((900, 470, 1216, 590), radius=10, fill=(22, 39, 51))
    text(draw, (930, 495), "ACTION", 16, MUTED, True)
    text(draw, (930, 530), action_text, 21, INK, True)
    return image


def result_frame(report: dict) -> Image.Image:
    image, draw = base("Verification across randomized seeds")
    text(draw, (64, 210), "Reference-policy simulation result", 28, MUTED)
    cards = [(64, "SUCCESS RATE", f"{report['success_rate'] * 100:.0f}%", GREEN), (420, "EPISODES", str(report["episodes"]), CYAN), (776, "COLLISION-FREE", f"{report['collision_free_rate'] * 100:.0f}%", ORANGE)]
    for x, label, value, color in cards:
        draw.rounded_rectangle((x, 290, x + 300, 465), radius=12, fill=(22, 39, 51), outline=(55, 82, 94), width=2)
        text(draw, (x + 24, 325), label, 16, MUTED, True)
        text(draw, (x + 24, 370), value, 54, color, True)
    text(draw, (64, 570), "MuJoCo  |  200 generated episodes  |  No physical hardware required", 22, INK)
    text(draw, (64, 620), "Current evidence uses the deterministic ReferencePolicy; no trained SmolVLA claim.", 17, MUTED)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a presentation MP4 from the TABLEMIND simulation.")
    parser.add_argument("--output", default="outputs/tablemind_submission.mp4")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required and was not found on PATH")

    sim = MuJoCoAdapter(args.seed)
    planner = BimanualTaskPlanner()
    actions = []
    while not planner.is_complete():
        action = planner.next_action(sim.observe("Set the dinner table with the utensils, plate, and cup."))
        if action is None:
            break
        sim.execute(action)
        planner.verify_current(True)
        actions.append(f"{action.arm.upper()}  {action.verb} {action.target}")
    report = evaluate(10)

    with tempfile.TemporaryDirectory(prefix="tablemind-video-") as temp:
        frames = Path(temp)
        sequence = [title_frame()]
        for index, action in enumerate(actions):
            sequence.extend([scene_frame(args.seed, index, action, "VERIFIED") for _ in range(FPS // 2)])
        sequence.extend([result_frame(report) for _ in range(FPS * 3)])
        for index, image in enumerate(sequence):
            image.save(frames / f"frame_{index:05d}.png")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-framerate", str(FPS), "-i", str(frames / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote presentation video to {args.output}")


if __name__ == "__main__":
    main()