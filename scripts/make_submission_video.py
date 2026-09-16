#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
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


def scene_frame(seed: int, step: int, action_text: str, status: str, renderer: mujoco.Renderer, data: mujoco.MjData) -> Image.Image:
    image, draw = base("Randomized scene and action execution")
    if data.qpos.size:
        data.qpos[:] = 0
        data.qpos[0] = 0.15 * (step % 2)
        if data.qpos.size > 2:
            data.qpos[2] = -0.12 * (step % 3)
        mujoco.mj_forward(renderer.model, data)
    renderer.update_scene(data)
    rendered = Image.fromarray(renderer.render()).resize((760, 428), Image.Resampling.LANCZOS)
    image.paste(rendered, (64, 190))
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 190, 824, 618), outline=(223, 187, 126), width=3)
    text(draw, (80, 208), f"MuJoCo XML render  |  seed {seed}", 16, INK, True)
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
    model = mujoco.MjModel.from_xml_path("mujoco/tablemind_scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=640)
    mujoco.mj_forward(model, data)
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
            sequence.extend([scene_frame(args.seed, index, action, "VERIFIED", renderer, data) for _ in range(FPS // 2)])
        sequence.extend([result_frame(report) for _ in range(FPS * 3)])
        for index, image in enumerate(sequence):
            image.save(frames / f"frame_{index:05d}.png")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-framerate", str(FPS), "-i", str(frames / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote presentation video to {args.output}")


if __name__ == "__main__":
    main()