# Simulation-Only Submission Evidence

Generated on 2026-09-16 from the local Windows Python 3.11 environment.

## Verified artifacts

- MuJoCo XML model loads successfully: `nq=5`, `nv=5`.
- `data/demos/reference.jsonl` contains 200 generated reference episodes.
- Ten-seed evaluation: 10/10 successful, 100% success rate, 100% collision-free rate.
- Robustness evaluation: 100% success for initial placement, lighting, friction, background, and shape variation categories.

## Reproduce

Create the presentation video from the terminal:

```powershell
.\.venv311\Scripts\python.exe scripts\make_submission_video.py --output outputs\tablemind_submission.mp4
```

The video uses real offscreen MuJoCo renders from `mujoco/tablemind_scene.xml`, with the task flow, randomized seed, bimanual planner actions, verification, and evaluation result overlaid. It is simulation evidence, not physical-robot footage.

```powershell
.\.venv311\Scripts\tablemind.exe evaluate --seeds 10 --output outputs\evaluation.json
.\.venv311\Scripts\tablemind.exe robustness --output outputs\robustness.json
.\.venv311\Scripts\python.exe scripts\generate_demos.py --episodes 200 --output data\demos
```

The generated JSON files and demonstrations are local evidence artifacts and are excluded from Git. The current evaluation uses the deterministic `ReferencePolicy`; it is not evidence of a trained SmolVLA checkpoint. No OpenVINO, Intel hardware, or physical robot measurements are claimed.