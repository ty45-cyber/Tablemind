# TABLEMIND policy training

This directory is intentionally aligned with the official Intel challenge path:
**MuJoCo -> LeRobot -> SmolVLA fine-tuning -> OpenVINO export/deployment**.

Training can run on local or cloud hardware. This guide supports a simulation-only deliverable: MuJoCo generation, LeRobot/SmolVLA training, and randomized simulation evaluation do not require physical Intel hardware. OpenVINO export is optional.

## 1. Generate demonstrations

From the repository root:

```bash
python scripts/generate_demos.py --episodes 200 --output data/demos
```

The generator creates task/episode metadata and successful reference trajectories for dataset development. For a production LeRobot dataset, convert the generated trajectories into the exact LeRobot dataset schema and preserve multiple examples per scene variation.

Build the local LeRobot dataset used by the training command with:

```bash
python scripts/build_lerobot_dataset.py \
  --input data/demos/reference.jsonl \
  --output outputs/datasets/tablemind_smolvla
```

## 2. Fine-tune SmolVLA

The current LeRobot documentation recommends fine-tuning `lerobot/smolvla_base` on task-specific data, with roughly 50 episodes as a starting point and repeated coverage of each task variation. For TABLEMIND we target 100–200 demonstrations so the perturbation matrix is represented rather than relying on one nominal scene.

Example:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=tablemind_smolvla \
  --dataset.root=outputs/datasets/tablemind_smolvla \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/train/tablemind_smolvla \
  --job_name=tablemind_smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false
```

The installed LeRobot 0.4.x release defaults to pushing policies to the Hub, so `--policy.push_to_hub=false` is required for local training. Set `DATASET_REPO_ID=tablemind_smolvla` and `DATASET_ROOT=outputs/datasets/tablemind_smolvla` when using the local dataset. Set `--policy.push_to_hub=true` and provide `--policy.repo_id=<HF_USER>/<POLICY_REPO>` only when Hub authentication and publishing are intended. Use the installed LeRobot `--help` output to adapt flags to the exact release in the environment. Do not commit model weights into this repository.

## 3. Export for Intel deployment

Produce an exported policy package compatible with OpenVINO Physical AI, preserving its manifest and preprocessing/postprocessing artifacts. The final package belongs under `exports/policy/` locally and is intentionally ignored from source control.

## 4. Validate the exported policy

```bash
python scripts/benchmark_policy.py --export-dir exports/policy --device CPU --iterations 500
```

Then repeat on the actual Core Ultra target for supported devices (`CPU`, `GPU`, `NPU` as available). Record real measurements only.



