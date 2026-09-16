# TABLEMIND v0.6.0 — Intel Physical AI / Simulation-First

TABLEMIND is a **simulation-first adaptive bimanual Vision-Language-Action (VLA) system** for Intel's online challenge, **"Bimanual VLA Manipulation with Multi-Modal Reasoning — Setting Up a Dinner Table."**

## Track boundary — locked

The official track specifies:

- **Event format:** Online / Simulation-first
- **Robot target:** Simulated Dual SO-101 Arms
- **Primary simulator:** MuJoCo
- **Training:** local hardware or cloud is allowed
- **Final deployment target:** Intel Core Ultra Series 2/3
- **Inference runtime:** OpenVINO
- **Physical SO-101 hardware:** not required

The simulation-only deliverable is a **MuJoCo simulation of two SO-101 arms**, with AI/VLA inference, randomized evaluation, and training performed without physical SO-101 or Intel hardware. OpenVINO and Core Ultra benchmarking are optional deployment extensions.

## Official software stack mapped into TABLEMIND

| Official track resource | TABLEMIND role |
|---|---|
| **MuJoCo** | Primary dual-arm simulation + randomized evaluation |
| **Hugging Face LeRobot** | Policy training/fine-tuning and dataset workflow |
| **SmolVLA** | Primary VLA candidate for task-specific fine-tuning |
| **Pi0.5 / ACT** | Alternative policy baselines, not required for v0.6 |
| **OpenVINO Toolkit** | Conversion/optimization/inference on Intel |
| **OpenVINO Physical AI** | Exported-policy runtime / action-chunk inference boundary |
| **Intel Physical AI Studio** | Optional workflow/integration layer |
| **Open Edge Platform / Edge AI Suites** | Optional deployment components if needed |
| **Intel Geti** | Optional perception tooling; not required in the core path |

The official track says to train or fine-tune in MuJoCo using LeRobot or compatible tooling and explicitly names SmolVLA, Pi0.5 and ACT as candidate policies. It separately lists OpenVINO, OpenVINO Physical AI, Physical AI Studio, Open Edge Platform, Edge AI Suites and Geti as software resources. [Official track PDF]

## End-to-end architecture

```text
Natural-language command
        |
        v
Language + scene grounding
        |
        v
Simulated camera observations + robot state
        |
        v
LeRobot-compatible SmolVLA policy
        |
        v
Action chunk / task intent
        |
        v
TABLEMIND bimanual coordination layer
     /                         \
 Left-arm control           Right-arm control
     \                         /
      --------> MuJoCo <-------
                 |
                 v
        State / visual verification
          /                 \
      success             failure
         |                  |
       next              diagnose
         |                  |
         +-------> replan --+

Optional Intel deployment boundary:
trained/exported policy
        -> OpenVINO Physical AI InferenceModel
        -> action chunk runtime
        -> Intel CPU/GPU/NPU
        -> Core Ultra Series 2/3
```

## Why SmolVLA is the primary candidate

Current LeRobot documentation describes SmolVLA as a 450M-parameter robotics VLA designed for task-specific fine-tuning. The documentation recommends starting around 50 demonstrations and emphasizes repeating each task variation enough times to support generalization. TABLEMIND therefore targets 100–200 high-quality simulation demonstrations covering the challenge's perturbation axes rather than relying on a single nominal scene.

## Training path

```text
MuJoCo randomized episodes
          |
          v
LeRobot-compatible demonstrations
          |
          v
SmolVLA fine-tuning
          |
          v
TABLEMIND task policy
          |
          v
Exported policy package
          |
          v
OpenVINO Physical AI
          |
          v
Core Ultra validation
```

Training can be performed on cloud or local hardware. For the simulation-only workflow, the final demonstration is the MuJoCo simulation plus AI/VLA/VLM inference and randomized evaluation; no Intel Core Ultra machine is required.

## Commands

### Offline CI/reference evaluation

```bash
pip install -e '.[dev]'
pytest -q
python -m tablemind.cli evaluate --seeds 10
```

### Real MuJoCo environment

```bash
pip install -e '.[sim]'
python -c "from tablemind.execution.simulation import load_mujoco_model; load_mujoco_model()"
```

### SmolVLA

```bash
pip install -e '.[vla]'
```

Then use `scripts/train_smolvla.sh` or the documented `lerobot-train` command from `training/README.md`.

### Simulation-only evaluation

```bash
pip install -e '.[sim,dev]'
python -m tablemind.cli evaluate --seeds 10
```

This is the required validation path for a simulation-only submission. It does not require OpenVINO, `physicalai`, Intel hardware, or a physical robot.

### OpenVINO Physical AI

```bash
pip install -e '.[intel]'
python scripts/export_openvino.py --export-dir exports/policy
python scripts/benchmark_policy.py --export-dir exports/policy --device CPU --iterations 500
```

OpenVINO export and Core Ultra benchmarking are optional and should only be run when the corresponding software and hardware are available.

## Robustness protocol

The official rubric calls for randomized object placement, weight, friction, shape, lighting and background, with the final demonstration across 10 randomized seeds. TABLEMIND supports a larger 100-seed internal suite via `configs/robustness.yaml`, so the public 10-seed clip can be backed by broader evidence.

## Optional Intel benchmark protocol

The project follows the current OpenVINO Physical AI boundary:

```text
exported package -> InferenceModel -> action selection/chunks
```

The current OpenVINO Physical AI docs expose `select_action()` for scripts/evaluation and `predict_action_chunk()` when a runtime owns queueing/timing. The benchmark script uses the package's `InferenceLatencyBenchmark` for per-iteration inference measurements.

Required evidence:

- device
- precision/export configuration
- warmup and measured inference
- mean/median/P95-style latency derived from captured samples
- throughput where applicable
- task success preserved after optimization

No Intel/Core Ultra numbers are hard-coded or fabricated.

## Railway

Railway is only the lightweight judge-facing evidence/API layer. It does **not** replace the required Intel Core Ultra execution environment.

```text
GitHub -> Railway -> FastAPI evidence API
                   |
                   +-- /health
                   +-- /backends
                   +-- /evaluate?seeds=10
```

The simulation-only demonstration is reproducible from this repository without a qualifying Intel machine. Railway remains optional evidence/API hosting and is not part of the local simulation requirement.

## Submission evidence plan

The simulation-only evidence package is a reproducible repository, MuJoCo simulation, 10-seed evaluation, demonstration video, and technical README. Intel benchmark evidence is optional for this workflow.

Our final recording should show:

1. Natural-language dinner-table instruction.
2. Randomized MuJoCo scene.
3. Simulated camera observation and scene grounding.
4. SmolVLA policy inference.
5. Coordinated two-arm manipulation, including complementary/handoff behavior.
6. Verification and recovery/replanning.
7. Ten randomized seeds and measured success rate.
8. Optional OpenVINO optimization if deployment evidence is desired.

## Truth boundary

This repository does not claim that a task-specific SmolVLA checkpoint has been trained until one is actually produced. It does not claim Core Ultra measurements until they are executed on qualifying hardware. It does not require or imply physical SO-101 hardware for the online event.
