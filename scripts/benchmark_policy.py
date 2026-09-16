#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="exports/policy")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", default="benchmarks/policy.json")
    args = parser.parse_args()

    report: dict[str, object] = {
        "benchmark_type": "openvino_physical_ai_inference",
        "device_requested": args.device,
        "export_dir": args.export_dir,
        "iterations_requested": args.iterations,
        "status": "not_run",
    }

    if importlib.util.find_spec("physicalai") is None:
        report["reason"] = "physicalai is not installed"
    else:
        try:
            from physicalai.inference import InferenceModel
            from physicalai.benchmark.performance.inference_benchmark import InferenceLatencyBenchmark

            model = InferenceModel(args.export_dir, backend="openvino", device=args.device)
            metrics = InferenceLatencyBenchmark(
                max_iters=args.iterations,
                warmup_iters=min(5, max(1, args.iterations)),
            ).run(model)
            report["status"] = "completed"
            report["metrics"] = metrics
            report["measurement_note"] = "Per-iteration inference metrics from the OpenVINO Physical AI benchmark."
            model.close()
        except Exception as exc:
            report["reason"] = str(exc)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
