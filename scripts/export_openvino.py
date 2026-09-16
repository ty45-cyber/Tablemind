#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/validate an OpenVINO Physical AI export directory.")
    parser.add_argument("--export-dir", default="exports/policy")
    parser.add_argument("--output", default="benchmarks/export_check.json")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    result = {
        "export_dir": str(export_dir),
        "physicalai_installed": importlib.util.find_spec("physicalai") is not None,
        "openvino_installed": importlib.util.find_spec("openvino") is not None,
        "export_present": export_dir.exists() and any(export_dir.iterdir()),
    }
    if result["export_present"]:
        result["artifacts"] = sorted(p.name for p in export_dir.rglob("*") if p.is_file())
    else:
        result["artifacts"] = []
        result["status"] = "not_ready"
        result["reason"] = "No exported policy package is present. Train and export a real policy first."

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
