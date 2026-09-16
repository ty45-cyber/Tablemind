#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import importlib.util, json, platform

def status(name):
    try:
        __import__(name)
        return {"name": name, "installed": True}
    except Exception:
        return {"name": name, "installed": False}

print(json.dumps({"python": platform.python_version(), "platform": platform.platform(), "packages": [status(x) for x in ["mujoco","lerobot","openvino","physicalai","fastapi"]]}, indent=2))
