from __future__ import annotations

from pathlib import Path

from tablemind.evaluation.evaluator import evaluate
from tablemind.inference.router import PolicyRouter


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install the api extra: pip install -e '.[api]'") from exc

    app = FastAPI(
        title="TABLEMIND",
        version="0.6.0",
        description="Simulation-first bimanual Physical AI demo for the Intel online challenge.",
    )
    web_index = Path(__file__).resolve().parents[2] / "web" / "index.html"

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(web_index)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "track": "online-simulation-first",
            "physical_robot_required": False,
            "intel_validation": "requires_qualifying_core_ultra_hardware",
        }

    @app.get("/backends")
    def backends():
        return {"backends": [s.__dict__ for s in PolicyRouter().status()]}

    @app.get("/evaluate")
    def evaluation(seeds: int = 10):
        seeds = max(0, min(seeds, 1000))
        return evaluate(seeds)

    return app
