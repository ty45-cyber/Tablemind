.PHONY: test eval robustness demos api

test:
	python -m pytest -q

eval:
	python -m tablemind.cli evaluate --seeds 10 --output outputs/evaluation_10_seeds.json

robustness:
	python -m tablemind.cli robustness --output outputs/robustness.json

demos:
	python scripts/generate_demos.py --episodes 50

api:
	uvicorn tablemind.api.app:create_app --factory --reload
