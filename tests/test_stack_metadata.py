from pathlib import Path


def test_official_stack_files_exist():
    assert Path("training/README.md").exists()
    assert Path("scripts/train_smolvla.sh").exists()
    assert Path("scripts/export_openvino.py").exists()
    assert Path("configs/smolvla_training.yaml").exists()


def test_training_script_uses_official_base_policy():
    text = Path("scripts/train_smolvla.sh").read_text(encoding="utf-8")
    assert "lerobot-train" in text
    assert "lerobot/smolvla_base" in text
