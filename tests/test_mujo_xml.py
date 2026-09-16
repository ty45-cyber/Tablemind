from pathlib import Path
import xml.etree.ElementTree as ET

def test_mujoco_scene_is_valid_xml():
    root=ET.parse(Path('mujoco/tablemind_scene.xml')).getroot()
    assert root.tag=='mujoco'
    names={b.attrib['name'] for b in root.findall('.//body') if 'name' in b.attrib}
    assert 'so101_left_base' in names and 'so101_right_base' in names


def test_real_mujoco_loader_has_clear_optional_boundary():
    from tablemind.execution.simulation import load_mujoco_model
    try:
        model = load_mujoco_model()
    except RuntimeError as exc:
        assert ("MuJoCo is not installed" in str(exc)) or ("does not expose the expected MjModel API" in str(exc))
    else:
        assert model.nq >= 4
        assert model.nv >= 4
