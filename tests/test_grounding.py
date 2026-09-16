from tablemind.perception.grounding import ground_instruction
from tablemind.core.models import Observation, SceneObject

def test_grounding_extracts_task_terms():
    obs=Observation(0,'x',[SceneObject('spoon','utensil',0,0,0),SceneObject('cup','drinkware',0,0,0)], [0])
    r=ground_instruction('Set the dinner table with a spoon and cup',obs)
    assert r['intent']=='set_dinner_table'
    assert 'spoons' in r['requested_categories']
    assert 'cup' in r['visible_objects']
