from tablemind.planning.task_graph import BimanualTaskPlanner

def test_plan_has_both_arms_and_final_verify():
    p=BimanualTaskPlanner(); arms=[]
    while not p.is_complete():
        a=p.next_action(type('Obs',(object,),{'objects':[]})())
        assert a is not None
        arms.append(a.arm); p.verify_current(True)
    assert 'left' in arms and 'right' in arms and 'both' in arms
    assert len(arms)==6
