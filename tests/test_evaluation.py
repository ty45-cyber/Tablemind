from tablemind.evaluation.evaluator import evaluate

def test_ten_seed_reference_evaluation_is_reproducible():
    r=evaluate(10)
    assert r['episodes']==10
    assert r['success_rate']==1.0
    assert r['collision_free_rate']==1.0
