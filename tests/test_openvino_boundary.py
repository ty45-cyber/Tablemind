from tablemind.inference.router import PolicyRouter

def test_router_always_has_reference_backend():
    status=PolicyRouter().status()
    assert any(s.backend=='reference' and s.available for s in status)
