def test_cli_importable():
    from tablemind import cli
    assert callable(cli.main)
