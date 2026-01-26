from benchkit.cli import main


def test_cli_validate_ok():
    code = main([
        "validate",
        "tests/fixtures/simple_suite.yaml",
        "--models",
        "tests/fixtures/models.yaml",
        "--pricing",
        "tests/fixtures/pricing.yaml",
    ])
    assert code == 0
