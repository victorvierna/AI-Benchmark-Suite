from benchkit.config import load_models, load_pricing, load_suite


def test_load_suite():
    suite = load_suite("tests/fixtures/simple_suite.yaml")
    assert suite.id == "simple_suite"
    assert suite.request.user_template


def test_load_models():
    models = load_models("tests/fixtures/models.yaml")
    assert len(models.models) == 1
    assert models.models[0].name == "gpt-test"


def test_load_pricing():
    pricing = load_pricing("tests/fixtures/pricing.yaml")
    assert "openai" in pricing.providers
