import pytest

from benchkit.template import render_with_case


def test_render_with_case():
    case = {"input": {"text": "hi"}}
    out = render_with_case("Say {{ input.text }}", case)
    assert out == "Say hi"


def test_missing_variable():
    case = {"input": {}}
    with pytest.raises(Exception):
        render_with_case("Say {{ input.text }}", case)
