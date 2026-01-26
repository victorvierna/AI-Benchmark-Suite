import pytest

from benchkit.dataset import DatasetError, filter_cases, load_cases


def test_load_cases_ok():
    cases = load_cases("tests/fixtures/simple_cases.jsonl")
    assert len(cases) == 2


def test_filter_cases():
    cases = load_cases("tests/fixtures/simple_cases.jsonl")
    cases = filter_cases(cases, ["id:T"], None)
    assert len(cases) == 2


def test_invalid_filter():
    cases = load_cases("tests/fixtures/simple_cases.jsonl")
    with pytest.raises(DatasetError):
        filter_cases(cases, ["badfilter"], None)
