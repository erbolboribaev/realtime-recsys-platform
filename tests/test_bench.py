import pytest

from recsys.serving.bench import percentile, summarize


def test_percentile_uses_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 100) == 100


def test_percentile_of_one_value_is_that_value():
    assert percentile([7.0], 99) == 7.0


def test_percentile_needs_data():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summary_reports_the_main_statistics():
    summary = summarize([10.0, 20.0, 30.0, 40.0])
    assert summary["n"] == 4.0
    assert summary["mean"] == 25.0
    assert summary["p50"] == 20.0
