import pytest

from neurotune.benchmark.timing import summarize


def test_summary_statistics():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s.median_ms == 3.0 and s.min_ms == 1.0 and s.n == 5
    assert s.p25_ms == 2.0 and s.p75_ms == 4.0
    assert s.statistic("median") == 3.0


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [-1.0]])
def test_rejects_bad_samples(bad):
    with pytest.raises(ValueError):
        summarize(bad)
