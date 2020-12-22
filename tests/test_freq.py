import math

import numpy as np
import pytest

from src.freq import two_proportion_ztest, welch_ttest, chi_square_test


def test_ztest_no_difference():
    res = two_proportion_ztest(50, 1000, 50, 1000)
    assert abs(res.statistic) < 1e-9
    assert res.p_value > 0.99


def test_ztest_clear_lift():
    res = two_proportion_ztest(50, 1000, 100, 1000)
    assert res.p_value < 0.001
    assert res.lift > 0


def test_ttest_basic():
    # bigger samples + a fixed seed; the older version was flaky on smaller n
    rng = np.random.RandomState(123)
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(0.25, 1, size=2000)
    res = welch_ttest(a, b)
    assert res.p_value < 0.001
    assert res.lift > 0


def test_ttest_no_signal():
    rng = np.random.RandomState(123)
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(0, 1, size=2000)
    res = welch_ttest(a, b)
    assert res.p_value > 0.05


def test_chi_square_2x2():
    res = chi_square_test([[50, 950], [100, 900]])
    assert res.p_value < 0.001
    assert res.lift is not None
