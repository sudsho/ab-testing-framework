import pytest

from src.power import (
    proportion_sample_size,
    continuous_sample_size,
    power_for_proportion,
)


def test_proportion_sample_size_known():
    # baseline 0.10, +0.02 absolute lift, alpha=0.05, power=0.8
    # rough rule of thumb: about 3800 per arm; we just check the order of magnitude
    n = proportion_sample_size(0.10, 0.02, power=0.8, alpha=0.05)
    assert 2000 < n < 5500


def test_proportion_sample_size_higher_power_needs_more():
    n_low = proportion_sample_size(0.10, 0.02, power=0.7, alpha=0.05)
    n_high = proportion_sample_size(0.10, 0.02, power=0.9, alpha=0.05)
    assert n_high > n_low


def test_continuous_sample_size_units():
    # MDE of 1, sd of 5, alpha=0.05, power=0.8 ~ 393 per arm
    n = continuous_sample_size(mde=1.0, sd=5.0, power=0.8, alpha=0.05)
    assert 380 < n < 410


def test_continuous_invalid():
    with pytest.raises(ValueError):
        continuous_sample_size(mde=0.5, sd=0.0)
    with pytest.raises(ValueError):
        continuous_sample_size(mde=0.0, sd=1.0)


def test_invalid_baseline():
    with pytest.raises(ValueError):
        proportion_sample_size(0.0, 0.02)
    with pytest.raises(ValueError):
        proportion_sample_size(1.0, 0.02)


def test_invalid_mde():
    with pytest.raises(ValueError):
        proportion_sample_size(0.5, 0.0)


def test_power_func_increases_with_n():
    p1 = power_for_proportion(0.1, 0.02, n_per_arm=2000)
    p2 = power_for_proportion(0.1, 0.02, n_per_arm=5000)
    assert p2 > p1
