import pytest

from src.bayes import (
    beta_posterior,
    sample_posterior,
    prob_b_beats_a,
    credible_interval,
    expected_loss,
)


def test_beta_posterior_basic():
    a, b = beta_posterior(10, 100, prior_alpha=1.0, prior_beta=1.0)
    assert a == 11
    assert b == 91


def test_beta_posterior_invalid():
    with pytest.raises(ValueError):
        beta_posterior(50, 10)


def test_prob_b_beats_a_with_lift():
    p = prob_b_beats_a(50, 1000, 100, 1000, n_samples=20000, seed=0)
    assert p > 0.99


def test_prob_b_beats_a_equal():
    p = prob_b_beats_a(100, 1000, 100, 1000, n_samples=20000, seed=0)
    # should hover near 0.5
    assert 0.4 < p < 0.6


def test_credible_interval_bounds():
    lo, hi = credible_interval(50, 1000)
    assert 0.0 < lo < hi < 1.0


def test_expected_loss_signs():
    loss_a, loss_b = expected_loss(50, 1000, 100, 1000, n_samples=20000, seed=0)
    # if B is clearly winning, the loss of choosing A is large, of choosing B small
    assert loss_a > loss_b
