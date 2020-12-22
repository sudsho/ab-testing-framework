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
    assert a == pytest.approx(11.0)
    assert b == pytest.approx(91.0)


def test_beta_posterior_jeffreys():
    a, b = beta_posterior(10, 100, prior_alpha=0.5, prior_beta=0.5)
    assert a == pytest.approx(10.5)
    assert b == pytest.approx(90.5)


def test_beta_posterior_invalid():
    with pytest.raises(ValueError):
        beta_posterior(50, 10)


def test_prob_b_beats_a_with_lift():
    p = prob_b_beats_a(50, 1000, 100, 1000, n_samples=20000, seed=0)
    assert p > 0.99


def test_prob_b_beats_a_equal():
    # bigger n + tighter band; was occasionally tripping at 0.4
    p = prob_b_beats_a(100, 1000, 100, 1000, n_samples=80000, seed=42)
    assert 0.45 < p < 0.55


def test_credible_interval_bounds():
    lo, hi = credible_interval(50, 1000)
    assert 0.0 < lo < hi < 1.0


def test_expected_loss_signs():
    loss_a, loss_b = expected_loss(50, 1000, 100, 1000, n_samples=20000, seed=0)
    # if B is clearly winning, the loss of choosing A is large, of choosing B small
    assert loss_a > loss_b
