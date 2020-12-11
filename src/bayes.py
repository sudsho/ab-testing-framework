"""
Bayesian A/B testing.

For binary outcomes (conversion rate, CTR) we use the Beta-Binomial conjugate
update; nice closed-form posterior, easy to draw from.  Non-conjugate cases
(revenue per user with a Normal-Normal model, for instance) drop into pymc3.
"""
import numpy as np
from scipy import stats


def beta_posterior(successes, trials, prior_alpha=1.0, prior_beta=1.0):
    """Return the Beta posterior parameters (alpha, beta) given a Beta prior
    and observed successes/trials."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid successes/trials")
    return prior_alpha + successes, prior_beta + (trials - successes)


def sample_posterior(successes, trials, prior=(1.0, 1.0), n_samples=50000, seed=None):
    """Draw samples from the Beta posterior."""
    a, b = beta_posterior(successes, trials, *prior)
    rng = np.random.RandomState(seed)
    return rng.beta(a, b, size=n_samples)


def prob_b_beats_a(succ_a, n_a, succ_b, n_b,
                   prior=(1.0, 1.0), n_samples=50000, seed=None):
    """Monte Carlo estimate of P(theta_B > theta_A) using Beta posteriors.

    Closed-form options exist (Cook 2005, Miller's bbeta sums) but the MC route
    is plenty fast for our use and easier to read.
    """
    a_samples = sample_posterior(succ_a, n_a, prior, n_samples, seed)
    seed_b = None if seed is None else seed + 1
    b_samples = sample_posterior(succ_b, n_b, prior, n_samples, seed_b)
    return float((b_samples > a_samples).mean())


def credible_interval(successes, trials, prior=(1.0, 1.0), level=0.95):
    """Equal-tailed credible interval on the rate."""
    a, b = beta_posterior(successes, trials, *prior)
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return float(stats.beta.ppf(lo, a, b)), float(stats.beta.ppf(hi, a, b))


def expected_loss(succ_a, n_a, succ_b, n_b,
                  prior=(1.0, 1.0), n_samples=50000, seed=None):
    """Expected loss of choosing each variant if the other one is in fact better.

    Useful for "stop the test" rules; if expected loss of choosing B is below
    a threshold (say 0.0001) you can call B the winner with low regret.
    """
    a_samples = sample_posterior(succ_a, n_a, prior, n_samples, seed)
    seed_b = None if seed is None else seed + 1
    b_samples = sample_posterior(succ_b, n_b, prior, n_samples, seed_b)
    loss_choose_a = np.maximum(b_samples - a_samples, 0.0).mean()
    loss_choose_b = np.maximum(a_samples - b_samples, 0.0).mean()
    return float(loss_choose_a), float(loss_choose_b)


def normal_normal_model(values_a, values_b, draws=2000, tune=1000, chains=2, seed=42):
    """pymc3 model for two-arm continuous metric (e.g. revenue per user).

    Uses weakly-informative priors centred on the pooled mean.  Returns the
    InferenceData object plus a summary dict with mean, 95% HDI, and
    P(mu_b > mu_a).

    The import of pymc3 is local because it is a heavy dependency and
    we don't want to pay the cost on every freq.py import.
    """
    import pymc3 as pm
    import arviz as az

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    pooled_mean = np.concatenate([a, b]).mean()
    pooled_sd = np.concatenate([a, b]).std(ddof=1)

    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=pooled_mean, sigma=pooled_sd * 5)
        mu_b = pm.Normal("mu_b", mu=pooled_mean, sigma=pooled_sd * 5)
        sigma_a = pm.HalfNormal("sigma_a", sigma=pooled_sd * 2)
        sigma_b = pm.HalfNormal("sigma_b", sigma=pooled_sd * 2)
        pm.Normal("obs_a", mu=mu_a, sigma=sigma_a, observed=a)
        pm.Normal("obs_b", mu=mu_b, sigma=sigma_b, observed=b)
        pm.Deterministic("diff", mu_b - mu_a)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=False,
        )

    diff_samples = trace.posterior["diff"].values.flatten()
    hdi = az.hdi(diff_samples, hdi_prob=0.95)
    summary = {
        "mean_diff": float(diff_samples.mean()),
        "hdi_low": float(hdi[0]),
        "hdi_high": float(hdi[1]),
        "prob_b_beats_a": float((diff_samples > 0).mean()),
    }
    return trace, summary
