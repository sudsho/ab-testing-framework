"""
Sample-size and power calculations.

Sits on top of statsmodels' power module so we don't reinvent every formula,
but provides convenience wrappers tuned to the way analysts actually pose the
question ("how many users do I need per arm to detect a 2 percentage-point
lift on a 10% baseline at 80% power?").
"""
import math

from scipy import stats


def proportion_sample_size(baseline_rate, mde, power=0.8, alpha=0.05,
                           alternative="two-sided"):
    """Per-arm sample size to detect an absolute MDE on a baseline conversion rate.

    Uses the standard normal approximation; equivalent to statsmodels'
    NormalIndPower for proportions but with both arms set to equal size.
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be in (0,1)")
    if mde == 0:
        raise ValueError("mde must be non-zero")

    p1 = baseline_rate
    p2 = baseline_rate + mde
    if not (0 < p2 < 1):
        raise ValueError("baseline_rate + mde must be in (0,1)")

    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha)
    z_beta = stats.norm.ppf(power)

    p_bar = (p1 + p2) / 2.0
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numerator / (mde ** 2)
    return int(math.ceil(n))


def continuous_sample_size(mde, sd, power=0.8, alpha=0.05, alternative="two-sided"):
    """Per-arm sample size for a two-sample mean test (assuming equal sd).

    ``mde`` and ``sd`` are on the same units as the metric.  Use this for
    revenue per user, session length, etc.
    """
    if sd <= 0:
        raise ValueError("sd must be positive")
    if mde == 0:
        raise ValueError("mde must be non-zero")

    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha)
    z_beta = stats.norm.ppf(power)

    n = 2.0 * ((z_alpha + z_beta) ** 2) * (sd ** 2) / (mde ** 2)
    return int(math.ceil(n))


def power_for_proportion(baseline_rate, mde, n_per_arm, alpha=0.05,
                         alternative="two-sided"):
    """Power achieved with given per-arm sample size, given baseline + MDE."""
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be in (0,1)")
    p1 = baseline_rate
    p2 = baseline_rate + mde
    if not (0 < p2 < 1):
        raise ValueError("baseline + mde must be in (0,1)")

    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha)

    p_bar = (p1 + p2) / 2.0
    se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
    z_required = (z_alpha * se_null - abs(mde)) / se_alt
    return float(stats.norm.cdf(-z_required))
