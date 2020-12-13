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
