"""
Frequentist hypothesis tests for A/B testing.

Right now: two-proportion z-test.
TODO: t-test for continuous, chi-square.
"""
from collections import namedtuple
import math

import numpy as np
from scipy import stats


# small container for a test result
TestResult = namedtuple(
    "TestResult",
    ["statistic", "p_value", "lift", "ci_low", "ci_high", "method"],
)


def two_proportion_ztest(success_a, n_a, success_b, n_b, alternative="two-sided"):
    """Pooled two-proportion z-test.

    Returns TestResult.  ``success_a`` is the control side, ``success_b`` is the
    treatment side. ``alternative`` is one of {"two-sided", "larger", "smaller"}.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("sample sizes must be positive")

    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)

    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        # both sides identical, no signal
        z = 0.0
    else:
        z = (p_b - p_a) / se

    if alternative == "two-sided":
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    elif alternative == "larger":
        p = 1.0 - stats.norm.cdf(z)
    elif alternative == "smaller":
        p = stats.norm.cdf(z)
    else:
        raise ValueError("unknown alternative: %s" % alternative)

    lift = p_b - p_a
    return TestResult(
        statistic=z,
        p_value=p,
        lift=lift,
        ci_low=None,   # filled in later
        ci_high=None,
        method="two-proportion z",
    )
