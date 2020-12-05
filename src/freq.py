"""
Frequentist hypothesis tests for A/B testing.

z-test for proportions, Welch's t-test for continuous metrics (revenue, time on page,
etc.). chi-square added separately.
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


def welch_ttest(values_a, values_b, alternative="two-sided"):
    """Welch's t-test (unequal variances) for continuous metrics.

    Useful for revenue per user, session duration, etc., where the per-arm
    variance is rarely equal.  Returns a TestResult.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each arm needs at least 2 observations")

    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    se = math.sqrt(var_a / a.size + var_b / b.size)
    if se == 0.0:
        t = 0.0
        df = a.size + b.size - 2
    else:
        t = (mean_b - mean_a) / se
        # Welch-Satterthwaite df
        num = (var_a / a.size + var_b / b.size) ** 2
        den = (var_a ** 2) / ((a.size ** 2) * (a.size - 1)) + \
              (var_b ** 2) / ((b.size ** 2) * (b.size - 1))
        df = num / den if den > 0 else (a.size + b.size - 2)

    if alternative == "two-sided":
        p = 2.0 * (1.0 - stats.t.cdf(abs(t), df))
    elif alternative == "larger":
        p = 1.0 - stats.t.cdf(t, df)
    elif alternative == "smaller":
        p = stats.t.cdf(t, df)
    else:
        raise ValueError("unknown alternative: %s" % alternative)

    return TestResult(
        statistic=t,
        p_value=p,
        lift=mean_b - mean_a,
        ci_low=None,
        ci_high=None,
        method="welch t",
    )
