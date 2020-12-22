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


def _normal_pvalue(z, alternative):
    if alternative == "two-sided":
        return 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    if alternative == "larger":
        return 1.0 - stats.norm.cdf(z)
    if alternative == "smaller":
        return stats.norm.cdf(z)
    raise ValueError("unknown alternative: %s" % alternative)


def _t_pvalue(t, df, alternative):
    if alternative == "two-sided":
        return 2.0 * (1.0 - stats.t.cdf(abs(t), df))
    if alternative == "larger":
        return 1.0 - stats.t.cdf(t, df)
    if alternative == "smaller":
        return stats.t.cdf(t, df)
    raise ValueError("unknown alternative: %s" % alternative)


def two_proportion_ztest(success_a, n_a, success_b, n_b, alpha=0.05,
                         alternative="two-sided", continuity_correction=False):
    """Pooled two-proportion z-test.

    Returns TestResult including a Wald CI on the difference (p_b - p_a).
    ``success_a`` is the control side, ``success_b`` is the treatment side.
    ``alternative`` is one of {"two-sided", "larger", "smaller"}.

    If ``continuity_correction`` is True, applies a Yates-style 0.5/n correction.
    Useful when sample sizes are small.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("sample sizes must be positive")

    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)

    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    diff_observed = p_b - p_a
    if continuity_correction and se_pool > 0:
        cc = 0.5 * (1.0 / n_a + 1.0 / n_b)
        # shrink the magnitude of the observed diff by cc, but never past zero
        diff_for_z = math.copysign(max(abs(diff_observed) - cc, 0.0), diff_observed)
    else:
        diff_for_z = diff_observed

    if se_pool == 0.0:
        z = 0.0
    else:
        z = diff_for_z / se_pool

    p = _normal_pvalue(z, alternative)

    # CI on the unpooled difference uses the unpooled SE
    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    diff = p_b - p_a

    return TestResult(
        statistic=z,
        p_value=p,
        lift=diff,
        ci_low=diff - z_crit * se_diff,
        ci_high=diff + z_crit * se_diff,
        method="two-proportion z",
    )


def welch_ttest(values_a, values_b, alpha=0.05, alternative="two-sided"):
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

    p = _t_pvalue(t, df, alternative)

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    diff = mean_b - mean_a
    return TestResult(
        statistic=t,
        p_value=p,
        lift=diff,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        method="welch t",
    )


def chi_square_test(table):
    """Pearson chi-square test of independence.

    ``table`` is a 2D list/array of counts, e.g. [[a_succ, a_fail], [b_succ, b_fail]]
    for a 2x2 case, or larger if more than two variants.  Returns a TestResult
    with ``lift=None`` since lift isn't meaningful past 2x2.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("contingency table must be 2D with at least 2x2")
    if (arr < 0).any():
        raise ValueError("counts cannot be negative")

    chi2, p, dof, _ = stats.chi2_contingency(arr, correction=False)

    lift = None
    if arr.shape == (2, 2):
        n_a = arr[0].sum()
        n_b = arr[1].sum()
        if n_a > 0 and n_b > 0:
            lift = (arr[1, 0] / n_b) - (arr[0, 0] / n_a)

    return TestResult(
        statistic=chi2,
        p_value=p,
        lift=lift,
        ci_low=None,
        ci_high=None,
        method="chi-square (df=%d)" % dof,
    )
