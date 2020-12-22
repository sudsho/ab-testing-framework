"""
Sequential testing helpers.

The standard z-test isn't safe under "peeking"; if you check the p-value every
day you inflate the false-positive rate.  Two safer-ish ways to do it:

1. Always-valid p-values via mSPRT (Johari, Pekelis, Walsh 2017).
2. Bayesian decision rules using expected_loss (see bayes.py); stop when
   expected loss of a decision drops below a threshold.

This module gives a small mSPRT helper for the proportion case.
"""
import math

import numpy as np


def msprt_pvalue(succ_a, n_a, succ_b, n_b, tau=1.0):
    """Always-valid p-value from a mixture sequential probability ratio test.

    Uses a normal mixing distribution on the effect size (variance ``tau``).
    Returns a p-value that can be checked at every interim look without
    inflating the type-I error.

    This is a back-of-envelope implementation; for production you'd want the
    Johari et al. paper's exact form.  Good enough for an internal tool.
    """
    if n_a <= 0 or n_b <= 0:
        return 1.0

    p_a = succ_a / n_a
    p_b = succ_b / n_b
    p_pool = (succ_a + succ_b) / (n_a + n_b)
    if p_pool <= 0 or p_pool >= 1:
        return 1.0

    # observed Z
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 1.0
    z = (p_b - p_a) / se

    # effective sample size (harmonic mean style)
    n_eff = 1.0 / (1.0 / n_a + 1.0 / n_b)
    # mixture likelihood ratio (Robbins 1970 form, simplified)
    factor = math.sqrt(1.0 + n_eff * tau)
    exponent = (n_eff * tau * z * z) / (2.0 * (1.0 + n_eff * tau))
    # guard against overflow on huge z
    try:
        lr = math.exp(exponent) / factor
    except OverflowError:
        return 0.0

    if lr <= 0.0:
        return 1.0
    return min(1.0, 1.0 / lr)


def reject_at_each_look(succ_a_seq, n_a_seq, succ_b_seq, n_b_seq, alpha=0.05, tau=1.0):
    """Walk through interim looks and return the first index where mSPRT rejects.

    Inputs are aligned arrays of cumulative counts at each look.  Returns the
    look-index of rejection or None if never.
    """
    if not (len(succ_a_seq) == len(n_a_seq) == len(succ_b_seq) == len(n_b_seq)):
        raise ValueError("all sequences must be the same length")
    for i in range(len(succ_a_seq)):
        p = msprt_pvalue(succ_a_seq[i], n_a_seq[i],
                         succ_b_seq[i], n_b_seq[i], tau=tau)
        if p < alpha:
            return i
    return None
