"""
Synthetic data generators.  Handy for tests, demos and the streamlit upload tab
when the user just wants something to look at.
"""
import numpy as np
import pandas as pd


def generate_binary(n_per_arm, p_a=0.10, p_b=0.12, seed=42, variant_col="variant",
                    metric_col="converted"):
    """Two-arm binary outcome (e.g. CTR test).

    Returns a tidy DataFrame with one row per user.
    """
    rng = np.random.RandomState(seed)
    a = rng.binomial(1, p_a, size=n_per_arm)
    b = rng.binomial(1, p_b, size=n_per_arm)
    df = pd.DataFrame({
        variant_col: ["A"] * n_per_arm + ["B"] * n_per_arm,
        metric_col: np.concatenate([a, b]),
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_revenue(n_per_arm, mean_a=12.0, mean_b=12.6, sd=8.0, seed=42,
                     variant_col="variant", metric_col="revenue"):
    """Two-arm continuous outcome with optional left-clipping at 0.

    A rough stand-in for revenue per user; we generate from a normal and
    clip at 0 because revenue can't go negative.
    """
    rng = np.random.RandomState(seed)
    a = np.clip(rng.normal(mean_a, sd, size=n_per_arm), 0, None)
    b = np.clip(rng.normal(mean_b, sd, size=n_per_arm), 0, None)
    df = pd.DataFrame({
        variant_col: ["A"] * n_per_arm + ["B"] * n_per_arm,
        metric_col: np.concatenate([a, b]),
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
