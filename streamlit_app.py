"""Streamlit UI for the A/B testing toolkit.

Three sections: Frequentist, Bayesian, Power Calculator.

Run with:  streamlit run streamlit_app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.freq import two_proportion_ztest, welch_ttest, chi_square_test
from src.bayes import (
    sample_posterior,
    prob_b_beats_a,
    credible_interval,
    expected_loss,
)
from src.power import (
    proportion_sample_size,
    continuous_sample_size,
    power_for_proportion,
)


st.set_page_config(page_title="A/B testing toolkit", layout="wide")
st.title("A/B testing toolkit")
st.caption("frequentist + bayesian; upload a CSV with a variant column")

section = st.sidebar.radio(
    "Section",
    options=["Frequentist", "Bayesian", "Power Calculator"],
    index=0,
)


def _load_csv():
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with at least a variant column and a metric column.")
        return None
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return None
    if df.empty:
        st.error("CSV is empty.")
        return None
    st.write("Preview:")
    st.dataframe(df.head())
    return df


if section == "Frequentist":
    st.header("Frequentist tests")
    df = _load_csv()
    if df is not None:
        cols = list(df.columns)
        variant_col = st.selectbox("variant column", cols, index=0)
        metric_col = st.selectbox("metric column", cols, index=min(1, len(cols)-1))
        metric_kind = st.radio("metric type", ["binary (0/1)", "continuous"])
        alpha = st.number_input("alpha", min_value=0.001, max_value=0.5, value=0.05, step=0.01)

        variants = df[variant_col].unique().tolist()
        if len(variants) != 2:
            st.error("Frequentist tab needs exactly 2 variants for now.")
        else:
            arm_a = df[df[variant_col] == variants[0]][metric_col]
            arm_b = df[df[variant_col] == variants[1]][metric_col]
            if metric_kind.startswith("binary"):
                res = two_proportion_ztest(
                    int(arm_a.sum()), len(arm_a),
                    int(arm_b.sum()), len(arm_b),
                    alpha=alpha,
                )
            else:
                res = welch_ttest(arm_a.values, arm_b.values, alpha=alpha)
            st.write(res._asdict())

elif section == "Bayesian":
    st.header("Bayesian inference")
    df = _load_csv()
    if df is not None:
        cols = list(df.columns)
        variant_col = st.selectbox("variant column", cols, index=0, key="b_var")
        metric_col = st.selectbox("metric column", cols,
                                  index=min(1, len(cols)-1), key="b_met")
        prior_alpha = st.number_input("prior alpha", value=1.0, min_value=0.01)
        prior_beta = st.number_input("prior beta", value=1.0, min_value=0.01)

        variants = df[variant_col].unique().tolist()
        if len(variants) != 2:
            st.error("Bayesian tab needs exactly 2 variants for now.")
        else:
            arm_a = df[df[variant_col] == variants[0]][metric_col]
            arm_b = df[df[variant_col] == variants[1]][metric_col]
            sa, na = int(arm_a.sum()), len(arm_a)
            sb, nb = int(arm_b.sum()), len(arm_b)

            prior = (prior_alpha, prior_beta)
            p_b_beats_a = prob_b_beats_a(sa, na, sb, nb, prior=prior, seed=0)
            ci_a = credible_interval(sa, na, prior=prior)
            ci_b = credible_interval(sb, nb, prior=prior)
            loss_a, loss_b = expected_loss(sa, na, sb, nb, prior=prior, seed=0)

            col1, col2, col3 = st.beta_columns(3)
            col1.metric("Chance to beat control", f"{p_b_beats_a*100:.1f}%")
            col2.metric("Expected loss (B)", f"{loss_b:.5f}")
            col3.metric("Expected loss (A)", f"{loss_a:.5f}")
            st.write(f"95% credible interval, {variants[0]}: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
            st.write(f"95% credible interval, {variants[1]}: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
            if p_b_beats_a > 0.95:
                st.success("Strong evidence B beats A.")
            elif p_b_beats_a < 0.05:
                st.warning("Strong evidence A beats B.")
            else:
                st.info("Inconclusive yet; collect more data.")

            # Posterior overlay chart
            samp_a = sample_posterior(sa, na, prior=prior, n_samples=20000, seed=0)
            samp_b = sample_posterior(sb, nb, prior=prior, n_samples=20000, seed=1)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=samp_a, name=str(variants[0]),
                                       opacity=0.55, nbinsx=80, histnorm="probability density"))
            fig.add_trace(go.Histogram(x=samp_b, name=str(variants[1]),
                                       opacity=0.55, nbinsx=80, histnorm="probability density"))
            fig.update_layout(barmode="overlay", title="Posterior densities",
                              xaxis_title="rate", yaxis_title="density")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.header("Power calculator")
    metric_kind = st.radio("metric type", ["proportion", "continuous"])
    alpha = st.number_input("alpha", min_value=0.001, max_value=0.5,
                            value=0.05, step=0.01, key="pc_alpha")
    power = st.number_input("power (1 - beta)", min_value=0.5, max_value=0.999,
                            value=0.8, step=0.05, key="pc_power")
    if metric_kind == "proportion":
        baseline = st.number_input("baseline rate", min_value=0.001,
                                   max_value=0.999, value=0.10, step=0.01)
        mde = st.number_input("MDE (absolute)", min_value=0.0001,
                              max_value=0.5, value=0.02, step=0.005)
        if st.button("compute"):
            n = proportion_sample_size(baseline, mde, power=power, alpha=alpha)
            st.success(f"Per-arm sample size: {n}")
            achieved = power_for_proportion(baseline, mde, n, alpha=alpha)
            st.write(f"(achieved power at this n: {achieved:.3f})")
    else:
        sd = st.number_input("metric std dev", min_value=0.0001, value=8.0)
        mde = st.number_input("MDE (absolute, same units)", min_value=0.0001,
                              value=0.5, key="pc_mde_cont")
        if st.button("compute "):
            n = continuous_sample_size(mde, sd, power=power, alpha=alpha)
            st.success(f"Per-arm sample size: {n}")
