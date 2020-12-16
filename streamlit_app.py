"""Streamlit UI for the A/B testing toolkit.

Three sections: Frequentist, Bayesian, Power Calculator.

Run with:  streamlit run streamlit_app.py
"""
import io

import pandas as pd
import streamlit as st

from src.freq import two_proportion_ztest, welch_ttest, chi_square_test
from src.bayes import (
    sample_posterior,
    prob_b_beats_a,
    credible_interval,
    expected_loss,
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
    df = pd.read_csv(uploaded)
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

            st.metric("P(B > A)", f"{p_b_beats_a:.3f}")
            st.write(f"95% credible interval, A: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
            st.write(f"95% credible interval, B: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
            st.write(f"Expected loss choosing A: {loss_a:.5f}")
            st.write(f"Expected loss choosing B: {loss_b:.5f}")
else:
    st.header("Power calculator")
    st.write("Coming soon.")
