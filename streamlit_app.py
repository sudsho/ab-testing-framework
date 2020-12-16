"""Streamlit UI for the A/B testing toolkit.

Three tabs: Frequentist, Bayesian, Power Calculator.

Run with:  streamlit run streamlit_app.py
"""
import streamlit as st


st.set_page_config(page_title="A/B testing toolkit", layout="wide")
st.title("A/B testing toolkit")
st.caption("frequentist + bayesian; upload a CSV with a variant column")

# Streamlit's tabs API came in 0.84; in 0.71 we fake tabs with a sidebar radio.
section = st.sidebar.radio(
    "Section",
    options=["Frequentist", "Bayesian", "Power Calculator"],
    index=0,
)

if section == "Frequentist":
    st.header("Frequentist tests")
    st.write("Coming soon: upload a CSV, pick a metric column, see z-test / t-test / chi-square.")
elif section == "Bayesian":
    st.header("Bayesian inference")
    st.write("Coming soon: Beta-Binomial conjugate update for binary metrics.")
else:
    st.header("Power calculator")
    st.write("Coming soon: pick baseline, MDE, alpha, power; get per-arm sample size.")
