# ab-testing-framework

A small toolkit for running A/B tests on product/data team experiments. Frequentist and Bayesian methods side-by-side, plus a Streamlit UI for the analysts who would rather not open a notebook.

## What's in here

- frequentist tests: z-test for proportions, t-test for continuous, chi-square for contingency tables
- Bayesian: Beta-Binomial conjugate update for conversion-rate tests, pymc3 model for non-conjugate cases (Normal-Normal for revenue)
- sample size + power calculator
- streamlit UI with three tabs: Frequentist, Bayesian, Power Calculator
- upload a CSV with a `variant` column, see lift estimates, p-values, posteriors and "chance to beat control"

## Layout

```
src/        core stats functions
configs/    default knobs
tests/      pytest suite
notebooks/  example walkthrough
streamlit_app.py
```

## Running

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

More details below once it's filled in.
