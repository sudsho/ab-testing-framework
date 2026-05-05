# ab-testing-framework

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=sudsho/ab-testing-framework)

A small toolkit for analysing A/B tests on product/data team experiments. Frequentist and Bayesian methods side-by-side, plus a Streamlit UI for the analysts who would rather not open a notebook.

> **Try it live**: click the badge above to one-click-deploy the Streamlit app on Streamlit Community Cloud. No API keys needed, runs entirely on the bundled toy data.

## What's in here

- **frequentist tests**: pooled two-proportion z-test (with optional continuity correction), Welch's t-test for continuous metrics, chi-square for contingency tables. All return `(statistic, p_value, lift, ci_low, ci_high, method)`.
- **Bayesian**: Beta-Binomial conjugate update for conversion-rate tests (closed-form posterior, MC sampling for `P(B > A)` and expected loss); pymc3 model for Normal-Normal (revenue / continuous metrics).
- **sample-size + power calculator**: `proportion_sample_size`, `continuous_sample_size`, and a `power_for_proportion` for "what power do I get with this n".
- **sequential testing**: a small mSPRT helper for "always-valid" p-values when you want to peek without inflating type-I (see `src/sequential.py`).
- **Streamlit UI** with three sections: Frequentist, Bayesian, Power Calculator. Upload a CSV with a `variant` column, see lift, p-values, posterior densities, "chance to beat control", and expected loss.

## Layout

```
src/
  freq.py          z-test, t-test, chi-square
  bayes.py         beta-binomial + pymc3 normal-normal
  power.py         sample size + achieved power
  sequential.py    mSPRT
  sim.py           synthetic data
streamlit_app.py
configs/default.yaml
tests/             pytest suite
notebooks/walkthrough.ipynb
Dockerfile, docker-compose.yml, .github/workflows/test.yml
```

## Running locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or with Docker:

```
docker-compose up --build
```

App is then on http://localhost:8501.

## Worked example: CTR test

Suppose the control banner shows a 5% CTR and we redesigned a treatment we hope lifts CTR by half a percentage point.

**1. Plan the experiment**

```python
from src import proportion_sample_size

n = proportion_sample_size(baseline_rate=0.05, mde=0.005,
                           power=0.8, alpha=0.05)
# per-arm n ~ 30,000
```

**2. Run it; collect the data**

Say after two weeks:

| variant | impressions | clicks |
|---|---|---|
| control | 30,000 | 1,500 |
| treatment | 30,000 | 1,680 |

**3. Frequentist**

```python
from src import two_proportion_ztest

res = two_proportion_ztest(1500, 30000, 1680, 30000)
# z ~ 3.20, p ~ 0.0014, lift = 0.006 (i.e. +0.6pp)
# 95% CI on lift: roughly [0.0023, 0.0097]
```

p-value below alpha, so reject the null at 5%.

**4. Bayesian**

```python
from src import prob_b_beats_a, expected_loss, credible_interval

p_b_better = prob_b_beats_a(1500, 30000, 1680, 30000, seed=0)
# ~ 0.999 -> very strong evidence treatment beats control

loss_a, loss_b = expected_loss(1500, 30000, 1680, 30000, seed=0)
# loss of choosing B (the winner) is essentially zero;
# loss of choosing A is around 0.006, i.e. you'd give up ~0.6pp CTR
```

The Bayesian view says the same thing as the frequentist test, with a nicer interpretation: there's a 99.9% chance B is better than A, and the regret of picking B if you turn out to be wrong is negligible.

The notebook in `notebooks/walkthrough.ipynb` runs both this CTR example and a revenue-per-user example end to end.

## Tests

```
pytest -q
```

The pymc3 smoke test is opt-in:

```
RUN_PYMC_TESTS=1 pytest -q tests/test_bayes.py
```

## License

MIT.
