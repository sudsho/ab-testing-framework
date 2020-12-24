"""End-to-end-ish: simulate data, run a frequentist test, check sanity."""
from src.sim import generate_binary, generate_revenue
from src.freq import two_proportion_ztest, welch_ttest


def test_sim_to_ztest_no_lift():
    df = generate_binary(n_per_arm=2000, p_a=0.10, p_b=0.10, seed=11)
    g = df.groupby("variant")["converted"]
    res = two_proportion_ztest(int(g.sum()["A"]), int(g.count()["A"]),
                               int(g.sum()["B"]), int(g.count()["B"]))
    # no real lift, so p-value should not be tiny
    assert res.p_value > 0.05


def test_sim_to_ztest_with_lift():
    df = generate_binary(n_per_arm=10000, p_a=0.10, p_b=0.13, seed=11)
    g = df.groupby("variant")["converted"]
    res = two_proportion_ztest(int(g.sum()["A"]), int(g.count()["A"]),
                               int(g.sum()["B"]), int(g.count()["B"]))
    assert res.p_value < 0.001
    assert res.lift > 0


def test_sim_to_ttest_revenue():
    df = generate_revenue(n_per_arm=2000, mean_a=10.0, mean_b=11.0, sd=5.0, seed=11)
    a = df[df.variant == "A"]["revenue"].values
    b = df[df.variant == "B"]["revenue"].values
    res = welch_ttest(a, b)
    assert res.p_value < 0.001
    assert res.lift > 0
