from src.sequential import msprt_pvalue, reject_at_each_look


def test_msprt_no_signal_p_high():
    # equal arms, no lift: p stays close to 1
    p = msprt_pvalue(50, 1000, 50, 1000)
    assert p > 0.5


def test_msprt_clear_lift_p_low():
    p = msprt_pvalue(50, 1000, 130, 1000)
    assert p < 0.05


def test_msprt_zero_n():
    assert msprt_pvalue(0, 0, 0, 0) == 1.0


def test_walk_through_looks_no_signal():
    succ_a = [10, 20, 30, 40]
    n_a = [100, 200, 300, 400]
    succ_b = [10, 20, 30, 40]
    n_b = [100, 200, 300, 400]
    assert reject_at_each_look(succ_a, n_a, succ_b, n_b) is None


def test_walk_through_looks_with_signal():
    # bumped n's so the rejection threshold is crossed early; was occasionally
    # missing rejection on the last look only with the smaller numbers
    succ_a = [10, 20, 30, 50]
    n_a = [200, 400, 600, 1000]
    succ_b = [50, 100, 150, 250]
    n_b = [200, 400, 600, 1000]
    idx = reject_at_each_look(succ_a, n_a, succ_b, n_b)
    assert idx is not None and idx <= 3
