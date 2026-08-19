"""
Offline smoke test for the A/B testing framework.

Runs the full analysis path on synthetic experiment data with a KNOWN effect,
and on a null A/A experiment, using only numpy / scipy / pandas (no keys, no
downloads, no cluster). Proves the toolkit:

  1. plans an experiment (sample size + achieved power),
  2. detects a real effect via the frequentist two-proportion z-test
     (p-value + confidence interval + accept/reject),
  3. agrees via the sequential mSPRT and the Bayesian Beta-Binomial views,
  4. does NOT flag a null (A/A) experiment as significant,
  5. handles a continuous-metric (revenue) test with Welch's t-test.

The pymc3 Normal-Normal model is heavy and optional; it is only exercised when
RUN_PYMC_TESTS=1 is set, and its absence never fails this smoke.

Run:  python scripts/smoke.py   (or:  make smoke)
Exit code 0 means every assertion held.
"""
import os
import sys

# make "src" importable when run from the repo root or elsewhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.sim import generate_binary, generate_revenue
from src.freq import two_proportion_ztest, welch_ttest, chi_square_test
from src.power import (
    proportion_sample_size,
    continuous_sample_size,
    power_for_proportion,
)
from src.sequential import msprt_pvalue
from src.bayes import prob_b_beats_a, credible_interval, expected_loss

ALPHA = 0.05
SEED = 7


def _counts(df, metric_col="converted"):
    g = df.groupby("variant")[metric_col]
    return (int(g.sum()["A"]), int(g.count()["A"]),
            int(g.sum()["B"]), int(g.count()["B"]))


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main():
    checks = []

    def check(label, ok):
        checks.append((label, bool(ok)))
        print("  [%s] %s" % ("PASS" if ok else "FAIL", label))

    # ------------------------------------------------------------------
    section("1. Experiment planning (sample size + power)")
    # baseline 10% CTR, want to detect a +2pp absolute lift at 80% power
    baseline, mde = 0.10, 0.02
    n_req = proportion_sample_size(baseline, mde, power=0.8, alpha=ALPHA)
    ach_power = power_for_proportion(baseline, mde, n_per_arm=n_req, alpha=ALPHA)
    print("  baseline=%.0f%%  MDE=+%.0fpp  ->  need n=%d per arm  "
          "(achieved power=%.3f)" % (baseline * 100, mde * 100, n_req, ach_power))
    n_rev = continuous_sample_size(mde=1.0, sd=5.0, power=0.8, alpha=ALPHA)
    print("  continuous metric (MDE=1.0, sd=5.0)  ->  need n=%d per arm" % n_rev)
    check("sample size is a sensible positive int", 2000 < n_req < 5500)
    check("achieved power at required n is ~0.8", 0.78 <= ach_power <= 0.86)
    check("continuous sample size ~393", 380 < n_rev < 410)

    # ------------------------------------------------------------------
    section("2. Real effect: B genuinely beats A (10.0% vs 13.0% CTR)")
    df = generate_binary(n_per_arm=12000, p_a=0.10, p_b=0.13, seed=SEED)
    sa, na, sb, nb = _counts(df)
    print("  observed: A = %d/%d (%.2f%%)   B = %d/%d (%.2f%%)"
          % (sa, na, 100 * sa / na, sb, nb, 100 * sb / nb))

    res = two_proportion_ztest(sa, na, sb, nb, alpha=ALPHA)
    print("  z-test:   z=%.3f  p=%.2e  lift=%+.4f  95%% CI=[%.4f, %.4f]"
          % (res.statistic, res.p_value, res.lift, res.ci_low, res.ci_high))
    detected = res.p_value < ALPHA
    print("  decision: %s (p %s alpha=%.2f)"
          % ("REJECT null - effect detected" if detected else "fail to reject",
             "<" if detected else ">=", ALPHA))
    check("frequentist detects the real effect", detected)
    check("lift is positive", res.lift > 0)
    check("CI excludes zero for a real effect", res.ci_low > 0)

    p_seq = msprt_pvalue(sa, na, sb, nb)
    print("  mSPRT:    always-valid p=%.2e  ->  %s"
          % (p_seq, "reject" if p_seq < ALPHA else "continue"))
    check("sequential mSPRT also flags the effect", p_seq < ALPHA)

    p_bb = prob_b_beats_a(sa, na, sb, nb, n_samples=40000, seed=SEED)
    lo, hi = credible_interval(sb, nb)
    loss_a, loss_b = expected_loss(sa, na, sb, nb, n_samples=40000, seed=SEED)
    print("  Bayesian: P(B>A)=%.4f   B 95%% credible=[%.4f, %.4f]"
          % (p_bb, lo, hi))
    print("            expected loss: choose A=%.5f  choose B=%.5f" % (loss_a, loss_b))
    check("Bayesian P(B>A) is decisive (>0.99)", p_bb > 0.99)
    check("expected loss of the winner (B) is tiny", loss_b < loss_a and loss_b < 1e-3)

    # ------------------------------------------------------------------
    section("3. Null A/A experiment: no real effect (10.0% vs 10.0%)")
    df0 = generate_binary(n_per_arm=12000, p_a=0.10, p_b=0.10, seed=SEED)
    sa0, na0, sb0, nb0 = _counts(df0)
    print("  observed: A = %d/%d (%.2f%%)   B = %d/%d (%.2f%%)"
          % (sa0, na0, 100 * sa0 / na0, sb0, nb0, 100 * sb0 / nb0))
    res0 = two_proportion_ztest(sa0, na0, sb0, nb0, alpha=ALPHA)
    print("  z-test:   z=%.3f  p=%.4f  95%% CI=[%.4f, %.4f]"
          % (res0.statistic, res0.p_value, res0.ci_low, res0.ci_high))
    flagged = res0.p_value < ALPHA
    print("  decision: %s"
          % ("FALSE POSITIVE - flagged null!" if flagged else "correctly NOT significant"))
    check("frequentist does NOT flag the A/A null", not flagged)
    check("A/A confidence interval straddles zero", res0.ci_low < 0 < res0.ci_high)

    p_seq0 = msprt_pvalue(sa0, na0, sb0, nb0)
    print("  mSPRT:    always-valid p=%.4f  ->  %s"
          % (p_seq0, "reject" if p_seq0 < ALPHA else "continue (correct)"))
    check("sequential mSPRT does NOT flag the null", p_seq0 >= ALPHA)

    # ------------------------------------------------------------------
    section("4. Continuous metric: revenue per user (Welch t-test)")
    dfr = generate_revenue(n_per_arm=4000, mean_a=12.0, mean_b=12.8, sd=8.0, seed=SEED)
    a = dfr[dfr.variant == "A"]["revenue"].values
    b = dfr[dfr.variant == "B"]["revenue"].values
    rr = welch_ttest(a, b, alpha=ALPHA)
    print("  means:    A=%.3f   B=%.3f   diff=%+.3f" % (a.mean(), b.mean(), rr.lift))
    print("  t-test:   t=%.3f  p=%.2e  95%% CI=[%.3f, %.3f]"
          % (rr.statistic, rr.p_value, rr.ci_low, rr.ci_high))
    check("Welch t-test detects the revenue lift", rr.p_value < ALPHA and rr.lift > 0)

    # chi-square cross-check on the real-effect 2x2 table
    chi = chi_square_test([[na - sa, sa], [nb - sb, sb]])
    print("  chi-sq:   stat=%.3f  p=%.2e  (%s)" % (chi.statistic, chi.p_value, chi.method))
    check("chi-square agrees the real effect is significant", chi.p_value < ALPHA)

    # ------------------------------------------------------------------
    section("5. Optional heavy dep: pymc3 Normal-Normal model")
    if os.getenv("RUN_PYMC_TESTS") == "1":
        try:
            from src.bayes import normal_normal_model
            _, summ = normal_normal_model(a, b, draws=300, tune=300, chains=1, seed=0)
            print("  pymc3 mean_diff=%.3f  P(B>A)=%.3f"
                  % (summ["mean_diff"], summ["prob_b_beats_a"]))
            check("pymc3 model agrees B > A", summ["prob_b_beats_a"] > 0.7)
        except Exception as exc:  # pragma: no cover - optional path
            print("  pymc3 requested but unavailable (%s); skipping." % type(exc).__name__)
    else:
        print("  skipped (set RUN_PYMC_TESTS=1 to exercise the pymc3 path).")
        print("  core smoke needs only numpy / scipy / pandas.")

    # ------------------------------------------------------------------
    section("SUMMARY")
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    for label, ok in checks:
        print("  %-52s %s" % (label, "ok" if ok else "FAILED"))
    print("\n  %d/%d checks passed." % (passed, total))
    if passed != total:
        print("  SMOKE FAILED")
        return 1
    print("  SMOKE OK - framework detects real effects and ignores A/A nulls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
