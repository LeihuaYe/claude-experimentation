"""Tests that prove the design math is right, not just plausible.

The credibility anchor: the analytic sample sizes are checked against MONTE-CARLO
power — if sample_size() says n is enough for 80% power, a simulation at that n must
actually reject the null ~80% of the time. Plus the algebraic invariants (sample_size
<-> mde round-trip, CUPED cuts n by 1-rho^2, the cluster design effect, alpha split).
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ab_design import (  # noqa: E402
    sample_size, mde, power, power_curve, cluster_design_effect,
    allocate_alpha, ratio_variance, switchback_design, duration,
)

RNG = np.random.default_rng(7)


def test_sample_size_mde_round_trip_mean():
    n = sample_size("mean", baseline=10, mde=0.1, sd=5)["n_per_arm_raw"]
    recovered = mde("mean", baseline=10, n=n, sd=5)
    assert abs(recovered - 0.1) / 0.1 < 0.01  # inverse to <1%


def test_sample_size_mde_round_trip_proportion():
    n = sample_size("proportion", baseline=0.20, mde=0.05)["n_per_arm_raw"]
    recovered = mde("proportion", baseline=0.20, n=n)
    assert abs(recovered - 0.05) / 0.05 < 0.03


def test_empirical_power_mean():
    sd, delta, target = 1.0, 0.2, 0.80
    n = sample_size("mean", baseline=0, mde=delta, sd=sd, power=target)["n_per_arm"]
    reps, zc = 5000, stats.norm.ppf(0.975)
    c = RNG.normal(0, sd, size=(reps, n))
    t = RNG.normal(delta, sd, size=(reps, n))
    diff = t.mean(1) - c.mean(1)
    se = np.sqrt(c.var(1, ddof=1) / n + t.var(1, ddof=1) / n)
    emp = (np.abs(diff / se) > zc).mean()
    assert abs(emp - target) < 0.035, f"empirical power {emp:.3f} vs target {target}"


def test_empirical_power_proportion():
    p1, delta, target = 0.20, 0.05, 0.80
    n = sample_size("proportion", baseline=p1, mde=delta, power=target)["n_per_arm"]
    reps, zc = 4000, stats.norm.ppf(0.975)
    c = (RNG.random((reps, n)) < p1).mean(1)
    t = (RNG.random((reps, n)) < p1 + delta).mean(1)
    pbar = (c + t) / 2
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
    emp = (np.abs(t - c) / se0 > zc).mean()
    assert abs(emp - target) < 0.04, f"empirical power {emp:.3f} vs target {target}"


def test_cuped_cuts_n_by_one_minus_rho_squared():
    base = sample_size("mean", baseline=10, mde=0.1, sd=5)["n_per_arm_raw"]
    rho = 0.6
    cuped = sample_size("mean", baseline=10, mde=0.1, sd=5, rho=rho)["n_per_arm_raw"]
    assert abs(cuped / base - (1 - rho ** 2)) < 0.01  # n scales by 1-rho^2


def test_cluster_design_effect_inflates_n():
    m, icc = 10, 0.05
    base = sample_size("mean", baseline=0, mde=0.1, sd=1)["n_per_arm_raw"]
    clus = sample_size("mean", baseline=0, mde=0.1, sd=1, icc=icc, cluster_size=m)
    deff = cluster_design_effect(m, icc)
    assert abs(deff - 1.45) < 1e-9
    assert abs(clus["n_per_arm"] / base - deff) < 0.02
    assert clus["clusters_per_arm"] == int(np.ceil(clus["n_per_arm"] / m))


def test_more_primary_metrics_need_more_n():
    n1 = sample_size("mean", baseline=0, mde=0.1, sd=1, alpha=0.05)["n_per_arm"]
    a3 = allocate_alpha(0.05, 3)[0]
    n3 = sample_size("mean", baseline=0, mde=0.1, sd=1, alpha=a3)["n_per_arm"]
    assert n3 > n1  # spending alpha 3 ways costs sample


def test_allocate_alpha_sums_and_splits():
    a = allocate_alpha(0.05, 4)
    assert abs(sum(a) - 0.05) < 1e-12 and all(abs(x - 0.0125) < 1e-12 for x in a)
    w = allocate_alpha(0.06, 3, weights=[3, 2, 1])
    assert abs(sum(w) - 0.06) < 1e-12 and w[0] > w[1] > w[2]


def test_power_is_monotonic_in_n():
    curve = power_curve("mean", baseline=0, mde=0.1, ns=[200, 800, 3200], sd=1)
    powers = [c["power"] for c in curve]
    assert powers[0] < powers[1] < powers[2]


def test_ratio_variance_delta_method():
    n = 5000
    den = RNG.poisson(20, n) + 1.0          # impressions
    num = RNG.binomial(den.astype(int), 0.1)  # clicks ~ 10% CTR
    r = ratio_variance(num, den)
    assert 0.07 < r["ratio"] < 0.13 and r["unit_variance"] > 0


def test_switchback_balanced_and_sized():
    sb = switchback_design(48, seed=1, block=4)
    assert sb["n_periods"] == 48
    assert sb["n_control_periods"] == sb["n_treat_periods"] == 24
    assert len(sb["assignment"]) == 48


def test_duration_from_traffic():
    assert duration(10_000, 2_500) == 4.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
