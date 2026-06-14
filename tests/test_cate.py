"""Tests that prove the CATE methods recover known heterogeneity and that the
subgroup-fishing guard actually controls false discoveries.

Ground truth: Y = 3 + 0.5*X1 - X2 + tau(X)*T + noise, with a HETEROGENEOUS effect
tau(X) = 1 + 2*X1. So the true ATE = 1.0 and the true CATE = 1 + 2*X1 — both known, so
the estimates are checkable, not just plausible.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ab_cate import (  # noqa: E402
    lin_estimator, s_learner, t_learner, x_learner, honest_cate,
    subgroup_fishing_guard, cate_summary,
)

A0, A1, B1, B2 = 1.0, 2.0, 0.5, -1.0  # tau = A0 + A1*X1 ; baseline slopes B1,B2


def make(n=8000, seed=0, p_treat=0.5):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    T = (rng.random(n) < p_treat).astype(int)
    tau = A0 + A1 * X1
    Y = 3 + B1 * X1 + B2 * X2 + tau * T + rng.normal(0, 1, n)
    return np.column_stack([X1, X2]), T, Y, tau


X, T, Y, TAU = make()


def test_lin_estimator_recovers_ate():
    r = lin_estimator(X, T, Y)
    assert r["ci"][0] <= A0 <= r["ci"][1]              # CI covers true ATE = 1.0
    assert abs(r["ate"] - A0) < 0.1


def test_lin_estimator_reduces_variance():
    r = lin_estimator(X, T, Y)
    assert r["se"] < r["naive_se"] and r["variance_reduction"] > 0.1


def test_t_learner_recovers_cate():
    cate = t_learner(X, T, Y)["cate"](X)
    assert np.corrcoef(cate, TAU)[0, 1] > 0.9
    assert np.sqrt(np.mean((cate - TAU) ** 2)) < 0.3


def test_s_learner_captures_heterogeneity():
    cate = s_learner(X, T, Y)["cate"](X)
    assert np.corrcoef(cate, TAU)[0, 1] > 0.9


def test_x_learner_robust_under_imbalance():
    Xi, Ti, Yi, taui = make(n=8000, seed=1, p_treat=0.2)  # only 20% treated
    fit = x_learner(Xi, Ti, Yi)
    cate = fit["cate"](Xi)
    assert abs(fit["ate"] - A0) < 0.15
    assert np.corrcoef(cate, taui)[0, 1] > 0.85


def test_honest_cate_uses_disjoint_splits():
    h = honest_cate(X, T, Y, learner="x", test_frac=0.4, seed=3)
    assert set(h["train_idx"]).isdisjoint(set(h["test_idx"]))
    assert abs(h["ate_holdout"] - A0) < 0.15


def test_cate_summary_shows_spread():
    s = cate_summary(t_learner(X, T, Y), X)
    assert s["p90"] - s["p10"] > 1.0     # tau spans ~[1-2sd, 1+2sd]*2, real spread


def test_fishing_guard_controls_false_discoveries():
    # HOMOGENEOUS truth: constant effect, no modifier -> fished slices are all noise.
    rng = np.random.default_rng(5)
    n = 8000
    Xn = rng.normal(0, 1, (n, 4))
    Tn = (rng.random(n) < 0.5).astype(int)
    Yn = 3 + Xn @ [0.5, -0.3, 0.2, 0.1] + 1.0 * Tn + rng.normal(0, 1, n)  # tau=1 everywhere
    subs = {f"rand_{j}": (Xn[:, j % 4] > rng.normal(0, 0.5)) for j in range(20)}
    g = subgroup_fishing_guard(Tn, Yn, subs)
    assert g["n_tested"] == 20
    assert g["bh_significant"] <= 1            # BH controls the fishing


def test_fishing_guard_finds_real_modifier():
    # X1 is a real effect modifier (tau = 1 + 2*X1) -> X1>0 differs from the rest.
    subs = {"X1_positive": X[:, 0] > 0, "X1_negative": X[:, 0] <= 0,
            "X2_positive": X[:, 1] > 0}   # X2 is NOT a modifier
    g = subgroup_fishing_guard(T, Y, subs)
    real = {r["subgroup"] for r in g["subgroups"] if r["bh_real"]}
    assert "X1_positive" in real and "X2_positive" not in real


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
