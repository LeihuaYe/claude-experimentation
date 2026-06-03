"""Tests that prove the readout recovers known ground truth and behaves correctly.

These double as the credibility anchor for the README: the stats are not just
plausible-looking, they're checked against a simulation with a known answer.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ab_readout import run_readout, srm_check, cuped, benjamini_hochberg  # noqa: E402
from examples.make_synthetic import make, GROUND_TRUTH, RHO  # noqa: E402

DF = make(n=30_000, seed=42)
RES = run_readout(
    DF, arm_col="group", primary="revenue",
    metrics=["revenue", "engagement_min", "conversion", "latency_ms", "support_tickets"],
    binary=["conversion"], guardrails=["latency_ms", "support_tickets"],
    covariate="pre_revenue",
)


def _metric(name):
    return next(r for r in RES["metrics"] if r["metric"] == name)


def test_srm_passes_on_clean_randomization():
    assert RES["srm"]["passed"]


def test_srm_catches_imbalance():
    import pandas as pd
    bad = pd.Series(["control"] * 15450 + ["treatment"] * 14550)
    assert not srm_check(bad)["passed"]


def test_cuped_reduces_variance_near_rho_squared():
    _, _, var_red = cuped(DF["revenue"], DF["pre_revenue"])
    assert abs(var_red - RHO ** 2) < 0.03  # ~0.36


def test_cuped_tightens_primary_ci():
    prim = _metric("revenue")
    raw_w = prim["raw"]["ci"][1] - prim["raw"]["ci"][0]
    cup_w = prim["ci"][1] - prim["ci"][0]
    assert cup_w < raw_w


def test_primary_ci_covers_ground_truth():
    lo, hi = _metric("revenue")["ci"]
    assert lo <= GROUND_TRUTH["revenue"] <= hi


def test_guardrails_are_not_discoveries():
    assert not _metric("latency_ms")["bh_discovery"]
    assert not _metric("support_tickets")["bh_discovery"]


def test_primary_is_a_discovery():
    assert _metric("revenue")["bh_discovery"]


def test_bh_is_stricter_than_raw_alpha():
    # a metric with raw p just under 0.05 should be demotable by BH
    disc, crit = benjamini_hochberg([0.001, 0.04, 0.045, 0.5, 0.6], alpha=0.05)
    assert disc[0] and not disc[2]  # smallest passes; the 0.045 does not


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
