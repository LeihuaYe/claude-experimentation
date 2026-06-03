"""Generate a synthetic experiment with KNOWN ground truth, for the example + tests.

A pre-experiment covariate `pre_revenue` is correlated with `revenue` (the CUPED
lever). `latency_ms` and `support_tickets` are true nulls (guardrails should stay
clean). `engagement_min` and `conversion` carry small real effects.
"""
import numpy as np
import pandas as pd

GROUND_TRUTH = {"revenue": 0.10, "engagement_min": 0.04,
                "conversion": 0.004, "latency_ms": 0.0, "support_tickets": 0.0}
RHO = 0.6  # corr(pre_revenue, revenue)


def make(n=30_000, seed=42):
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, size=n)
    pre = rng.normal(10, 5, size=n)
    revenue = (10 + RHO * (pre - 10)
               + rng.normal(0, 5 * np.sqrt(1 - RHO ** 2), size=n)
               + GROUND_TRUTH["revenue"] * arm)
    return pd.DataFrame({
        "group": np.where(arm == 1, "treatment", "control"),
        "pre_revenue": pre,
        "revenue": revenue,
        "engagement_min": rng.normal(20, 8, size=n) + GROUND_TRUTH["engagement_min"] * arm,
        "conversion": (rng.random(n) < (0.10 + GROUND_TRUTH["conversion"] * arm)).astype(int),
        "latency_ms": rng.normal(250, 40, size=n) + GROUND_TRUTH["latency_ms"] * arm,
        "support_tickets": rng.normal(0.3, 1.0, size=n) + GROUND_TRUTH["support_tickets"] * arm,
    })


if __name__ == "__main__":
    make().to_csv("examples/example_experiment.csv", index=False)
    print("wrote examples/example_experiment.csv")
