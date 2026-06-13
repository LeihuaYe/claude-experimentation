---
name: ab-design
description: Design an experiment / power analysis before it runs — sample size, minimum detectable effect (MDE), power, and duration for mean or proportion metrics, including CUPED variance reduction, clustered randomization (design effect / ICC), ratio-metric sizing via the delta method, alpha allocation across multiple primary metrics, and switchback designs. Use when the user says "how many users do I need", "sample size for this test", "is this test powered", "what MDE can I detect", "how long should the experiment run", "power analysis", "design an A/B test", "size an experiment", "switchback design", or asks how big / how long before launching a test.
---

You are sizing an experiment **before** it runs. Most underpowered tests are lost here, not in the readout. Sample size, MDE, power, and significance are one quadrilateral — fix any three and compute the fourth. Pair with `ab-readout` (the post-experiment analysis); this is its mirror image.

## Step 1 — Gather the inputs
- **Metric kind** — a **mean** (revenue, time-on-page; needs the metric's standard deviation `sd`) or a **proportion** (conversion, CTR; needs the baseline rate).
- **Baseline** — control mean, or control rate.
- **What to detect** — either the **MDE** (smallest effect worth shipping, in absolute units) to get sample size, or the **n you can afford** to get the MDE.
- **power** (default 0.80) and **alpha** (default 0.05, two-sided).
- Then the design realities that move n — ask only if relevant:
  - **CUPED covariate** — a pre-experiment column correlated (`rho`) with the metric; cuts required n by `1 − rho²`. Means only; the covariate must be pre-treatment.
  - **Clustering** — if you randomize a unit that contains many measurements (users → sessions), pass `cluster_size` + `icc`; n inflates by the design effect `1 + (m−1)·ICC`.
  - **Multiple primary metrics** — `k` primaries split alpha; size for the corrected alpha or you arrive underpowered.
  - **Ratio metric** (clicks/impression) — its variance needs the delta method, not the naive sample variance.

## Step 2 — Compute with the library (don't hand-roll the stats)

```python
from ab_design import sample_size, mde, power, allocate_alpha, ratio_variance, format_design_report

# how many users? (mean metric, with CUPED + clustering)
res = sample_size("mean", baseline=10, mde=0.1, sd=5, rho=0.6, icc=0.05, cluster_size=12)
print(format_design_report(res, daily_users_per_arm=8000))

# inverse: what can I detect with the traffic I have?
detectable = mde("proportion", baseline=0.10, n=20000)

# k primary metrics -> size for the Bonferroni-corrected alpha
a = allocate_alpha(0.05, k=3)[0]
res = sample_size("proportion", baseline=0.10, mde=0.02, alpha=a)
```

For a ratio metric, get the per-unit variance first, then size as a mean:
`v = ratio_variance(clicks, impressions); sample_size("mean", baseline=v["ratio"], mde=..., sd=v["unit_sd"])`.

## Step 3 — Interpret honestly
- **Report the whole quadrilateral**, not just n: "to detect a 1% relative lift at 80% power you need N/arm ≈ D days — if you can only run W days, the detectable effect is M." A number with its tradeoff is the deliverable.
- **CUPED**: only meaningfully cuts n when |rho| is appreciable (≈ rho² reduction). Don't promise a big cut on a weakly correlated covariate.
- **Clustering is not optional accounting**: sizing on raw n when you randomize clusters is *the* classic over-optimistic mistake — the test is underpowered by the design effect. Always surface clusters/arm.
- **Multiple primaries**: state the per-metric alpha and that n was sized for it.
- If the required n or duration is infeasible, say so plainly and offer the levers: raise MDE (ship only bigger wins), add a CUPED covariate, reduce metric variance, or accept lower power — don't quietly run an underpowered test.

## Step 4 — Hand off to the readout
The design fixes the decision rule before data exists (the anti-peeking discipline). When the experiment finishes, analyze it with **`ab-readout`** at the pre-registered n — same alpha, same primary, same guardrails.
