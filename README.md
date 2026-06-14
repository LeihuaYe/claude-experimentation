# claude-experimentation

Trustworthy experiment analysis as Claude Code skills (and small, dependency-light Python libraries) — covering an experiment end to end:

- **`ab-design`** — size it *before* it runs: sample size / MDE / power / duration, with CUPED, clustering, ratio metrics, and multiple-metric alpha.
- **`ab-readout`** — read it out *after* it runs, **as a pipeline of gates, not a single p-value.**
- **`ab-cate`** — go *beyond the average*: heterogeneous effects (S/T/X-learners), regression-adjusted ATE, and a subgroup-fishing guard so segment stories aren't false positives.

> Most experiments are lost before they start (underpowered) or misread when they end (a significant result on a broken test). These skills check the experiment at both ends.

## Why this exists

Most A/B readouts skip the steps that decide whether the number means anything:

| Step | Question it answers | What goes wrong without it |
|------|--------------------|----------------------------|
| **1. SRM** | Was randomization even valid? | A 50.4/49.6 split at scale silently invalidates the whole test |
| **2. CUPED** | Can we cut the noise for free? | You ship a "flat" result that was actually a win hidden under variance |
| **3. Effect + CI** | How big, how sure? | A bare p-value hides the effect size and the uncertainty |
| **4. Benjamini-Hochberg** | Are we fooling ourselves across metrics? | The best of 8 metrics is "significant" by chance |
| **5. Verdict** | Ship or not? | Cherry-picked metrics, ignored guardrails |

## Install

```bash
pip install -r requirements.txt   # numpy, scipy, pandas
```

## Use it

**CLI** — point at a CSV:

```bash
python -m ab_readout \
  --data examples/example_experiment.csv \
  --arm group --primary revenue \
  --metrics revenue,engagement_min,conversion,latency_ms,support_tickets \
  --binary conversion \
  --guardrails latency_ms,support_tickets \
  --covariate pre_revenue
```

**Library:**

```python
import pandas as pd
from ab_readout import run_readout, format_report

df = pd.read_csv("my_experiment.csv")
res = run_readout(df, arm_col="group", primary="revenue",
                  metrics=["revenue", "conversion"], binary=["conversion"],
                  guardrails=["latency_ms"], covariate="pre_revenue")
print(format_report(res))   # or use res (a dict) programmatically
```

**As a Claude Code skill:** drop `SKILL.md` into `~/.claude/skills/ab-readout/` and ask Claude to "read out this experiment" — it gathers the columns, runs the pipeline, and interprets the result conservatively (SRM failure halts the readout).

## Example output

```
[1] SAMPLE RATIO MISMATCH  ->  PASS — randomization looks valid
[2] CUPED  corr=+0.599  variance reduction=35.9%  (CI ~20% narrower, same users)
[3] PRIMARY EFFECT
      raw    rel=+1.73%  95% CI [+0.0586, +0.2853]  p=0.0029
      CUPED  rel=+1.68%  95% CI [+0.0770, +0.2585]  p=0.0003
[4] METRIC PANEL + BH FDR
      revenue   [primary]  p=0.0003  <-- discovery
      engagement_min       p=0.0423  (does NOT survive BH)
      ...guardrails clean
[5] VERDICT  ->  SHIP
```

Full output in [`examples/example_output.txt`](examples/example_output.txt).

## Design the experiment first (`ab-design`)

A readout is only trustworthy if the test was powered. `ab-design` sizes it before launch — sample size, MDE, power, and duration are one quadrilateral; fix any three and it computes the fourth.

```bash
# how many users to detect a 0.1 lift on a mean metric, with a CUPED covariate?
python -m ab_design --kind mean --baseline 10 --sd 5 --mde 0.1 --rho 0.6 --daily 8000

# proportion metric, clustered randomization, 3 primary metrics
python -m ab_design --kind proportion --baseline 0.10 --mde 0.02 \
  --cluster-size 12 --icc 0.05 --k-primary 3
```

```python
from ab_design import sample_size, mde, format_design_report

res = sample_size("mean", baseline=10, mde=0.1, sd=5, rho=0.6)   # CUPED cuts n by 1-rho^2
print(format_design_report(res, daily_users_per_arm=8000))

mde("proportion", baseline=0.10, n=20000)   # inverse: what can I detect with the traffic I have?
```

It handles the design realities that actually move required n: **CUPED** (covariate cuts n by `1−ρ²`), **clustered randomization** (design effect `1+(m−1)·ICC`), **ratio metrics** (delta-method variance), and **multiple primary metrics** (alpha allocation). Validated against Monte-Carlo power — `python tests/test_design.py` (12/12). Hand the result to `ab-readout` at the pre-registered n.

## Go beyond the average effect (`ab-cate`)

The ATE hides the story — and slicing by segment until something turns green is how analysts fool themselves. `ab-cate` estimates *how* the effect varies and guards against fished subgroups.

```bash
python -m ab_cate --data exp.csv --arm group --outcome revenue \
  --covariates pre_revenue,tenure_days --learner x --subgroup-col country
```

```python
from ab_cate import lin_estimator, x_learner, subgroup_fishing_guard, cate_summary

lin = lin_estimator(X, T, Y)                 # regression-adjusted ATE + variance reduction (CUPED's cousin)
summ = cate_summary(x_learner(X, T, Y), X)   # CATE spread: is there real heterogeneity?
guard = subgroup_fishing_guard(T, Y, {"power_users": mask_a, "new_users": mask_b})
```

It gives the **regression-adjusted ATE** (Lin's estimator — unbiased *and* lower-variance), **S/T/X-learner** CATE estimates, **honest** (sample-split) estimation, and a **subgroup-fishing guard** that Benjamini-Hochberg-corrects across every slice you tried — so "it worked better for X" has to survive the correction. Validated against known heterogeneity — `python tests/test_cate.py` (9/9).

## The stats are checked, not just plausible

The methods are validated against a simulation with a **known ground truth** — see [`tests/test_readout.py`](tests/test_readout.py): CUPED variance reduction lands near ρ², the primary CI covers the true effect, BH demotes the true nulls, SRM catches a real imbalance.

```bash
python tests/test_readout.py    # 8/8 passed
```

## Methods

- **SRM** — chi-square goodness-of-fit on the assignment split (α=0.001, because it's a data-quality alarm).
- **CUPED** — `Y_adj = Y − θ(X − E[X])`, `θ = Cov(Y,X)/Var(X)`; variance falls ~ρ², CIs shrink for free.
- **Effect** — Welch (unequal-variance) for means, two-proportion z for binary, with 95% CIs.
- **Benjamini-Hochberg** — FDR control across the metric panel so the best borderline metric isn't a false positive.

## Roadmap

`ab-design`, `ab-readout`, and `ab-cate` ship today. Planned: **sequential / always-valid** inference (peeking-safe stopping) and **causal inference without randomization** (DiD / synthetic control).

## License

MIT — see [LICENSE](LICENSE). Built by Leihua Ye.
