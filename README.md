# claude-experimentation

Trustworthy experiment analysis as a Claude Code skill (and a small, dependency-light Python library). Point it at an A/B test and it runs the readout the way a careful data scientist would — **as a pipeline of gates, not a single p-value.**

> A statistically significant result on a broken experiment is worse than no result. This skill checks the experiment *before* it reads the effect.

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

`ab-readout` is the first skill. Planned: experiment **design** (power / MDE / switchback), **sequential / always-valid** inference, and **heterogeneous effects** (CATE).

## License

MIT — see [LICENSE](LICENSE). Built by Leihua Ye.
