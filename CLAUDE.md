# claude-experimentation — project context

Public OSS Claude Code skill(s) for trustworthy experiment analysis. First skill: `ab-readout`. This is the COI-clean **OSS-reputation** lane (free MIT, no monetization) — positioning/credibility for Leihua's causal-inference/experimentation expertise, distributed via his Medium/LinkedIn audience. See `~/.claude/.../memory/reference_cc_profit_replication_analysis.md` for the strategy and the latent monetization path.

## Layout
- `ab_readout/` — the library (`readout.py` = SRM/CUPED/effect/BH/verdict; `cli.py`; `__main__.py`).
- `SKILL.md` — the Claude Code skill definition (drop into `~/.claude/skills/ab-readout/`).
- `examples/make_synthetic.py` — synthetic data with known ground truth; `example_output.txt` committed.
- `tests/test_readout.py` — validates the stats against ground truth (run: `python tests/test_readout.py`).

## Conventions
- Stats must be correct first (Leihua is a PhD experimentation lead — wrong stats kill the credibility play). Every method is checked against a simulation with a known answer; keep it that way.
- Dependency-light (numpy/scipy/pandas only). No heavy frameworks.
- Match existing code density; no speculative abstractions.

## Status
- Built 2026-06-02 from the validated inline prototype. Repo created PRIVATE — **public launch is a separate, explicit decision** (the OSS-reputation payoff requires going public + a launch post, but that's Leihua's call, not automatic).
- Launch plan when ready: flip public → Medium/LinkedIn post demoing it end-to-end (data + code, per his content rule) → drives stars/credibility.

## Sunset criterion (per skill-sunset-at-creation rule)
Re-evaluate at the **2026-08-05 quarterly skill audit**: keep only if it has been (a) launched publicly and gained traction (stars / inbound / cited in a post), OR (b) used by Leihua in real readouts ≥1×/month. If neither by the following quarter (Nov 5), archive — a private repo that never launched isn't earning the reputation payoff it was built for.
