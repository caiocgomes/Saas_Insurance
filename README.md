# sims/ — Monte Carlo simulations for the paper

Reproducible code for the worked example in §4.7 of the paper and the
sensitivity figures referenced in §4.4 and §4.2.

## Stack

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environment and lockfile
- numpy, scipy, pandas, matplotlib only

## Setup

```bash
cd sims
uv sync
```

## Reproduce all figures and tables

```bash
uv run python figures/fig_aggregate_cost_distribution.py   # Figure 1
uv run python figures/fig_reserve_by_portfolio_size.py     # Figure 2
uv run python studies/study_censoring_bias.py              # §4.2 numbers
uv run python tables/tab_reserve_comparison.py             # Table 2 (baseline)
uv run python studies/study_policy_alternatives.py         # Table 3 (stress policy)
uv run python studies/study_mixed_population.py            # Table 5 (heterogeneity)
uv run python studies/study_vercel.py                      # Table 4 (Vercel overage)
```

Output lands in `output/`. Figures are PDF (vector) for direct inclusion in
the arXiv LaTeX build. Tables are CSV.

## Reproducibility

- All randomness flows through a single `numpy.random.Generator` seeded by
  `default_scenario().seed` (`20260515`). Sub-experiments derive deterministic
  sub-seeds from this value.
- `pyproject.toml` pins minimum versions; `uv.lock` (committed) pins exact
  versions. To reproduce bit-for-bit, run `uv sync --frozen`.

## Calibration

Each table in the paper has its own calibrated scenario. Per-script docstrings
state the parameters used and how they were chosen. Parameters are illustrative
orders-of-magnitude consistent with publicly posted Anthropic, OpenAI, Vercel,
Cloudflare, and Supabase pricing as of May 2026. None are estimates from
proprietary data.

Reference scenarios in `src/saas_actuaria/calibration.py`:

- `default_scenario()` — Claude Code Max 20x-like, matched-mean baseline. Premium $50/week, cap $1,000/week, expected per-user uncapped consumption $30/week. Drives Table 4 in the paper.
- `heavy_consumption_scenario()` — same product context with deliberately heavy users (~$760/week expected uncapped). Drives Table 5 (policy alternatives stress).
- `vercel_scenario()` — Vercel Pro-like overage regime. Premium $20/period, cap $1,000 retail-equivalent. Drives Table 2.

Pricing sources used to set magnitudes (URLs as of May 2026):

- Anthropic: https://www.anthropic.com/pricing
- OpenAI: https://openai.com/api/pricing/
- GitHub Copilot: https://github.com/features/copilot/plans
- Vercel: https://vercel.com/pricing
- Cloudflare Workers: https://developers.cloudflare.com/workers/platform/pricing/
- Supabase: https://supabase.com/pricing

## Layout

```
sims/
├── pyproject.toml
├── README.md
├── src/saas_actuaria/
│   ├── models.py          PoissonGamma and NBLogNormal compound generators
│   ├── metrics.py         VaR, TVaR, reserve, loss ratio
│   └── calibration.py     Default scenario
├── figures/
│   ├── fig_aggregate_cost_distribution.py
│   └── fig_reserve_by_portfolio_size.py
├── studies/
│   └── study_censoring_bias.py
├── studies/
│   ├── study_censoring_bias.py         Tobit MLE vs naive on capped LogNormal
│   ├── study_policy_alternatives.py    Counterfactual cap/no-cap/pay-per-use comparison
│   ├── study_mixed_population.py       Heterogeneity / adverse-selection simulation
│   └── study_vercel.py                 Overage-regime Monte Carlo (Vercel-like)
├── tables/
│   └── tab_reserve_comparison.py       Table 2 (matched-mean tail comparison)
└── output/                Generated PDFs and CSVs (gitignored)
```

## Data policy

Zero proprietary data. All pricing references are public posts; all parameters
are illustrative and stated explicitly in `calibration.py`.
