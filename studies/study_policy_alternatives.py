"""Counterfactual study: how would aggregate cost behave under alternative
seller-side policies?

Compares four policy regimes against the default scenario (NB-LogNormal severity),
holding the user behavior model fixed. The point is to quantify what the cap is
buying for the seller and to characterize where the cap-free regime fails.

Policies compared (all evaluated against the same NB-LogNormal user model):

    P0. Hard cap at K = $1,000 (the default; current Anthropic-like setting).
    P1. Loose cap at K = $5,000 (5x the default).
    P2. No cap (K = infinity).
    P3. Pure pay-per-use (no premium, no cap; user is billed exact severity).

For P0-P2 the seller charges a fixed premium $50/week per user; P3 represents
the API-direct model (Demirer-Fradkin-Tadelis-Peng 2025 baseline).

Metrics: E[L], TVaR_0.99, loss ratio, fraction of users who hit the cap (if any),
implied per-user reserve at TVaR_0.999 above E[L].

Run: uv run python studies/study_policy_alternatives.py
Output: stdout table + output/tab_policy_alternatives.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from saas_actuaria import (
    NBLogNormalModel,
    heavy_consumption_scenario,
    simulate_portfolio,
    tvar_alpha,
    var_alpha,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH = OUTPUT_DIR / "tab_policy_alternatives.csv"


@dataclass(frozen=True)
class Policy:
    name: str
    cap: float
    premium_per_user: float
    description: str


def cap_hit_fraction(
    model: NBLogNormalModel,
    n_users: int,
    cap: float,
    n_replications: int,
    seed: int,
) -> float:
    """Estimate the fraction of users whose uncapped aggregate would exceed the cap.

    This is computed by simulating uncapped per-user aggregates and counting
    those that exceed the cap. Independent of the portfolio aggregate sim.
    """
    rng = np.random.default_rng(seed)
    # Sample n_replications * n_users users uncapped; very large draw at once.
    sample_size = min(n_users * n_replications, 5_000_000)
    counts = rng.negative_binomial(model.r, model.p, size=sample_size)
    out = np.empty(sample_size, dtype=np.float64)
    total_events = int(counts.sum())
    if total_events == 0:
        return 0.0
    all_severities = rng.lognormal(model.mu, model.sigma, size=total_events)
    cursor = 0
    for i, k in enumerate(counts):
        if k == 0:
            out[i] = 0.0
        else:
            out[i] = all_severities[cursor : cursor + k].sum()
            cursor += k
    return float((out > cap).mean())


def main() -> None:
    scenario = heavy_consumption_scenario()
    n_users = scenario.n_users
    n_reps = scenario.n_replications
    model = scenario.nb_lognormal

    policies = [
        Policy(
            name="P0. Hard cap $1,000",
            cap=1_000.0,
            premium_per_user=50.0,
            description="Default; current Anthropic-like setting",
        ),
        Policy(
            name="P1. Loose cap $5,000",
            cap=5_000.0,
            premium_per_user=50.0,
            description="5x the default cap; lets the tail leak",
        ),
        Policy(
            name="P2. No cap",
            cap=float("inf"),
            premium_per_user=50.0,
            description="Pre-Aug-2025 Claude Code; pure flat-rate with no cap",
        ),
        Policy(
            name="P3. Pure pay-per-use",
            cap=float("inf"),
            premium_per_user=0.0,
            description="API-direct model; no subscription, user pays exact",
        ),
    ]

    rows: list[dict] = []
    for i, pol in enumerate(policies):
        # For P3, premium income is the realized loss exactly, so loss ratio = 1
        # by construction. We still simulate to report E[L] and tail.
        losses = simulate_portfolio(
            model=model,
            n_users=n_users,
            cap=pol.cap,
            n_replications=n_reps,
            seed=scenario.seed + 100 + i,
        )
        premium_income = pol.premium_per_user * n_users
        e_l = float(losses.mean())
        var_99 = var_alpha(losses, 0.01)
        tvar_99 = tvar_alpha(losses, 0.01)
        tvar_999 = tvar_alpha(losses, 0.001)
        # cap-hit rate for finite caps only
        if np.isfinite(pol.cap):
            cap_hit = cap_hit_fraction(
                model=model,
                n_users=n_users,
                cap=pol.cap,
                n_replications=1,
                seed=scenario.seed + 200 + i,
            )
        else:
            cap_hit = 0.0
        # Loss ratio: for P3, by construction == 1.
        if pol.premium_per_user > 0:
            lr = e_l / premium_income
            margin_99 = premium_income - tvar_99
        else:
            lr = float("nan")
            margin_99 = float("nan")
        rows.append(
            dict(
                policy=pol.name,
                description=pol.description,
                cap=pol.cap,
                premium_per_user=pol.premium_per_user,
                premium_income=premium_income,
                expected_L=e_l,
                var_99=var_99,
                tvar_99=tvar_99,
                tvar_999=tvar_999,
                cap_hit_fraction=cap_hit,
                loss_ratio=lr,
                margin_over_tvar_99=margin_99,
            )
        )

    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")

    # Pretty print.
    def fmt_usd(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "        n/a"
        return f"${v:>12,.0f}"

    def fmt_ratio(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "     n/a"
        return f"{v:>8.1%}"

    print(f"Scenario: {scenario.name}")
    print(f"  n_users   = {n_users:,}")
    print(f"  user model = NB-LogNormal(r={model.r}, p={model.p}, "
          f"mu={model.mu}, sigma={model.sigma})")
    print(f"  E[per-user uncapped] = ${model.mean_aggregate_uncapped():,.2f}")
    print()
    cols_to_show = [
        "policy", "cap", "premium_per_user", "expected_L",
        "tvar_99", "tvar_999", "cap_hit_fraction", "loss_ratio", "margin_over_tvar_99",
    ]
    df_show = df[cols_to_show].copy()
    df_show["cap"] = df_show["cap"].apply(
        lambda v: "no cap" if not np.isfinite(v) else f"${v:,.0f}"
    )
    df_show["premium_per_user"] = df_show["premium_per_user"].apply(
        lambda v: f"${v:,.0f}"
    )
    print(
        df_show.to_string(
            index=False,
            formatters={
                "expected_L": fmt_usd,
                "tvar_99": fmt_usd,
                "tvar_999": fmt_usd,
                "cap_hit_fraction": fmt_ratio,
                "loss_ratio": fmt_ratio,
                "margin_over_tvar_99": fmt_usd,
            },
        )
    )
    print(f"\nCSV written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
