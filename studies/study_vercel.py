"""Vercel Pro-like worked example: included-plus-overage regime.

Unlike the hard-cap regime in tables/tab_reserve_comparison.py (Claude Code
Max), the Vercel Pro contract bills overage above the included cap rather
than halting service. The seller's per-user net loss is

    L_i^net = cost_i - P_i - r_i * max(0, S_i^agg - K_i)

where cost_i is the seller's realized marginal cost (here approximated as
kappa * S_i^agg with kappa the cost-to-retail-price ratio), P_i is the
fixed premium, r_i is the overage rate, K_i is the included allowance.

Reports:
- E[S_agg] (gross consumption value at retail)
- Capped-style cost (for comparison)
- Overage revenue collected
- Net loss / net profit
- TVaR on net loss

Run: uv run python studies/study_vercel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from saas_actuaria import (
    NBLogNormalModel,
    simulate_portfolio,
    tvar_alpha,
    var_alpha,
    vercel_scenario,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH = OUTPUT_DIR / "tab_vercel.csv"


def simulate_overage_portfolio(
    model: NBLogNormalModel,
    n_users: int,
    cap_allowance: float,
    premium_per_user: float,
    overage_rate: float,
    cost_to_retail_ratio: float,
    n_replications: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate an overage-regime portfolio for n_replications.

    Returns dict with arrays of length n_replications for:
        s_agg_total: sum of S_i^agg (gross consumption value at retail)
        seller_cost: sum of cost_i = kappa * S_i^agg (seller's actual cost)
        overage_revenue: sum of r * max(0, S_i^agg - K) over users
        premium_income: scalar n * premium
        net_loss: seller_cost - premium_income - overage_revenue
    """
    rng = np.random.default_rng(seed)
    s_agg_total = np.empty(n_replications)
    seller_cost = np.empty(n_replications)
    overage_revenue = np.empty(n_replications)
    net_loss = np.empty(n_replications)
    premium_income = premium_per_user * n_users

    for rep in range(n_replications):
        # Sample uncapped per-user gross consumption value at retail prices.
        counts = rng.negative_binomial(model.r, model.p, size=n_users)
        out = np.zeros(n_users, dtype=np.float64)
        total_events = int(counts.sum())
        if total_events > 0:
            all_severities = rng.lognormal(model.mu, model.sigma, size=total_events)
            cursor = 0
            for i, k in enumerate(counts):
                if k == 0:
                    continue
                out[i] = all_severities[cursor : cursor + k].sum()
                cursor += k

        # Per-user gross consumption value (uncapped).
        s_agg = out  # shape (n_users,)
        # Seller's marginal cost: kappa fraction of retail value.
        cost_i = cost_to_retail_ratio * s_agg
        # Overage revenue: posted overage rate times consumption above the cap,
        # but priced at the per-unit overage rate relative to retail. We model
        # the overage rate as a fraction of the retail price.
        over_units = np.maximum(0.0, s_agg - cap_allowance)
        overage_rev_i = overage_rate * over_units

        s_agg_total[rep] = s_agg.sum()
        seller_cost[rep] = cost_i.sum()
        overage_revenue[rep] = overage_rev_i.sum()
        net_loss[rep] = seller_cost[rep] - premium_income - overage_revenue[rep]

    return dict(
        s_agg_total=s_agg_total,
        seller_cost=seller_cost,
        overage_revenue=overage_revenue,
        premium_income=premium_income,
        net_loss=net_loss,
    )


def main() -> None:
    scenario = vercel_scenario()
    n_users = scenario.n_users
    n_reps = scenario.n_replications
    cap = scenario.cap
    premium = scenario.premium_weekly

    # Two calibrations:
    # - "light": median-developer calibration of vercel_scenario(). Mean ~$45/user.
    # - "heavy": e-commerce / viral-release cohort. Mean ~$1100/user, with thick
    #   tail that materially crosses cap=$1000 for a non-trivial fraction of users.
    light_model = scenario.nb_lognormal
    heavy_model = NBLogNormalModel(r=2.0, p=0.10, mu=3.0, sigma=1.5)

    cohorts = [("light", light_model), ("heavy", heavy_model)]

    # Vercel Pro overage rate: ~$0.15/GB on a "$1/GB-equivalent retail" calibration
    # means r ~ 0.15. We report one representative overage rate and sweep kappa.
    overage_rates = [0.15]
    kappas = [0.25, 0.50, 1.00]

    rows = []
    for cohort_name, model in cohorts:
        for kappa in kappas:
            for r_rate in overage_rates:
                sim = simulate_overage_portfolio(
                    model=model,
                    n_users=n_users,
                    cap_allowance=cap,
                    premium_per_user=premium,
                    overage_rate=r_rate,
                    cost_to_retail_ratio=kappa,
                    n_replications=n_reps,
                    seed=scenario.seed + int(kappa * 100) + int(r_rate * 100) + (1 if cohort_name == "heavy" else 0),
                )
                net = sim["net_loss"]
                rows.append(dict(
                    cohort=cohort_name,
                    kappa=kappa,
                    overage_rate=r_rate,
                    E_S_agg=float(sim["s_agg_total"].mean()),
                    E_cost=float(sim["seller_cost"].mean()),
                    premium_income=sim["premium_income"],
                    E_overage_revenue=float(sim["overage_revenue"].mean()),
                    E_net_loss=float(net.mean()),
                    tvar99_net_loss=tvar_alpha(net, 0.01),
                    solvent_in_expectation=bool(net.mean() < 0),
                ))

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")

    print(f"Vercel-like overage scenario: n={n_users:,}, premium=${premium}/period, "
          f"cap allowance=${cap:,.0f}")
    print(f"  Light cohort: NB(r={light_model.r}, p={light_model.p}, mu={light_model.mu}, sigma={light_model.sigma}); E[per-user] = ${light_model.mean_aggregate_uncapped():,.2f}")
    print(f"  Heavy cohort: NB(r={heavy_model.r}, p={heavy_model.p}, mu={heavy_model.mu}, sigma={heavy_model.sigma}); E[per-user] = ${heavy_model.mean_aggregate_uncapped():,.2f}")
    print()

    def fmt_usd(v) -> str:
        return f"${v:>12,.0f}"

    def fmt_pct(v) -> str:
        return f"{v:>5.2f}"

    print(df.to_string(
        index=False,
        formatters={
            "E_S_agg": fmt_usd,
            "E_cost": fmt_usd,
            "premium_income": fmt_usd,
            "E_overage_revenue": fmt_usd,
            "E_net_loss": fmt_usd,
            "tvar99_net_loss": fmt_usd,
            "kappa": fmt_pct,
            "overage_rate": fmt_pct,
        },
    ))
    print(f"\nCSV written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
