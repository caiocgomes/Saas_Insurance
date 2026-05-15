"""Table: reserve adequacy under three pricing models.

Generates a compact comparison table for §4.7 of the paper:
    Model            | E[L]  | VaR_0.99 | TVaR_0.99 | Reserve (TVaR) | Loss ratio
    Naive baseline   | x     | x        | x         | x              | x
    Poisson-Gamma    | ...
    NB-LogNormal     | ...

Output: output/tab_reserve_comparison.csv

Run: uv run python tables/tab_reserve_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from saas_actuaria import (
    default_scenario,
    simulate_portfolio,
    var_alpha,
    tvar_alpha,
    reserve_required,
    loss_ratio,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH = OUTPUT_DIR / "tab_reserve_comparison.csv"


def main() -> None:
    scenario = default_scenario()
    alpha = 0.01

    losses_pg = simulate_portfolio(
        model=scenario.poisson_gamma,
        n_users=scenario.n_users,
        cap=scenario.cap,
        n_replications=scenario.n_replications,
        seed=scenario.seed,
    )
    losses_nb = simulate_portfolio(
        model=scenario.nb_lognormal,
        n_users=scenario.n_users,
        cap=scenario.cap,
        n_replications=scenario.n_replications,
        seed=scenario.seed + 1,
    )

    naive_expected = scenario.poisson_gamma.mean_aggregate_uncapped() * scenario.n_users
    premium_income = scenario.premium_income

    def row(name: str, losses, treat_as_point: bool = False) -> dict:
        if treat_as_point:
            point = losses
            return dict(
                model=name,
                expected_L=point,
                var_99=point,
                tvar_99=point,
                reserve_tvar=max(0.0, point - premium_income),
                loss_ratio=point / premium_income,
            )
        return dict(
            model=name,
            expected_L=losses.mean(),
            var_99=var_alpha(losses, alpha),
            tvar_99=tvar_alpha(losses, alpha),
            reserve_tvar=reserve_required(losses, premium_income, alpha, "tvar"),
            loss_ratio=loss_ratio(losses, premium_income),
        )

    df = pd.DataFrame(
        [
            row("Naive baseline (point estimate)", naive_expected, treat_as_point=True),
            row("Poisson-Gamma (capped)", losses_pg),
            row("NB-LogNormal (capped)", losses_nb),
        ]
    )

    df["margin_over_tvar"] = premium_income - df["tvar_99"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")

    # Pretty-print with column-specific formatters so loss_ratio is not USD-formatted.
    def fmt_usd(v: float) -> str:
        return f"${v:>12,.0f}"

    def fmt_ratio(v: float) -> str:
        return f"{v:>8.1%}"

    print(f"Scenario: {scenario.name}")
    print(f"  Premium income: ${premium_income:,.0f}")
    print()
    print(
        df.to_string(
            index=False,
            formatters={
                "expected_L": fmt_usd,
                "var_99": fmt_usd,
                "tvar_99": fmt_usd,
                "reserve_tvar": fmt_usd,
                "margin_over_tvar": fmt_usd,
                "loss_ratio": fmt_ratio,
            },
        )
    )
    print(f"\nCSV written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
