"""Figure: aggregate portfolio cost distribution under three pricing models.

Compares the distribution of L = sum_i C_i across Monte Carlo replications for:
    (a) naive baseline (deterministic E[L] = n * E[per-user uncapped cost])
    (b) Poisson-Gamma compound with cap
    (c) NB-LogNormal compound with cap

Headline message: the right tail of (b) and (c) is materially fatter than the
point mass at (a). Reserve adequacy based on (a) is silently undercapitalized.

Output: output/fig_aggregate_cost_distribution.pdf

Run: uv run python figures/fig_aggregate_cost_distribution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from saas_actuaria import (
    default_scenario,
    simulate_portfolio,
    var_alpha,
    tvar_alpha,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH = OUTPUT_DIR / "fig_aggregate_cost_distribution.pdf"


def main() -> None:
    scenario = default_scenario()

    # Simulate portfolio aggregate under each compound family.
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

    # Naive baseline: deterministic expected portfolio cost (no variance).
    # We use the uncapped expected per-user aggregate under Poisson-Gamma as the
    # "naive unit-economics" point estimate, because it is what a per-token cost
    # estimate would yield in the absence of compound modeling.
    naive_expected = (
        scenario.poisson_gamma.mean_aggregate_uncapped() * scenario.n_users
    )

    premium_income = scenario.premium_income

    # Tail metrics at alpha = 0.01.
    alpha = 0.01
    var_pg = var_alpha(losses_pg, alpha)
    tvar_pg = tvar_alpha(losses_pg, alpha)
    var_nb = var_alpha(losses_nb, alpha)
    tvar_nb = tvar_alpha(losses_nb, alpha)

    # Plot.
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bins = np.linspace(
        min(losses_pg.min(), losses_nb.min()) * 0.95,
        max(losses_pg.max(), losses_nb.max()) * 1.05,
        80,
    )
    ax.hist(
        losses_pg,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Poisson-Gamma (compound, capped)",
        edgecolor="none",
    )
    ax.hist(
        losses_nb,
        bins=bins,
        density=True,
        alpha=0.55,
        label="NB-LogNormal (compound, capped)",
        edgecolor="none",
    )
    ymax = ax.get_ylim()[1]
    ax.axvline(
        naive_expected,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Naive baseline (unit economics): ${naive_expected:,.0f}",
    )
    ax.axvline(
        premium_income,
        color="firebrick",
        linestyle=":",
        linewidth=1.4,
        label=f"Premium income: ${premium_income:,.0f}",
    )
    ax.text(
        var_nb,
        ymax * 0.92,
        f"NB-LogNormal\nVaR$_{{0.99}}$ = \\${var_nb:,.0f}\nTVaR$_{{0.99}}$ = \\${tvar_nb:,.0f}",
        fontsize=8,
        va="top",
        ha="left",
    )
    ax.set_xlabel("Aggregate portfolio cost over one reset period, USD")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Aggregate cost distribution ({scenario.n_users:,} users, "
        f"weekly cap \\${scenario.cap:,.0f})"
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf")
    plt.close(fig)

    # Summary to stdout.
    print(f"Scenario: {scenario.name}")
    print(f"  n_users          = {scenario.n_users:,}")
    print(f"  cap              = ${scenario.cap:,.0f}")
    print(f"  premium income   = ${premium_income:,.0f}")
    print(f"  naive baseline   = ${naive_expected:,.0f}")
    print("Poisson-Gamma:")
    print(f"  E[L]             = ${losses_pg.mean():,.0f}")
    print(f"  VaR_0.99(L)      = ${var_pg:,.0f}")
    print(f"  TVaR_0.99(L)     = ${tvar_pg:,.0f}")
    print("NB-LogNormal:")
    print(f"  E[L]             = ${losses_nb.mean():,.0f}")
    print(f"  VaR_0.99(L)      = ${var_nb:,.0f}")
    print(f"  TVaR_0.99(L)     = ${tvar_nb:,.0f}")
    print(f"Figure written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
