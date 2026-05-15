"""Figure: required reserve per user as a function of portfolio size n.

Illustrates the law-of-large-numbers effect on per-user reserve:
required reserve R(n) / n decreases as 1/sqrt(n) up to a heavy-tail correction.
This is the actuarial argument for why community rating works at portfolio
scale and why small-portfolio capped-usage businesses are more capital-intensive
per user than large ones.

Uses NB-LogNormal (the heavy-tailed case) so the law-of-large-numbers effect
is visible against a fat-tailed background.

Output: output/fig_reserve_by_portfolio_size.pdf

Run: uv run python figures/fig_reserve_by_portfolio_size.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from saas_actuaria import (
    default_scenario,
    simulate_portfolio,
    tvar_alpha,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH = OUTPUT_DIR / "fig_reserve_by_portfolio_size.pdf"


def main() -> None:
    scenario = default_scenario()
    # Use a tighter tail to make the per-user reserve non-degenerate at the
    # scenario's premium level. We measure "reserve relative to expected loss",
    # i.e. TVaR_{1-alpha}(L) - E[L], which is a pure measure of tail capital
    # need per user, independent of pricing.
    alpha = 0.001  # 0.1%, catastrophic tail

    portfolio_sizes = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    n_reps = 2_000  # need more replications to estimate 0.1% tail well

    per_user_reserve = []
    for n in portfolio_sizes:
        losses = simulate_portfolio(
            model=scenario.nb_lognormal,
            n_users=n,
            cap=scenario.cap,
            n_replications=n_reps,
            seed=scenario.seed + 2 + n,
        )
        # Tail capital per user: TVaR above expected loss, normalized by n.
        tail_capital = max(0.0, tvar_alpha(losses, alpha) - losses.mean())
        per_user_reserve.append(tail_capital / n)

    per_user_reserve = np.asarray(per_user_reserve)
    sizes = np.asarray(portfolio_sizes)

    # Reference line: R/n proportional to 1/sqrt(n), anchored at the first point.
    anchor = per_user_reserve[0] * np.sqrt(sizes[0])
    reference = anchor / np.sqrt(sizes)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        sizes,
        per_user_reserve,
        marker="o",
        linewidth=1.6,
        label=r"Simulated tail capital / $n$ (TVaR$_{0.999}$ above $E[L]$)",
    )
    ax.plot(
        sizes,
        reference,
        linestyle="--",
        color="grey",
        linewidth=1.0,
        label=r"$\propto 1/\sqrt{n}$ reference",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Portfolio size $n$ (subscribers)")
    ax.set_ylabel("Per-user tail capital, USD")
    ax.set_title(
        f"Per-user tail capital vs. portfolio size (NB-LogNormal, cap \\${scenario.cap:,.0f}/week)"
    )
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf")
    plt.close(fig)

    print(f"Per-user tail capital at TVaR_0.999 above E[L], by portfolio size:")
    for n, R in zip(portfolio_sizes, per_user_reserve):
        print(f"  n = {n:>7,}   tail_capital/n = ${R:>8,.4f}")
    print(f"Figure written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
