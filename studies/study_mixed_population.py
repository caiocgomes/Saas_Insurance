"""Counterfactual study: aggregate risk under explicit user heterogeneity.

Compares a homogeneous NB-LogNormal population (the default scenario, where
heterogeneity is implicit through the NB overdispersion) against three mixed
populations in which a small fraction of "power users" carries most of the
consumption. All scenarios are calibrated so that the portfolio expected loss
matches the homogeneous case ($30/user, $300k aggregate at n=10,000).

Scenarios:
    H.  Homogeneous: single NB-LogNormal, E[per-user] = $30.
    M1. 90% light + 10% power, light E=$10, power E=$210.
    M2. 80% light + 20% power, light E=$10, power E=$110.
    M3. 95% light + 5% power,  light E=$15, power E=$315.

For each scenario we report E[L], VaR_0.99(L), TVaR_0.99(L), and the cap-hit
fraction in the power and light segments separately (where defined).

Run: uv run python studies/study_mixed_population.py
Output: stdout table + output/tab_mixed_population.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from saas_actuaria import (
    MixedPopulationModel,
    NBLogNormalModel,
    default_scenario,
    simulate_portfolio,
    tvar_alpha,
    var_alpha,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_PATH = OUTPUT_DIR / "tab_mixed_population.csv"


# Common calibration: cap, n_users, n_replications.
CAP = 1_000.0
N_USERS = 10_000
N_REPS = 2_000
PREMIUM_WEEKLY = 50.0
SEED = 20260515


def light_model(mean_n: float, mean_s: float, sigma: float = 0.8) -> NBLogNormalModel:
    """NB-LogNormal calibrated to a target mean event count and event severity.

    NB parameter convention: mean_N = r * (1 - p) / p. We fix r = 1 and solve
    for p to hit the target mean_n. LogNormal: mu = log(mean_s) - sigma^2 / 2.
    """
    r = 1.0
    p = r / (r + mean_n)
    mu = float(np.log(mean_s) - 0.5 * sigma**2)
    return NBLogNormalModel(r=r, p=p, mu=mu, sigma=sigma)


def power_model(mean_n: float, mean_s: float, sigma: float = 1.5) -> NBLogNormalModel:
    """NB-LogNormal with heavier tail and higher dispersion in counts."""
    r = 2.0
    p = r / (r + mean_n)
    mu = float(np.log(mean_s) - 0.5 * sigma**2)
    return NBLogNormalModel(r=r, p=p, mu=mu, sigma=sigma)


@dataclass(frozen=True)
class MixedScenario:
    name: str
    pi_power: float
    mixed: MixedPopulationModel


SCENARIOS: list[MixedScenario] = [
    MixedScenario(
        name="M1 (90% light, 10% power)",
        pi_power=0.10,
        mixed=MixedPopulationModel(
            pi_power=0.10,
            light=light_model(mean_n=5.0, mean_s=2.0),   # E_L = $10
            power=power_model(mean_n=30.0, mean_s=7.0),  # E_P = $210
        ),
    ),
    MixedScenario(
        name="M2 (80% light, 20% power)",
        pi_power=0.20,
        mixed=MixedPopulationModel(
            pi_power=0.20,
            light=light_model(mean_n=5.0, mean_s=2.0),   # E_L = $10
            power=power_model(mean_n=22.0, mean_s=5.0),  # E_P = $110
        ),
    ),
    MixedScenario(
        name="M3 (95% light, 5% power)",
        pi_power=0.05,
        mixed=MixedPopulationModel(
            pi_power=0.05,
            light=light_model(mean_n=5.0, mean_s=3.0),    # E_L = $15
            power=power_model(mean_n=45.0, mean_s=7.0),   # E_P = $315
        ),
    ),
]


def cap_hit_fractions_by_segment(
    scenario: MixedScenario,
    n_users: int,
    n_replications: int,
    cap: float,
    seed: int,
) -> tuple[float, float]:
    """Return (cap_hit_power, cap_hit_light) averaged across replications."""
    rng = np.random.default_rng(seed)
    power_hits = []
    light_hits = []
    for _ in range(n_replications):
        costs, is_power = scenario.mixed.sample_portfolio_with_segments(rng, n_users, cap)
        # A user "hits the cap" if their realized cost equals the cap exactly.
        # Use a near-equality test to avoid floating point edge cases.
        hits = np.isclose(costs, cap, rtol=0.0, atol=1e-9)
        n_power = int(is_power.sum())
        n_light = int((~is_power).sum())
        if n_power > 0:
            power_hits.append(int((hits & is_power).sum()) / n_power)
        if n_light > 0:
            light_hits.append(int((hits & ~is_power).sum()) / n_light)
    return float(np.mean(power_hits)) if power_hits else 0.0, \
           float(np.mean(light_hits)) if light_hits else 0.0


def main() -> None:
    # Homogeneous baseline (matched-mean default scenario).
    homog = default_scenario()
    losses_hom = simulate_portfolio(
        model=homog.nb_lognormal,
        n_users=N_USERS,
        cap=CAP,
        n_replications=N_REPS,
        seed=SEED,
    )
    premium_income = PREMIUM_WEEKLY * N_USERS

    rows = [
        dict(
            scenario="H. Homogeneous (default)",
            pi_power=float("nan"),
            mean_user_E_L_dollar=homog.nb_lognormal.mean_aggregate_uncapped(),
            expected_L=float(losses_hom.mean()),
            var_99=var_alpha(losses_hom, 0.01),
            tvar_99=tvar_alpha(losses_hom, 0.01),
            tvar_999=tvar_alpha(losses_hom, 0.001),
            cap_hit_power=float("nan"),
            cap_hit_light=float("nan"),
            reserve_tvar99=max(0.0, tvar_alpha(losses_hom, 0.01) - premium_income),
        )
    ]

    for sc in SCENARIOS:
        losses = simulate_portfolio(
            model=sc.mixed,
            n_users=N_USERS,
            cap=CAP,
            n_replications=N_REPS,
            seed=SEED + int(sc.pi_power * 1000),
        )
        cap_hit_p, cap_hit_l = cap_hit_fractions_by_segment(
            sc,
            n_users=N_USERS,
            n_replications=200,  # smaller for cap-hit detail
            cap=CAP,
            seed=SEED + 1 + int(sc.pi_power * 1000),
        )
        rows.append(
            dict(
                scenario=sc.name,
                pi_power=sc.pi_power,
                mean_user_E_L_dollar=sc.mixed.mean_aggregate_uncapped(),
                expected_L=float(losses.mean()),
                var_99=var_alpha(losses, 0.01),
                tvar_99=tvar_alpha(losses, 0.01),
                tvar_999=tvar_alpha(losses, 0.001),
                cap_hit_power=cap_hit_p,
                cap_hit_light=cap_hit_l,
                reserve_tvar99=max(0.0, tvar_alpha(losses, 0.01) - premium_income),
            )
        )

    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, float_format="%.4f")

    def fmt_usd(v) -> str:
        return "    n/a    " if not np.isfinite(v) else f"${v:>11,.0f}"

    def fmt_pct(v) -> str:
        return "  n/a " if not np.isfinite(v) else f"{v:>6.1%}"

    print(
        f"Portfolio: n = {N_USERS:,}, cap = ${CAP:,.0f}, premium income = ${premium_income:,.0f}"
    )
    print()
    cols = [
        "scenario", "pi_power", "expected_L", "var_99", "tvar_99",
        "cap_hit_power", "cap_hit_light", "reserve_tvar99",
    ]
    print(df[cols].to_string(
        index=False,
        formatters={
            "expected_L": fmt_usd,
            "var_99": fmt_usd,
            "tvar_99": fmt_usd,
            "reserve_tvar99": fmt_usd,
            "cap_hit_power": fmt_pct,
            "cap_hit_light": fmt_pct,
            "pi_power": lambda v: " n/a " if not np.isfinite(v) else f"{v:>5.0%}",
        },
    ))
    print(f"\nCSV written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
