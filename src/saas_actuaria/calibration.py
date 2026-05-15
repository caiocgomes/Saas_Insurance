"""Calibration of illustrative scenarios from public capped-usage AI pricing.

All parameters are illustrative and conservative. They are NOT estimates from
proprietary data; they are parameter settings chosen to produce a plausible
heavy-tailed weekly aggregate cost for a single subscription tier, calibrated
against publicly posted token pricing from Anthropic, OpenAI, and GitHub
(see references in the README). The scenario is a Claude Code Max 20x-like
weekly contract.

Units: USD-equivalent cost per event, USD premium per user per period,
period = one week.
"""

from __future__ import annotations

from dataclasses import dataclass

from saas_actuaria.models import NBLogNormalModel, PoissonGammaModel


@dataclass(frozen=True)
class Scenario:
    """A single capped-usage scenario for Monte Carlo experiments.

    Parameters
    ----------
    name :
        Short identifier (used in plot legends and filenames).
    premium_weekly :
        Per-user fixed premium for the period, in USD.
    cap :
        Non-fungible cap K, in USD-equivalent cost per period.
    n_users :
        Portfolio size (number of subscribers).
    n_replications :
        Monte Carlo replication count for portfolio aggregate.
    seed :
        Master RNG seed. Sub-experiments derive their own generators from this.
    poisson_gamma :
        PoissonGammaModel instance for the Poisson-Gamma family.
    nb_lognormal :
        NBLogNormalModel instance for the NB-LogNormal family.
    """

    name: str
    premium_weekly: float
    cap: float
    n_users: int
    n_replications: int
    seed: int
    poisson_gamma: PoissonGammaModel
    nb_lognormal: NBLogNormalModel

    @property
    def premium_income(self) -> float:
        return self.premium_weekly * self.n_users


def heavy_consumption_scenario() -> Scenario:
    """Counterfactual scenario where users are heavier consumers than the seller's
    premium implicitly assumes. Used by `studies/study_policy_alternatives.py`
    to evaluate the value of the cap as a seller-side instrument.

    Calibration:
    - Same premium ($50/week) and portfolio size as the default.
    - NB-LogNormal with r=2, p=0.05 (mean events = 38, var/mean = 20).
    - log(S) ~ N(mu=0.996, sigma=2.0): mean event = $20, heavy tail.
    - E[per-user uncapped] = 38 * 20 = $760/week.

    Without a cap, expected loss per user (~$760) is 15x premium ($50). A
    fraction of users would consume above any reasonable cap. This is the
    pre-cap regime that motivates introducing a non-fungible cap K in the
    first place.
    """
    return Scenario(
        name="heavy_consumption_unrestricted",
        premium_weekly=50.0,
        cap=1_000.0,  # baseline cap; alternatives override in the study
        n_users=10_000,
        n_replications=2_000,
        seed=20260515,
        poisson_gamma=PoissonGammaModel(lam=5.0, alpha=2.0, theta=3.0),  # unused
        nb_lognormal=NBLogNormalModel(r=2.0, p=0.05, mu=0.996, sigma=2.0),
    )


def vercel_scenario() -> Scenario:
    """Illustrative non-LLM scenario: Vercel Pro deployment platform.

    Used by Section 3.x of the paper to demonstrate that the actuarial
    structure applies to a non-LLM SaaS without invoking the LLM-specific
    severity decomposition of equation (5). The severity here is dollar-
    equivalent bandwidth + function-compute + build-time cost per event,
    where an "event" is a deployment combined with the traffic it serves
    over the month.

    Calibration (order-of-magnitude, public Vercel Pro pricing May 2026):

    - Premium: $20 / month / developer.
    - Cap (effective, bandwidth-axis): 1,000 GB / month. Vercel Pro is
      a minimum-spend contract with $0.40/GB overage above this cap, so
      the cap is the kink in the marginal price rather than a hard halt.
    - Frequency (deployments per developer per month):
        NB(r=1.5, p=0.15) -> mean = 8.5 deployments, var/mean ~ 6.7.
    - Severity (dollar-equivalent per deployment):
        log(S) ~ N(mu=0.69, sigma=1.4) -> mean event ~ $5.30,
        heavy tail to capture viral-release weeks.

    Expected per-user uncapped cost: ~ $45/month, premium $20, so
    overage exposure for the median user is positive but small; the
    operational tension is at the heavy tail (e-commerce, viral demos).
    """
    return Scenario(
        name="vercel_pro_like",
        premium_weekly=20.0,  # interpreted here as premium per period (monthly)
        cap=1_000.0,  # USD-equivalent of 1 TB bandwidth at $1/GB internal cost
        n_users=10_000,
        n_replications=2_000,
        seed=20260516,
        poisson_gamma=PoissonGammaModel(lam=8.5, alpha=2.0, theta=2.65),
        nb_lognormal=NBLogNormalModel(r=1.5, p=0.15, mu=0.69, sigma=1.4),
    )


def default_scenario() -> Scenario:
    """Illustrative scenario: Claude Code Max 20x-like, weekly contract.

    Calibration target: a viable subscription tier where premium income
    covers expected loss with comfortable margin under thin-tailed models
    but where heavy-tailed severity compresses operational margin even at
    *identical expected loss*. The headline of §4.7 is that distributional
    family matters at fixed mean — not merely that some models predict
    higher mean than others.

    - Premium: $200 / month / user = $50 / week / user
    - Cap K: $1,000 / week / user, USD-equivalent. Order-of-magnitude
      consistent with publicly reported heavy-usage consumption.

    Both models calibrated to E[per-user uncapped] = $30/week, so
    portfolio E[L] is identical across the two compound families at
    n = 10,000 users (~$300k expected, ~60% loss ratio against $500k
    premium income). The two differ only in distributional shape:

    - Poisson-Gamma (thin-tailed reference):
        N ~ Poisson(lam=5), S ~ Gamma(alpha=2, theta=3)
        Mean event = $6, mean events = 5, E[per-user] = $30.

    - NB-LogNormal (heavy-tailed, same mean):
        N ~ NB(r=1, p=0.07): mean events = 13.3, var/mean ~ 14
        log(S) ~ N(mu=-0.311, sigma=1.5): mean event = $2.256
        sigma/mean ratio of severity ~ 2.9 (heavy right tail).
        E[per-user] = 13.3 * 2.256 = $30 (matched).

    With expected loss matched, the difference in TVaR / E[L] ratio
    isolates the effect of distributional shape on tail capital needs.
    """
    return Scenario(
        name="claude_code_max20x_like",
        premium_weekly=50.0,
        cap=1000.0,
        n_users=10_000,
        n_replications=2_000,
        seed=20260515,
        poisson_gamma=PoissonGammaModel(lam=5.0, alpha=2.0, theta=3.0),
        nb_lognormal=NBLogNormalModel(r=1.0, p=0.07, mu=-0.311, sigma=1.5),
    )
