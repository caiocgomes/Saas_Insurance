"""Compound frequency-severity models for capped-usage subscriptions.

Implements two generators of per-user aggregate cost under a non-fungible cap:
    - Poisson-Gamma: N ~ Poisson(lambda), S ~ Gamma(alpha, scale=theta)
    - NB-LogNormal:  N ~ NegativeBinomial(r, p), log(S) ~ Normal(mu, sigma)

Per-user aggregate cost over a single reset period T:
    C_i = min(K, sum_{j=1..N_i} S_{ij})

The cap K is non-fungible and applied to the cumulative event-level cost within
the period. All randomness is driven by a single numpy Generator (see
np.random.default_rng) to preserve reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoissonGammaModel:
    """Compound Poisson with Gamma severity.

    Parameters
    ----------
    lam :
        Poisson frequency rate (mean events per user per period).
    alpha :
        Gamma shape parameter for severity.
    theta :
        Gamma scale parameter for severity. Mean severity is alpha * theta.
    """

    lam: float
    alpha: float
    theta: float

    def mean_severity(self) -> float:
        return self.alpha * self.theta

    def mean_aggregate_uncapped(self) -> float:
        return self.lam * self.mean_severity()

    def sample_user(self, rng: np.random.Generator, cap: float) -> float:
        """Sample one user's capped aggregate cost over the period."""
        n = rng.poisson(self.lam)
        if n == 0:
            return 0.0
        severities = rng.gamma(self.alpha, self.theta, size=n)
        return float(min(cap, severities.sum()))

    def sample_portfolio(
        self,
        rng: np.random.Generator,
        n_users: int,
        cap: float,
    ) -> np.ndarray:
        """Vectorized sampling of `n_users` capped aggregate costs."""
        counts = rng.poisson(self.lam, size=n_users)
        out = np.zeros(n_users, dtype=np.float64)
        total_events = int(counts.sum())
        if total_events == 0:
            return out
        all_severities = rng.gamma(self.alpha, self.theta, size=total_events)
        cursor = 0
        for i, k in enumerate(counts):
            if k == 0:
                continue
            agg = all_severities[cursor : cursor + k].sum()
            out[i] = min(cap, agg)
            cursor += k
        return out


@dataclass(frozen=True)
class NBLogNormalModel:
    """Negative Binomial frequency with LogNormal severity.

    Parameterization follows the (r, p) convention of scipy.stats.nbinom:
    N counts the number of "successes" (events) before observing r "failures",
    with success probability per trial p. Mean E[N] = r * (1 - p) / p,
    variance Var[N] = r * (1 - p) / p**2 (so var/mean = 1/p, overdispersed
    relative to Poisson for p < 1).

    Severity is parameterized so that log(S) ~ N(mu, sigma**2).
    """

    r: float
    p: float
    mu: float
    sigma: float

    def mean_frequency(self) -> float:
        return self.r * (1.0 - self.p) / self.p

    def var_frequency(self) -> float:
        return self.r * (1.0 - self.p) / (self.p**2)

    def mean_severity(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma**2))

    def mean_aggregate_uncapped(self) -> float:
        return self.mean_frequency() * self.mean_severity()

    def sample_user(self, rng: np.random.Generator, cap: float) -> float:
        n = rng.negative_binomial(self.r, self.p)
        if n == 0:
            return 0.0
        severities = rng.lognormal(self.mu, self.sigma, size=n)
        return float(min(cap, severities.sum()))

    def sample_portfolio(
        self,
        rng: np.random.Generator,
        n_users: int,
        cap: float,
    ) -> np.ndarray:
        counts = rng.negative_binomial(self.r, self.p, size=n_users)
        out = np.zeros(n_users, dtype=np.float64)
        total_events = int(counts.sum())
        if total_events == 0:
            return out
        all_severities = rng.lognormal(self.mu, self.sigma, size=total_events)
        cursor = 0
        for i, k in enumerate(counts):
            if k == 0:
                continue
            agg = all_severities[cursor : cursor + k].sum()
            out[i] = min(cap, agg)
            cursor += k
        return out


@dataclass(frozen=True)
class MixedPopulationModel:
    """Two-segment population: `pi_power` fraction of users follow `power`,
    the rest follow `light`. Each segment is an NB-LogNormal model.

    The intended use is to model adverse-selection at the tier level: power
    users self-select into higher tiers (Max 20x) where their conditional
    consumption distribution materially differs from a light user's.
    """

    pi_power: float
    light: NBLogNormalModel
    power: NBLogNormalModel

    def mean_aggregate_uncapped(self) -> float:
        return (
            (1.0 - self.pi_power) * self.light.mean_aggregate_uncapped()
            + self.pi_power * self.power.mean_aggregate_uncapped()
        )

    def sample_portfolio_with_segments(
        self,
        rng: np.random.Generator,
        n_users: int,
        cap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample n_users capped costs and return (costs, is_power_flag)."""
        is_power = rng.random(n_users) < self.pi_power
        out = np.empty(n_users, dtype=np.float64)
        light_idx = np.where(~is_power)[0]
        power_idx = np.where(is_power)[0]
        if light_idx.size > 0:
            out[light_idx] = self.light.sample_portfolio(rng, n_users=light_idx.size, cap=cap)
        if power_idx.size > 0:
            out[power_idx] = self.power.sample_portfolio(rng, n_users=power_idx.size, cap=cap)
        return out, is_power

    def sample_portfolio(
        self,
        rng: np.random.Generator,
        n_users: int,
        cap: float,
    ) -> np.ndarray:
        costs, _ = self.sample_portfolio_with_segments(rng, n_users, cap)
        return costs


def simulate_portfolio(
    model: PoissonGammaModel | NBLogNormalModel | MixedPopulationModel,
    n_users: int,
    cap: float,
    n_replications: int,
    seed: int,
) -> np.ndarray:
    """Run `n_replications` Monte Carlo replications of a portfolio with `n_users`.

    Returns an array of shape (n_replications,) containing aggregate portfolio
    cost L = sum_i C_i for each replication.
    """
    rng = np.random.default_rng(seed)
    aggregate = np.empty(n_replications, dtype=np.float64)
    for rep in range(n_replications):
        per_user = model.sample_portfolio(rng, n_users=n_users, cap=cap)
        aggregate[rep] = per_user.sum()
    return aggregate
