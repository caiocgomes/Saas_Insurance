"""Study: naive MLE bias under cap-induced right-censoring of LogNormal severity.

Goal: quantify the bias in (mu_hat, sigma_hat) when one fits a LogNormal to
observed per-user aggregate costs C_i = min(K, S_i^agg), ignoring the censoring
mechanism.

Procedure:
    1. Draw N users with S_i^agg from a compound NB-LogNormal generator.
    2. Apply cap K to get observed C_i.
    3. Fit a LogNormal to {C_i : C_i < K} (naive) and via Tobit-style MLE
       (censored likelihood). Compare both to the true (mu, sigma).
    4. Vary the fraction of users hitting the cap by scaling K.

Reports the percent bias in mu_hat and sigma_hat for the naive fit at three
cap-induced censoring levels (~5%, ~20%, ~40%). The 20% case is the one
referenced in §4.2 of the paper.

Run: uv run python studies/study_censoring_bias.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, norm


def simulate_severities(
    n: int,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n LogNormal severities. Treated as a single 'aggregate' per user."""
    return rng.lognormal(mu, sigma, size=n)


def naive_lognormal_fit(observed: np.ndarray, cap: float) -> tuple[float, float]:
    """Fit LogNormal to observed-and-uncensored sample only, ignoring cap.

    Excludes users that hit the cap (their value is exactly K and the fit
    cannot distinguish them from a draw at K). This is the most charitable
    naive fit; many practitioners do worse by including the capped users
    at their reported value of K.
    """
    uncensored = observed[observed < cap]
    if uncensored.size < 2:
        return float("nan"), float("nan")
    log_x = np.log(uncensored)
    return float(log_x.mean()), float(log_x.std(ddof=0))


def censored_lognormal_mle(observed: np.ndarray, cap: float) -> tuple[float, float]:
    """Tobit-style censored MLE for LogNormal with right-censoring at cap K.

    Likelihood contributions:
        - Uncensored x_i < K:  log f(x_i; mu, sigma) where f is LogNormal density.
        - Censored x_i = K:    log P(X > K; mu, sigma) = log S(K; mu, sigma).
    """
    log_cap = np.log(cap)
    uncensored = observed[observed < cap]
    n_censored = int((observed >= cap).sum())
    log_uncensored = np.log(uncensored) if uncensored.size > 0 else np.array([])

    def neg_log_lik(params: np.ndarray) -> float:
        mu, sigma = params[0], params[1]
        if sigma <= 1e-6:
            return 1e12
        # Uncensored: log(density) = log(1/(x*sigma*sqrt(2*pi))) - (log x - mu)^2/(2 sigma^2)
        # Equivalent to log of LogNormal pdf.
        if log_uncensored.size > 0:
            ll_uncensored = (
                -0.5 * np.log(2.0 * np.pi)
                - np.log(sigma)
                - log_uncensored
                - 0.5 * ((log_uncensored - mu) / sigma) ** 2
            ).sum()
        else:
            ll_uncensored = 0.0
        # Censored: log P(X > K) under LogNormal = log(1 - Phi((log K - mu)/sigma))
        z = (log_cap - mu) / sigma
        log_surv = norm.logsf(z)
        ll_censored = n_censored * log_surv
        return -(ll_uncensored + ll_censored)

    init_mu, init_sigma = naive_lognormal_fit(observed, cap)
    if np.isnan(init_mu):
        init_mu, init_sigma = 0.0, 1.0
    result = minimize(
        neg_log_lik,
        x0=np.array([init_mu, init_sigma]),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 5000},
    )
    return float(result.x[0]), float(result.x[1])


@dataclass
class StudyConfig:
    n_users: int = 50_000
    mu_true: float = 2.6
    sigma_true: float = 1.3
    seed: int = 20260515
    target_censoring_fractions: tuple[float, ...] = (0.05, 0.20, 0.40)


def find_cap_for_censoring(
    mu: float,
    sigma: float,
    target_fraction: float,
) -> float:
    """Closed-form cap such that P(X > K) = target_fraction under LogNormal(mu, sigma)."""
    z = norm.isf(target_fraction)  # inverse survival
    return float(np.exp(mu + sigma * z))


def run_study(config: StudyConfig) -> list[dict]:
    rng = np.random.default_rng(config.seed)
    X = simulate_severities(config.n_users, config.mu_true, config.sigma_true, rng)

    rows: list[dict] = []
    for q in config.target_censoring_fractions:
        cap = find_cap_for_censoring(config.mu_true, config.sigma_true, q)
        observed = np.minimum(X, cap)
        n_capped = int((X >= cap).sum())

        mu_naive, sigma_naive = naive_lognormal_fit(observed, cap)
        mu_mle, sigma_mle = censored_lognormal_mle(observed, cap)

        rows.append(
            dict(
                target_censoring=q,
                empirical_censoring=n_capped / config.n_users,
                cap=cap,
                mu_true=config.mu_true,
                sigma_true=config.sigma_true,
                mu_naive=mu_naive,
                sigma_naive=sigma_naive,
                mu_naive_bias_pct=100.0 * (mu_naive - config.mu_true) / config.mu_true,
                sigma_naive_bias_pct=100.0 * (sigma_naive - config.sigma_true) / config.sigma_true,
                mu_mle=mu_mle,
                sigma_mle=sigma_mle,
                mu_mle_bias_pct=100.0 * (mu_mle - config.mu_true) / config.mu_true,
                sigma_mle_bias_pct=100.0 * (sigma_mle - config.sigma_true) / config.sigma_true,
            )
        )
    return rows


def main() -> None:
    config = StudyConfig()
    rows = run_study(config)

    print(f"Censoring bias study: LogNormal(mu={config.mu_true}, sigma={config.sigma_true})")
    print(f"n = {config.n_users:,} users\n")
    header = (
        f"{'cens_target':>11} {'cens_emp':>10} "
        f"{'mu_naive':>9} {'sigma_naive':>12} {'mu_naive_bias%':>15} {'sigma_naive_bias%':>18} "
        f"{'mu_mle':>9} {'sigma_mle':>10} {'mu_mle_bias%':>14} {'sigma_mle_bias%':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['target_censoring']:>11.2f} {r['empirical_censoring']:>10.4f} "
            f"{r['mu_naive']:>9.4f} {r['sigma_naive']:>12.4f} "
            f"{r['mu_naive_bias_pct']:>14.2f}% {r['sigma_naive_bias_pct']:>17.2f}% "
            f"{r['mu_mle']:>9.4f} {r['sigma_mle']:>10.4f} "
            f"{r['mu_mle_bias_pct']:>13.2f}% {r['sigma_mle_bias_pct']:>15.2f}%"
        )

    # Headline numbers for §4.2 of the paper.
    print()
    target_20 = next(r for r in rows if r["target_censoring"] == 0.20)
    print(
        f"Headline for paper §4.2 (~20% censoring): "
        f"naive bias in mu_hat = {target_20['mu_naive_bias_pct']:.1f}%, "
        f"naive bias in sigma_hat = {target_20['sigma_naive_bias_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
