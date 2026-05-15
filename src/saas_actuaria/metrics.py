"""Tail-risk metrics for portfolio aggregate cost distributions.

VaR and TVaR follow the actuarial convention used in Klugman-Panjer-Willmot:
VaR_{1-alpha}(L) is the (1-alpha)-quantile of the loss distribution; TVaR_{1-alpha}(L)
is the conditional mean of L above that quantile. Both are computed empirically
from a Monte Carlo sample.
"""

from __future__ import annotations

import numpy as np


def var_alpha(losses: np.ndarray, alpha: float) -> float:
    """Empirical Value-at-Risk at confidence (1 - alpha).

    Parameters
    ----------
    losses :
        Sample of portfolio aggregate losses (shape (n,)).
    alpha :
        Tail probability. Conventional values: 0.01, 0.005, 0.001.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(np.quantile(losses, 1.0 - alpha))


def tvar_alpha(losses: np.ndarray, alpha: float) -> float:
    """Empirical Tail-Value-at-Risk: E[L | L > VaR_{1-alpha}(L)]."""
    threshold = var_alpha(losses, alpha)
    tail = losses[losses > threshold]
    if tail.size == 0:
        # Degenerate case: empty tail (small sample). Fall back to VaR.
        return threshold
    return float(tail.mean())


def reserve_required(
    losses: np.ndarray,
    premium_income: float,
    alpha: float,
    measure: str = "tvar",
) -> float:
    """Reserve R such that P(L > premium_income + R) <= alpha.

    Uses VaR or TVaR as the tail measure. Returns max(0, measure(L) - premium_income).
    """
    if measure == "var":
        tail_quantity = var_alpha(losses, alpha)
    elif measure == "tvar":
        tail_quantity = tvar_alpha(losses, alpha)
    else:
        raise ValueError("measure must be 'var' or 'tvar'")
    return max(0.0, tail_quantity - premium_income)


def loss_ratio(losses: np.ndarray, premium_income: float) -> float:
    """Mean loss ratio E[L] / premium_income across Monte Carlo replications."""
    return float(losses.mean() / premium_income)
