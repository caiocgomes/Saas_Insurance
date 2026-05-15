"""Monte Carlo simulations for actuarial modeling of capped-usage AI subscriptions."""

from saas_actuaria.models import (
    PoissonGammaModel,
    NBLogNormalModel,
    MixedPopulationModel,
    simulate_portfolio,
)
from saas_actuaria.metrics import (
    var_alpha,
    tvar_alpha,
    reserve_required,
    loss_ratio,
)
from saas_actuaria.calibration import (
    default_scenario,
    heavy_consumption_scenario,
    vercel_scenario,
)

__all__ = [
    "PoissonGammaModel",
    "NBLogNormalModel",
    "MixedPopulationModel",
    "simulate_portfolio",
    "var_alpha",
    "tvar_alpha",
    "reserve_required",
    "loss_ratio",
    "default_scenario",
    "heavy_consumption_scenario",
    "vercel_scenario",
]
