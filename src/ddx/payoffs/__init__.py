"""DDX payoff functions — vectorized, operating on per-interval CF arrays."""

from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.distress import distress_activated_floor, soft_duration_cover
from ddx.payoffs.stoploss import aggregate_stop_loss

__all__ = [
    "vanilla_floor",
    "distress_activated_floor",
    "soft_duration_cover",
    "aggregate_stop_loss",
]
