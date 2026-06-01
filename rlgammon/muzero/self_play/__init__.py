"""Self-play package: the actors driving Stochastic MuZero search to produce trajectories.

:class:`SelfPlayActor` is the proven, single-game BASELINE actor (one batch-1 search per move) and
remains the default. :class:`BatchedSelfPlayActor` is the opt-in, performance-oriented feature actor
that plays many games in lockstep with a single batched Gumbel search per joint move; it is exported
here ALONGSIDE the baseline for A/B comparison, not as a replacement.
"""

from rlgammon.muzero.self_play.actor import SelfPlayActor
from rlgammon.muzero.self_play.batched_actor import BatchedSelfPlayActor

__all__ = [
    "BatchedSelfPlayActor",
    "SelfPlayActor",
]
