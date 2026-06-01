"""Package implementing the Stochastic MuZero Monte-Carlo tree search over the learned model.

:class:`StochasticMuZeroMCTS` is the proven, single-tree BASELINE search (pUCT + Dirichlet noise) and
remains the default everywhere. :class:`BatchedGumbelMCTS` (with its :class:`GumbelRootResult`) is the
opt-in, performance-oriented feature search -- it advances many trees in lockstep and uses Gumbel
MuZero root selection -- exported here ALONGSIDE the baseline for A/B comparison, not as a replacement.
"""

from rlgammon.muzero.mcts.batched_search import BatchedGumbelMCTS, GumbelRootResult
from rlgammon.muzero.mcts.node import ChanceNode, DecisionNode, MinMaxStats
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS

__all__ = [
    "BatchedGumbelMCTS",
    "ChanceNode",
    "DecisionNode",
    "GumbelRootResult",
    "MinMaxStats",
    "StochasticMuZeroMCTS",
]
