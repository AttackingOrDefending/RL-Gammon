"""Small factory helpers assembling the Stochastic MuZero network and search from a config.

:func:`build_mcts` returns the proven, single-tree BASELINE search and is the default everywhere.
:func:`build_batched_gumbel_mcts` and :func:`build_batched_actor` build the opt-in, performance
features (batched Gumbel search and batched self-play); they sit ALONGSIDE the baseline for A/B
comparison and never replace it.
"""
import numpy as np
import torch as th

from rlgammon.game.backgammon_protocol import BackgammonGame
from rlgammon.muzero.mcts.batched_search import BatchedGumbelMCTS
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.self_play.batched_actor import BatchedSelfPlayActor

# Default number ``m`` of root actions Gumbel considers for the opt-in batched feature search/actor;
# matches the training scripts' default. It is always capped at the number of legal actions.
DEFAULT_NUM_CONSIDERED = 16
# Default number ``K`` of games the opt-in batched self-play actor advances in lockstep per move.
DEFAULT_NUM_PARALLEL = 32


def resolve_device(requested: str) -> str:
    """
    Resolve a requested device string to one that is actually usable, guarding CUDA availability.

    ``"cuda"`` is only honoured when :func:`torch.cuda.is_available` returns ``True``; otherwise a note
    is printed and the device falls back to ``"cpu"`` so a run never crashes for lack of a GPU. Any
    other value (e.g. ``"cpu"``) is returned unchanged.

    :param requested: the device the caller asked for (e.g. ``"cuda"`` or ``"cpu"``)
    :return: the usable device string (``"cuda"`` only if a CUDA device is present, else ``"cpu"``)
    """
    if requested == "cuda" and not th.cuda.is_available():
        print("[device] cuda requested but torch.cuda.is_available() is False; falling back to cpu")
        return "cpu"
    return requested


def build_network(config: MuZeroConfig) -> StochasticMuZeroNetwork:
    """
    Build a Stochastic MuZero network from a configuration and move it onto the configured device.

    The network constructor already moves every parameter onto ``config.device``; the explicit
    ``.to`` here documents that the returned network is ready on that device (``"cpu"`` by default,
    ``"cuda"`` for GPU acceleration).

    :param config: the configuration providing every network dimension and the target device
    :return: the assembled :class:`StochasticMuZeroNetwork` on ``config.device``
    """
    return StochasticMuZeroNetwork(config).to(th.device(config.device))


def build_mcts(config: MuZeroConfig, network: StochasticMuZeroNetwork,
               rng: np.random.Generator | None = None) -> StochasticMuZeroMCTS:
    """
    Build the BASELINE Stochastic MuZero search bound to a configuration and a network.

    This is the proven, single-tree pUCT + Dirichlet search and the DEFAULT search throughout the
    package. For the opt-in batched Gumbel feature search use :func:`build_batched_gumbel_mcts`.

    :param config: the configuration providing the search settings
    :param network: the learned network driving the search
    :param rng: the random number generator for exploration noise (defaults to a config-seeded one)
    :return: the assembled baseline :class:`StochasticMuZeroMCTS`
    """
    return StochasticMuZeroMCTS(config, network, rng)


def build_batched_gumbel_mcts(config: MuZeroConfig, network: StochasticMuZeroNetwork,
                              rng: np.random.Generator | None = None, *,
                              num_considered: int = DEFAULT_NUM_CONSIDERED) -> BatchedGumbelMCTS:
    """
    Build the OPT-IN batched Gumbel Stochastic-MuZero search bound to a configuration and a network.

    This is the performance-oriented FEATURE search (many trees advanced in lockstep, Gumbel-top-k +
    sequential-halving root selection). It is NOT the default and does not replace :func:`build_mcts`;
    it is provided alongside the baseline for A/B comparison. The ``num_considered`` width is capped at
    the number of legal actions of each searched root.

    :param config: the configuration providing the pUCT, simulation and discount settings
    :param network: the learned network driving the batched search
    :param rng: the random number generator for the per-root Gumbel noise (defaults to config-seeded)
    :param num_considered: the number ``m`` of root actions Gumbel considers (capped at the legals)
    :return: the assembled feature :class:`BatchedGumbelMCTS`
    """
    generator = rng if rng is not None else np.random.default_rng(config.seed)
    return BatchedGumbelMCTS(config, network, generator, num_considered=num_considered)


def build_batched_actor(config: MuZeroConfig, game: BackgammonGame,
                        network: StochasticMuZeroNetwork, rng: np.random.Generator, *,
                        num_parallel: int = DEFAULT_NUM_PARALLEL,
                        num_considered: int = DEFAULT_NUM_CONSIDERED) -> BatchedSelfPlayActor:
    """
    Build the OPT-IN batched self-play actor bound to a configuration, game, network and generator.

    This is the performance-oriented FEATURE self-play path (``num_parallel`` real games advanced in
    lockstep with one batched Gumbel search per joint move). It is NOT the default; the baseline
    single-game :class:`~rlgammon.muzero.self_play.actor.SelfPlayActor` remains the default self-play
    actor. Provided alongside the baseline for A/B comparison.

    :param config: the configuration shared with the batched search (simulations, discount, ...)
    :param game: the game factory producing fresh initial states
    :param network: the learned network driving the batched search
    :param rng: the random number generator for chance sampling and the Gumbel noise
    :param num_parallel: the number ``K`` of games advanced simultaneously per batched search
    :param num_considered: the number ``m`` of root actions Gumbel considers per move (capped at legals)
    :return: the assembled feature :class:`BatchedSelfPlayActor`
    """
    return BatchedSelfPlayActor(
        config, game, network, rng, num_parallel=num_parallel, num_considered=num_considered,
    )
