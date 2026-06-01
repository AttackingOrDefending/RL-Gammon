"""Reusable evaluator pitting a trained move-policy against a uniform-random opponent.

This module plays a number of full backgammon games on the real OpenSpiel engine, alternating which
colour the trained agent controls, and reports the agent's win-rate and average points (the mean of
its signed terminal return, which lies in ``{-3, -2, -1, +1, +2, +3}``). Two policies are supported: a
:class:`~rlgammon.agents.td_agent.TDAgent` (via its 1-ply :meth:`~rlgammon.agents.td_agent.TDAgent.choose_move`)
and a Stochastic MuZero network (via a fresh, noise-free MCTS at every decision node). A small
``main`` loads a saved model from disk and prints the evaluation result.
"""
import argparse
from collections.abc import Callable
import pathlib

import numpy as np
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    board_features,
    create_game,
)
from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.muzero.mcts.batched_search import BatchedGumbelMCTS
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS
from rlgammon.muzero.muzero_factory import build_network, resolve_device
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.rlgammon_types import BLACK, WHITE

# Number of player colours; the evaluator alternates the trained agent's colour game by game.
NUM_COLOURS = 2
# Default number of evaluation games played by ``main``.
DEFAULT_GAMES = 200
# Default number of MuZero search simulations per move during evaluation.
DEFAULT_EVAL_SIMS = 50
# Default MuZero observation size (board features only, dice dropped).
MUZERO_OBSERVATION_SIZE = 198
# Default MuZero action-space size for the OpenSpiel backgammon engine.
MUZERO_NUM_ACTIONS = 1352
# Default latent state width of a saved MuZero network (matches ``scripts/train_muzero.py``).
MUZERO_STATE_CHANNELS = 256
# Default hidden-layer widths of a saved MuZero network (matches ``scripts/train_muzero.py``).
MUZERO_HIDDEN_SIZES = (256, 256)
# Default codebook size of a saved MuZero network (matches ``scripts/train_muzero.py``).
MUZERO_CODEBOOK_SIZE = 32
# Default categorical value/reward support size of a saved MuZero network.
MUZERO_SUPPORT_SIZE = 21
# A strictly positive terminal return counts as a win for the scored colour.
WIN_THRESHOLD = 0.0
# Gumbel considered root actions used by the batched evaluation search (the root keeps all legals).
MUZERO_EVAL_CONSIDERED = 16
# Hard cap on the number of joint decision rounds in a batched evaluation, guarding non-termination.
MAX_EVAL_MOVES = 2000
# Default MuZero evaluation search: the batched Gumbel FEATURE search (preserves current behaviour).
# Pass ``mcts="baseline"`` to instead evaluate with the proven single-tree pUCT search.
DEFAULT_EVAL_MCTS = "gumbel"

# A move policy maps the legal actions and the current state to the chosen action id.
MovePolicy = Callable[[list[int], GameState], int]


def _default_device() -> str:
    """
    Pick the default evaluation device: ``"cuda"`` when a CUDA GPU is present, else ``"cpu"``.

    :return: ``"cuda"`` if :func:`torch.cuda.is_available` is ``True``, otherwise ``"cpu"``
    """
    return "cuda" if th.cuda.is_available() else "cpu"


def _play_one_game(game: BackgammonGame, agent_policy: MovePolicy, agent_color: int,
                   rng: np.random.Generator) -> float:
    """
    Play one full game of ``agent_policy`` vs a uniform-random opponent and return the agent's return.

    Chance nodes (the dice rolls) are resolved on the real engine by sampling an outcome by its
    probability; at each decision node the side to move is asked for an action (the trained policy when
    it is the agent's colour, a uniform-random choice otherwise).

    :param game: the game factory producing a fresh initial state
    :param agent_policy: the trained move policy under evaluation
    :param agent_color: the colour the trained policy controls (WHITE=0, BLACK=1)
    :param rng: the random number generator driving chance sampling and the random opponent
    :return: the trained agent's signed terminal return from ``agent_color``'s perspective
    """
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            apply_sampled_chance(state, rng)
            continue
        legal_actions = state.legal_actions()
        if state.current_player() == agent_color:
            action = agent_policy(legal_actions, state)
        else:
            action = int(rng.choice(legal_actions))
        state.apply_action(action)
    return float(state.returns()[agent_color])


def _evaluate_policy(agent_policy: MovePolicy, n_games: int,
                     rng: np.random.Generator) -> dict[str, float]:
    """
    Play ``n_games`` games of a move policy vs random, alternating colours, and aggregate the result.

    The trained policy plays WHITE on even game indices and BLACK on odd ones, so the report is not
    biased by any first-move advantage.

    :param agent_policy: the trained move policy under evaluation
    :param n_games: the number of games to play
    :param rng: the random number generator driving chance sampling and the random opponent
    :return: a dict with ``win_rate``, ``avg_points`` and ``games`` for the trained policy
    """
    game = create_game(PossibleEngine.OPEN_SPIEL)
    wins = 0
    total_points = 0.0
    for game_index in range(n_games):
        agent_color = WHITE if game_index % NUM_COLOURS == 0 else BLACK
        result = _play_one_game(game, agent_policy, agent_color, rng)
        total_points += result
        if result > WIN_THRESHOLD:
            wins += 1
    return {
        "win_rate": wins / n_games if n_games else 0.0,
        "avg_points": total_points / n_games if n_games else 0.0,
        "games": float(n_games),
    }


def play_td_vs_random(agent: TDAgent, n_games: int, rng: np.random.Generator) -> dict[str, float]:
    """
    Evaluate a TD agent against a uniform-random opponent over ``n_games`` games.

    :param agent: the trained TD agent whose :meth:`~rlgammon.agents.td_agent.TDAgent.choose_move` is used
    :param n_games: the number of games to play (colours alternate game by game)
    :param rng: the random number generator driving chance sampling and the random opponent
    :return: a dict with ``win_rate``, ``avg_points`` and ``games`` for the TD agent
    """
    return _evaluate_policy(agent.choose_move, n_games, rng)


def _baseline_muzero_policy(eval_config: MuZeroConfig, network: StochasticMuZeroNetwork,
                            rng: np.random.Generator) -> MovePolicy:
    """
    Build a single-tree BASELINE MuZero move policy for the one-game-at-a-time evaluation path.

    The returned policy runs a noise-free :class:`StochasticMuZeroMCTS` at the given state and returns
    the most-visited root action (greedy evaluation, mirroring the Gumbel path's greedy argmax).

    :param eval_config: the evaluation configuration (its ``num_simulations`` drives the search budget)
    :param network: the learned network driving the single-tree search
    :param rng: the random number generator the search is seeded with
    :return: a :data:`MovePolicy` choosing the most-visited action at each decision node
    """
    search = StochasticMuZeroMCTS(eval_config, network, rng)

    def policy(legal_actions: list[int], state: GameState) -> int:
        observation = th.tensor(
            board_features(state, state.current_player()), dtype=th.float32, device=network.device,
        ).unsqueeze(0)
        visit_counts = search.run(observation, legal_actions, add_exploration_noise=False)
        return max(visit_counts, key=lambda action: visit_counts[action])

    return policy


def play_muzero_vs_random(network: StochasticMuZeroNetwork, config: MuZeroConfig, n_games: int,
                          rng: np.random.Generator, num_simulations: int, *,
                          mcts: str = DEFAULT_EVAL_MCTS) -> dict[str, float]:
    """
    Evaluate a Stochastic MuZero network against a uniform-random opponent over ``n_games`` games.

    With the default ``mcts="gumbel"`` all ``n_games`` games are advanced in LOCKSTEP: at each joint
    step the games where the MuZero player is to move have their searches batched into a single
    :class:`BatchedGumbelMCTS` call (so every network inference batches across games on the GPU instead
    of running one batch-1 search per move), the chosen Gumbel-argmax action is played, the random
    opponent picks uniformly, and chance nodes are resolved on the real engine. Passing
    ``mcts="baseline"`` instead evaluates each game one move at a time with the proven single-tree
    :class:`StochasticMuZeroMCTS` (noise-free, greedy on the visit counts). Either way the MuZero agent
    plays WHITE on even game indices and BLACK on odd ones so the win-rate is unbiased.

    :param network: the learned Stochastic MuZero network driving the search
    :param config: the configuration the network was built with (its ``num_simulations`` is overridden)
    :param n_games: the number of games to play (colours alternate game by game)
    :param rng: the random number generator driving the search, chance sampling and the opponent
    :param num_simulations: the number of search simulations to run per evaluation move
    :param mcts: ``"gumbel"`` for the batched Gumbel feature search (default) or ``"baseline"`` for the
        single-tree pUCT search
    :return: a dict with ``win_rate``, ``avg_points`` and ``games`` for the MuZero network
    """
    if n_games <= 0:
        return {"win_rate": 0.0, "avg_points": 0.0, "games": 0.0}

    eval_config = MuZeroConfig(**{**config.__dict__, "num_simulations": num_simulations})
    if mcts == "baseline":
        return _evaluate_policy(_baseline_muzero_policy(eval_config, network, rng), n_games, rng)

    considered = min(MUZERO_EVAL_CONSIDERED, eval_config.num_actions)
    mcts_search = BatchedGumbelMCTS(eval_config, network, rng, num_considered=considered)
    game = create_game(PossibleEngine.OPEN_SPIEL)

    states = [game.new_initial_state() for _ in range(n_games)]
    agent_colors = [WHITE if index % NUM_COLOURS == 0 else BLACK for index in range(n_games)]
    for state in states:
        if state.is_chance_node():
            apply_sampled_chance(state, rng)

    for _ in range(MAX_EVAL_MOVES):
        if all(state.is_terminal() for state in states):
            break
        _advance_eval_round(states, agent_colors, mcts_search, rng, network.device)

    wins = 0
    total_points = 0.0
    for state, agent_color in zip(states, agent_colors, strict=True):
        points = float(state.returns()[agent_color]) if state.is_terminal() else 0.0
        total_points += points
        if points > WIN_THRESHOLD:
            wins += 1
    return {"win_rate": wins / n_games, "avg_points": total_points / n_games, "games": float(n_games)}


def _advance_eval_round(states: list[GameState], agent_colors: list[int], mcts: BatchedGumbelMCTS,
                        rng: np.random.Generator, device: th.device) -> None:
    """
    Advance every non-terminal evaluation game by one decision, batching the MuZero searches.

    Games where the MuZero player is to move have their root searches batched into one
    :class:`BatchedGumbelMCTS` call and play the Gumbel-argmax action; games where the opponent is to
    move play a uniform-random legal action. Any chance node produced is resolved on the real engine.

    :param states: the per-game states (mutated in place)
    :param agent_colors: the colour the MuZero agent controls in each game
    :param mcts: the batched Gumbel search shared across the agent-to-move games this round
    :param rng: the random number generator driving the random opponent and chance sampling
    :param device: the torch device the batched root observations are built on
    """
    agent_indices = []
    opponent_indices = []
    for index, state in enumerate(states):
        if state.is_terminal():
            continue
        if state.current_player() == agent_colors[index]:
            agent_indices.append(index)
        else:
            opponent_indices.append(index)

    if agent_indices:
        observations = th.tensor(
            [board_features(states[index], agent_colors[index]) for index in agent_indices],
            dtype=th.float32, device=device,
        )
        legal_actions = [states[index].legal_actions() for index in agent_indices]
        results = mcts.run_batch(observations, legal_actions)
        for index, result in zip(agent_indices, results, strict=True):
            _apply_eval_action(states[index], result.action, rng)

    for index in opponent_indices:
        action = int(rng.choice(states[index].legal_actions()))
        _apply_eval_action(states[index], action, rng)


def _apply_eval_action(state: GameState, action: int, rng: np.random.Generator) -> None:
    """
    Apply a decision action to an evaluation game and resolve any following chance node.

    :param state: the decision-node game state (mutated in place)
    :param action: the action id to apply
    :param rng: the random number generator used to resolve a following chance node
    """
    state.apply_action(action)
    if not state.is_terminal() and state.is_chance_node():
        apply_sampled_chance(state, rng)


def _evaluate_td(model_path: str, n_games: int, rng: np.random.Generator) -> dict[str, float]:
    """
    Load a saved TD model from ``saved_agents`` and evaluate it against random.

    :param model_path: the file name (within ``rlgammon/agents/saved_agents``) of the saved TD model
    :param n_games: the number of evaluation games to play
    :param rng: the random number generator driving the evaluation
    :return: the evaluation result dict for the TD agent
    """
    agent = TDAgent(pre_made_model_file_name=model_path)
    return play_td_vs_random(agent, n_games, rng)


def _evaluate_muzero(model_path: str, n_games: int, num_simulations: int, rng: np.random.Generator, *,
                     state_channels: int, hidden_sizes: tuple[int, ...],
                     codebook_size: int, device: str,
                     mcts: str = DEFAULT_EVAL_MCTS) -> dict[str, float]:
    """
    Load a saved MuZero state dict from ``model_path`` and evaluate it against random.

    The network architecture must match the one the checkpoint was trained with (a state dict carries
    no architecture), so the latent width, hidden sizes and codebook size are passed in; their defaults
    match ``scripts/train_muzero.py``. The checkpoint is mapped onto ``device`` so a CUDA-trained model
    can be evaluated on CPU and vice versa.

    :param model_path: the absolute or relative path to the saved network state dict
    :param n_games: the number of evaluation games to play
    :param num_simulations: the number of search simulations per evaluation move
    :param rng: the random number generator driving the evaluation
    :param state_channels: the latent state width the checkpoint was trained with
    :param hidden_sizes: the hidden-layer widths the checkpoint was trained with
    :param codebook_size: the chance codebook size the checkpoint was trained with
    :param device: the torch device the network and search tensors live on (``"cpu"`` or ``"cuda"``)
    :param mcts: the evaluation search, ``"gumbel"`` (default) or ``"baseline"`` (single-tree pUCT)
    :return: the evaluation result dict for the MuZero network
    """
    config = MuZeroConfig(
        observation_size=MUZERO_OBSERVATION_SIZE,
        num_actions=MUZERO_NUM_ACTIONS,
        state_channels=state_channels,
        hidden_sizes=hidden_sizes,
        codebook_size=codebook_size,
        num_simulations=num_simulations,
        value_support_size=MUZERO_SUPPORT_SIZE,
        reward_support_size=MUZERO_SUPPORT_SIZE,
        device=device,
    )
    network = build_network(config)
    state_dict = th.load(pathlib.Path(model_path), map_location=network.device, weights_only=True)
    network.load_state_dict(state_dict)
    return play_muzero_vs_random(network, config, n_games, rng, num_simulations, mcts=mcts)


def main() -> None:
    """Parse the command-line arguments, load the requested model and print its evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent against a random opponent.")
    parser.add_argument("--agent", choices=["td", "muzero"], required=True, help="which agent to evaluate")
    parser.add_argument("--model", type=str, required=True, help="path/filename of the saved model")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="number of evaluation games")
    parser.add_argument("--sims", type=int, default=DEFAULT_EVAL_SIMS, help="MuZero search simulations per move")
    parser.add_argument("--seed", type=int, default=0, help="seed for the evaluation random number generator")
    parser.add_argument(
        "--state-channels", type=int, default=MUZERO_STATE_CHANNELS,
        help="latent state width the MuZero checkpoint was trained with",
    )
    parser.add_argument(
        "--hidden", type=int, default=MUZERO_HIDDEN_SIZES[0],
        help="hidden-layer width (applied to both layers) the MuZero checkpoint was trained with",
    )
    parser.add_argument(
        "--codebook-size", type=int, default=MUZERO_CODEBOOK_SIZE,
        help="chance codebook size the MuZero checkpoint was trained with",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=_default_device(),
        help="torch device for MuZero inference (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--mcts", choices=["gumbel", "baseline"], default=DEFAULT_EVAL_MCTS,
        help="MuZero evaluation search: batched Gumbel (default) or baseline single-tree pUCT",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.agent == "td":
        result = _evaluate_td(args.model, args.games, rng)
    else:
        result = _evaluate_muzero(
            args.model, args.games, args.sims, rng,
            state_channels=args.state_channels,
            hidden_sizes=(args.hidden, args.hidden),
            codebook_size=args.codebook_size,
            device=resolve_device(args.device),
            mcts=args.mcts,
        )

    print(
        f"agent={args.agent} games={int(result['games'])} "
        f"win_rate={result['win_rate']:.4f} avg_points={result['avg_points']:.4f}",
    )


if __name__ == "__main__":
    main()
