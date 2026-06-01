"""Train a TD-Gammon agent by self-play and evaluate it in money cube games or matches.

OpenSpiel backgammon has no doubling cube, so the value network is trained cubelessly (exactly the
self-play TD(lambda) loop of :mod:`scripts.train_td`); the analytic cube is layered on only at
evaluation time. The ``--use-cube`` flag selects whether the evaluation plays cube games/matches
(honouring the agent's ``should_double`` / ``should_take``) or the plain cubeless games, so
``--use-cube`` off reproduces the cubeless evaluation as a regression guard. With ``--match-length``
greater than zero the evaluation plays matches and reports the match-winning chance; otherwise it
plays money games and reports win-rate and points-per-game. ``--eval-baseline`` pits the trained
agent against a never-double / always-take money baseline instead of a second trained agent.
"""
import argparse
import time
import uuid

import numpy as np
import torch as th

from rlgammon.agents.td_agent import TDAgent
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    board_features,
    create_game,
)
from rlgammon.models.model_types import ValueHead
from rlgammon.rlgammon_types import BLACK, WHITE
from rlgammon.trainer.testing.cube_testing import CubeTesting, NoDoubleMoneyTaker

# Default number of self-play training episodes.
DEFAULT_EPISODES = 200
# Default number of evaluation games (money) or matches (match play).
DEFAULT_GAMES = 50
# Default hidden-layer width of the value network.
DEFAULT_HIDDEN = 128
# Default learning rate.
DEFAULT_LR = 0.1
# Default TD(lambda) trace-decay parameter.
DEFAULT_LAMDA = 0.7
# Default random seed.
DEFAULT_SEED = 0
# A match length of zero selects money play instead of match play.
MONEY_MATCH_LENGTH = 0


def train_agent(*, episodes: int, hidden: int, lr: float, lamda: float, seed: int) -> TDAgent:
    """
    Train a TD-Gammon agent by self-play TD(lambda) and return it in memory.

    This mirrors :func:`scripts.train_td.train` (undiscounted TD(lambda), WHITE-perspective bootstrap,
    dice sampled at chance nodes) but keeps the trained agent rather than only saving it.

    :param episodes: the number of self-play training episodes to run
    :param hidden: the hidden-layer width of the value network
    :param lr: the learning rate
    :param lamda: the TD(lambda) trace-decay parameter
    :param seed: the seed for the agent and the self-play random number generator
    :return: the trained TD agent
    """
    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(seed)
    agent = TDAgent(lr=lr, lamda=lamda, hidden=hidden, value_head=ValueHead.EQUITY_SIGMOID, seed=seed)
    for _episode in range(episodes):
        agent.episode_setup()
        state = game.new_initial_state()
        apply_sampled_chance(state, rng)
        while not state.is_terminal():
            p = agent.evaluate_position(board_features(state, WHITE))
            action = agent.choose_move(state.legal_actions(), state)
            state.apply_action(action)
            if state.is_terminal():
                reward = th.tensor(state.returns()[WHITE], dtype=p.dtype)
                _ = agent.train(p, reward)
            else:
                if state.is_chance_node():
                    apply_sampled_chance(state, rng)
                p_next = agent.evaluate_position(board_features(state, WHITE))
                _ = agent.train(p, p_next)
    return agent


def evaluate(agent: TDAgent, *, games: int, match_length: int, seed: int, use_cube: bool,
             eval_baseline: bool) -> dict[str, float]:
    """
    Evaluate a trained agent in money cube games or matches against a second agent or a baseline.

    :param agent: the trained TD agent under evaluation (controls WHITE)
    :param games: the number of money games (money play) or matches (match play) to play
    :param match_length: the match length in points, or 0 for money play
    :param seed: the seed for the evaluation random number generator
    :param use_cube: whether doubling is active (off reproduces the cubeless result)
    :param eval_baseline: whether the opponent is the never-double / always-take money baseline
    :return: the evaluation result dictionary from the cube-testing harness
    """
    # Either a never-double / always-take baseline (sharing the agent's move policy) or the agent
    # itself as the second seat (cube self-play between two copies of the same trained policy).
    opponent: TDAgent | NoDoubleMoneyTaker = NoDoubleMoneyTaker(agent) if eval_baseline else agent
    agents = {WHITE: agent, BLACK: opponent}
    harness = CubeTesting()
    rng = np.random.default_rng(seed)
    if match_length > MONEY_MATCH_LENGTH:
        return harness.play_matches(agents, match_length, games, rng, use_cube=use_cube)
    return harness.play_money_games(agents, games, rng, use_cube=use_cube)


def run(*, episodes: int, games: int, match_length: int, hidden: int, lr: float, lamda: float,
        seed: int, use_cube: bool, eval_baseline: bool, load: str | None, save: bool) -> dict[str, float]:
    """
    Train (or load) a TD agent and evaluate it in cube games or matches.

    :param episodes: the number of self-play training episodes (ignored when ``load`` is given)
    :param games: the number of evaluation money games or matches
    :param match_length: the match length in points, or 0 for money play
    :param hidden: the hidden-layer width of the value network
    :param lr: the learning rate
    :param lamda: the TD(lambda) trace-decay parameter
    :param seed: the seed for training and evaluation
    :param use_cube: whether doubling is active during evaluation
    :param eval_baseline: whether to evaluate against the never-double / always-take baseline
    :param load: an optional saved-model file name to load instead of training
    :param save: whether to save the trained model under ``rlgammon/agents/saved_agents``
    :return: the evaluation result dictionary
    """
    start = time.time()
    if load is not None:
        agent = TDAgent(pre_made_model_file_name=load)
        print(f"[cube] loaded model {load}")
    else:
        agent = train_agent(episodes=episodes, hidden=hidden, lr=lr, lamda=lamda, seed=seed)
        print(f"[cube] trained {episodes} self-play episodes in {time.time() - start:.1f}s")
        if save:
            agent.save(uuid.uuid4(), episodes, main_filename="cube-td-backgammon")

    result = evaluate(agent, games=games, match_length=match_length, seed=seed, use_cube=use_cube,
                      eval_baseline=eval_baseline)
    mode = f"match(to {match_length})" if match_length > MONEY_MATCH_LENGTH else "money"
    print(
        f"[cube] mode={mode} use_cube={use_cube} baseline={eval_baseline} "
        f"win_rate={result['win_rate']:.4f} ppg={result['ppg']:.4f} mwc={result['mwc']:.4f} "
        f"doubles={result['doubles']:.0f} takes={result['takes']:.2f} passes={result['passes']:.2f} "
        f"mean_cube_turns={result['mean_cube_turns']:.2f}",
    )
    return result


def main() -> None:
    """Parse the command-line arguments and run cube self-play training plus evaluation."""
    parser = argparse.ArgumentParser(
        description="Train a TD-Gammon agent by self-play and evaluate it with or without the cube.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="self-play training episodes")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="evaluation games or matches")
    parser.add_argument("--match-length", type=int, default=MONEY_MATCH_LENGTH,
                        help="match length in points (0 = money play)")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="hidden-layer width")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--lamda", type=float, default=DEFAULT_LAMDA, help="TD(lambda) trace-decay parameter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--use-cube", action="store_true", help="enable doubling in the evaluation")
    parser.add_argument("--eval-baseline", action="store_true",
                        help="evaluate against a never-double / always-take baseline")
    parser.add_argument("--load", type=str, default=None, help="saved-model file name to load instead of training")
    parser.add_argument("--save", action="store_true", help="save the trained model after training")
    args = parser.parse_args()

    run(
        episodes=args.episodes,
        games=args.games,
        match_length=args.match_length,
        hidden=args.hidden,
        lr=args.lr,
        lamda=args.lamda,
        seed=args.seed,
        use_cube=args.use_cube,
        eval_baseline=args.eval_baseline,
        load=args.load,
        save=args.save,
    )


if __name__ == "__main__":
    main()
