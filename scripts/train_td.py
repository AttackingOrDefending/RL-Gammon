"""Self-play TD(lambda) training for the TD-Gammon value agent until it beats a random opponent.

This mirrors the reference self-play loop in :class:`~rlgammon.trainer.step_trainer.StepTrainer`: every
episode is a full game played from a seeded :class:`numpy.random.Generator`, features are always taken
from the WHITE perspective to keep the bootstrap consistent, the agent is trained undiscounted at every
step (bootstrapping on the next afterstate, or on the true WHITE-centric terminal return at the end),
and chance nodes are resolved by sampling a dice outcome by its probability. Periodically the agent is
evaluated against a uniform-random opponent via
:func:`~scripts.eval_vs_random.play_td_vs_random` and the win-rate is printed; the final model is saved
under ``rlgammon/agents/saved_agents``.
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
from rlgammon.rlgammon_types import WHITE
from scripts.eval_vs_random import play_td_vs_random

# Default number of self-play training episodes.
DEFAULT_EPISODES = 2000
# Default cadence (in episodes) at which to evaluate against the random opponent.
DEFAULT_EVAL_EVERY = 250
# Default number of evaluation games per evaluation.
DEFAULT_EVAL_GAMES = 200
# Default hidden-layer width of the value network.
DEFAULT_HIDDEN = 128
# Default learning rate.
DEFAULT_LR = 0.1
# Default TD(lambda) trace-decay parameter.
DEFAULT_LAMDA = 0.7
# Default random seed.
DEFAULT_SEED = 0


def train(*, episodes: int, eval_every: int, eval_games: int, hidden: int, lr: float,
          lamda: float, seed: int, out: str) -> dict[str, float]:
    """
    Run self-play TD(lambda) training and periodically evaluate against a random opponent.

    :param episodes: the number of self-play training episodes to run
    :param eval_every: evaluate against random every this many episodes
    :param eval_games: the number of games per periodic evaluation
    :param hidden: the hidden-layer width of the value network
    :param lr: the learning rate
    :param lamda: the TD(lambda) trace-decay parameter
    :param seed: the seed for the agent and the self-play random number generator
    :param out: the base file name to save the final model under
    :return: the final evaluation result dict (``win_rate``, ``avg_points``, ``games``)
    """
    session_id = uuid.uuid4()
    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(seed)
    agent = TDAgent(lr=lr, lamda=lamda, hidden=hidden, value_head=ValueHead.EQUITY_SIGMOID, seed=seed)

    start = time.time()
    final_result: dict[str, float] = {"win_rate": 0.0, "avg_points": 0.0, "games": 0.0}
    for episode in range(1, episodes + 1):
        agent.episode_setup()
        state = game.new_initial_state()
        apply_sampled_chance(state, rng)

        while not state.is_terminal():
            # Always evaluate from the WHITE perspective to keep the bootstrap consistent.
            features = board_features(state, WHITE)
            p = agent.evaluate_position(features)

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

        if episode % eval_every == 0 or episode == episodes:
            final_result = play_td_vs_random(agent, eval_games, np.random.default_rng(seed + episode))
            elapsed = time.time() - start
            print(
                f"[td] episode {episode}/{episodes} "
                f"win_rate={final_result['win_rate']:.4f} avg_points={final_result['avg_points']:.4f} "
                f"elapsed={elapsed:.1f}s",
            )

    agent.save(session_id, episodes, main_filename=out)
    print(f"[td] saved model rlgammon/agents/saved_agents/{out}-{session_id}-({episodes}).pt")
    return final_result


def main() -> None:
    """Parse the command-line arguments and run TD(lambda) self-play training."""
    parser = argparse.ArgumentParser(description="Train a TD-Gammon agent by self-play until it beats random.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="number of self-play episodes")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY, help="evaluate every N episodes")
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES, help="games per evaluation")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="hidden-layer width")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--lamda", type=float, default=DEFAULT_LAMDA, help="TD(lambda) trace-decay parameter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--out", type=str, default="td-backgammon", help="base file name for the saved model")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        hidden=args.hidden,
        lr=args.lr,
        lamda=args.lamda,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
