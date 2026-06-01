"""Play the strongest assembled agent against a random opponent and report its strength.

This is the runnable entry point for the strong agent assembled in
:mod:`rlgammon.agents.strong_agent`. It loads the calibrated TD value network, builds the strong
(expectiminimax-over-a-phase-aware-evaluator) agent, evaluates it against a uniform-random opponent
over a handful of games and prints its win-rate and average points. For contrast it also evaluates the
plain 1-ply greedy TD agent on the same network and the same number of games, so the lift from the
deeper search is visible.

Both agents expose ``choose_move(actions, state)``, so they slot straight into the shared
:func:`~scripts.eval_vs_random._evaluate_policy` loop on the real OpenSpiel engine. The strong agent's
2-ply search is markedly slower than the 1-ply baseline, so keep ``--games`` modest.

Run it as ``python -m scripts.play_strong`` (with the repository root on ``PYTHONPATH``).
"""

import argparse

import numpy as np

from rlgammon.agents.strong_agent import (
    CALIBRATED_MODEL,
    StrongAgentConfig,
    build_strong_agent,
)
from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE
from scripts.eval_vs_random import _evaluate_policy

# Default number of evaluation games: 2-ply search is slow, so this is kept small enough to finish in
# a couple of minutes while still giving a meaningful win-rate against the random opponent.
DEFAULT_GAMES = 50
# Default expectiminimax search depth (decision plies) of the strong agent.
DEFAULT_DEPTH = 2
# Default seed for the evaluation random number generator.
DEFAULT_SEED = 0


def _evaluate(label: str, choose_move: object, games: int, seed: int) -> dict[str, float]:
    """
    Evaluate a move policy against random over ``games`` games from a fresh, seeded generator.

    A fresh generator seeded identically per policy means the strong agent and the 1-ply baseline face
    the same dice and the same random-opponent choices, so the comparison is apples-to-apples.

    :param label: a short human-readable label for the policy (printed by the caller)
    :param choose_move: the policy's ``choose_move(actions, state)`` callable
    :param games: the number of evaluation games to play
    :param seed: the seed for the evaluation random number generator
    :return: the evaluation result dict (``win_rate``, ``avg_points``, ``games``)
    """
    del label
    return _evaluate_policy(choose_move, games, np.random.default_rng(seed))  # type: ignore[arg-type]


def main() -> None:
    """Parse the command-line arguments, build the strong agent and print its evaluation."""
    parser = argparse.ArgumentParser(
        description="Play the strongest assembled agent (calibrated net + expectiminimax) vs random.")
    parser.add_argument("--model", type=str, default=CALIBRATED_MODEL,
                        help="saved-model file name (within rlgammon/agents/saved_agents) to load")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="expectiminimax search depth in decision plies (1 = greedy, 2 = 2-ply)")
    parser.add_argument("--rollouts", action="store_true",
                        help="use a truncated-rollout leaf evaluator (much stronger, much slower)")
    parser.add_argument("--cube", action="store_true",
                        help="build the agent with the doubling-cube decision methods enabled")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="number of evaluation games (keep modest; 2-ply is slow)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed for the evaluation random number generator")
    args = parser.parse_args()

    config = StrongAgentConfig(max_depth=args.depth, use_rollouts=args.rollouts, use_cube=args.cube)
    strong = build_strong_agent(args.model, config, color=WHITE)
    baseline = TDAgent(pre_made_model_file_name=args.model, color=WHITE)

    strong_result = _evaluate("strong", strong.choose_move, args.games, args.seed)
    baseline_result = _evaluate("baseline", baseline.choose_move, args.games, args.seed)

    print(f"model={args.model}")
    print(f"strong   (depth={args.depth} rollouts={args.rollouts} cube={args.cube}) "
          f"games={int(strong_result['games'])} "
          f"win_rate={strong_result['win_rate']:.4f} avg_points={strong_result['avg_points']:.4f}")
    print(f"baseline (1-ply greedy TD)                games={int(baseline_result['games'])} "
          f"win_rate={baseline_result['win_rate']:.4f} avg_points={baseline_result['avg_points']:.4f}")
    print(f"lift     win_rate={strong_result['win_rate'] - baseline_result['win_rate']:+.4f} "
          f"avg_points={strong_result['avg_points'] - baseline_result['avg_points']:+.4f}")


if __name__ == "__main__":
    main()
