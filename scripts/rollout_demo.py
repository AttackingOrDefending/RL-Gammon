"""Demonstrate truncated rollouts with variance reduction over the calibrated TD value network.

This script loads the calibrated TD-Gammon value network and, on a few decision-node positions:

* (a) prints the static net equity next to the truncated-rollout equity with their standard errors;
* (b) shows that the control-variate ("lookahead") variance reduction lowers the rollout standard
  error at *equal* trials, using common random numbers so the two estimators see identical dice;
* (c) constructs a position where a deeper rollout-guided move choice can differ from the 1-ply
  greedy choice, and reports both choices and their rollout equities.

It is a runnable report (``python -m scripts.rollout_demo``); every number is seeded and reproducible.
"""

import argparse

import numpy as np

from rlgammon.agents.td_agent import TDAgent
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    create_game,
)
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator
from rlgammon.rlgammon_types import WHITE
from rlgammon.rollout.rollout import RolloutEvaluator, rollout_equity
from rlgammon.rollout.rollout_types import RolloutConfig

# File name (within ``rlgammon/agents/saved_agents``) of the calibrated TD network the demo loads.
CALIBRATED_MODEL = "td-calibrated-077c912f-18c5-4c02-98a7-8f64254922be-(1500).pt"
# Number of decision-node positions sampled from self-play for the static-vs-rollout comparison.
DEFAULT_NUM_POSITIONS = 3
# Number of trials used for the headline static-vs-rollout comparison.
DEFAULT_TRIALS = 300
# Truncation length (decision plies) before the value net is used to bootstrap a trial.
DEFAULT_MAX_DEPTH = 6
# Number of trials used for the (more expensive) variance-reduction comparison.
VR_TRIALS = 400
# Number of trials used per candidate move in the (move-by-move) disagreement search.
MOVE_TRIALS = 120
# Truncation depth used in the move-by-move disagreement search (kept short for speed).
MOVE_MAX_DEPTH = 4
# Seed for the position-sampling random number generator.
POSITION_SEED = 11
# Seed shared by the compared rollout variants (common random numbers).
ROLLOUT_SEED = 2024
# Number of decision plies to look for a position where rollout disagrees with the 1-ply choice.
DISAGREEMENT_PLIES = 12
# A wide separator line for the printed report.
RULE = "=" * 78


def _advance_to_decision(state: GameState, rng: np.random.Generator) -> None:
    """
    Advance a state in place past any pending chance nodes to the next decision node.

    :param state: the game state to advance (mutated in place)
    :param rng: the random number generator used to resolve chance nodes
    """
    while not state.is_terminal() and state.is_chance_node():
        apply_sampled_chance(state, rng)


def _sample_positions(num_positions: int, rng: np.random.Generator) -> list[GameState]:
    """
    Sample a few non-terminal decision-node positions from a self-play game on the real engine.

    :param num_positions: the number of decision-node positions to collect
    :param rng: the random number generator driving the self-play and chance sampling
    :return: a list of cloned decision-node states
    """
    agent = TDAgent(pre_made_model_file_name=CALIBRATED_MODEL)
    game = create_game(PossibleEngine.OPEN_SPIEL)
    state = game.new_initial_state()
    positions: list[GameState] = []
    plies = 0
    while len(positions) < num_positions and not state.is_terminal():
        _advance_to_decision(state, rng)
        if state.is_terminal():
            break
        # Skip the very first few plies so the sampled positions are past the symmetric opening.
        if plies >= num_positions and len(positions) < num_positions:
            positions.append(state.clone())
        state.apply_action(agent.choose_move(state.legal_actions(), state))
        plies += 1
    while len(positions) < num_positions:
        fresh = game.new_initial_state()
        _advance_to_decision(fresh, rng)
        positions.append(fresh.clone())
    return positions


def _print_static_vs_rollout(positions: list[GameState], agent: TDAgent, trials: int) -> None:
    """
    Print the static net equity next to the truncated-rollout equity for each position.

    :param positions: the decision-node positions to evaluate
    :param agent: the TD agent supplying the value net and the rollout move policy
    :param trials: the number of rollout trials per position
    """
    leaf = ValueNetEvaluator(agent.get_model())
    config = RolloutConfig(num_trials=trials, max_depth=DEFAULT_MAX_DEPTH, seed=ROLLOUT_SEED,
                           variance_reduction=True)
    print(RULE)
    print("(a) STATIC NET EQUITY vs TRUNCATED-ROLLOUT EQUITY (WHITE's perspective)")
    print(RULE)
    print(f"{'position':>9} | {'static':>9} | {'rollout':>9} | {'std-err':>8} | {'+/- 95% CI':>18}")
    for index, state in enumerate(positions):
        rng = np.random.default_rng(ROLLOUT_SEED)
        result = rollout_equity(state, leaf, agent, rng, config, perspective=WHITE)
        half_ci = 1.96 * result.std_error
        ci = f"[{result.equity - half_ci:+.3f}, {result.equity + half_ci:+.3f}]"
        print(f"{index:>9} | {result.baseline:>+9.4f} | {result.equity:>+9.4f} | "
              f"{result.std_error:>8.4f} | {ci:>18}")


def _print_variance_reduction(positions: list[GameState], agent: TDAgent, trials: int) -> None:
    """
    Print plain vs variance-reduced rollout standard errors at equal trials and common random numbers.

    :param positions: the decision-node positions to evaluate
    :param agent: the TD agent supplying the value net and the rollout move policy
    :param trials: the number of rollout trials per position (identical for both variants)
    """
    leaf = ValueNetEvaluator(agent.get_model())
    plain_cfg = RolloutConfig(num_trials=trials, max_depth=DEFAULT_MAX_DEPTH, seed=ROLLOUT_SEED,
                              variance_reduction=False)
    vr_cfg = RolloutConfig(num_trials=trials, max_depth=DEFAULT_MAX_DEPTH, seed=ROLLOUT_SEED,
                           variance_reduction=True)
    anti_cfg = RolloutConfig(num_trials=trials // 2, max_depth=DEFAULT_MAX_DEPTH, seed=ROLLOUT_SEED,
                             variance_reduction=True, antithetic=True)
    print(RULE)
    print(f"(b) VARIANCE REDUCTION AT EQUAL TRIALS (n={trials}, common random numbers)")
    print(RULE)
    print(f"{'position':>9} | {'plain SE':>9} | {'CV SE':>9} | {'CV+anti SE':>11} | "
          f"{'var drop':>9} | {'speed-up':>8}")
    for index, state in enumerate(positions):
        plain = rollout_equity(state, leaf, agent, np.random.default_rng(ROLLOUT_SEED), plain_cfg,
                               perspective=WHITE)
        vr = rollout_equity(state, leaf, agent, np.random.default_rng(ROLLOUT_SEED), vr_cfg,
                            perspective=WHITE)
        anti = rollout_equity(state, leaf, agent, np.random.default_rng(ROLLOUT_SEED), anti_cfg,
                              perspective=WHITE)
        var_drop = 1.0 - (vr.std_error / plain.std_error) ** 2 if plain.std_error > 0 else 0.0
        speed_up = (plain.std_error / vr.std_error) ** 2 if vr.std_error > 0 else float("inf")
        print(f"{index:>9} | {plain.std_error:>9.4f} | {vr.std_error:>9.4f} | {anti.std_error:>11.4f} | "
              f"{var_drop:>8.1%} | {speed_up:>7.2f}x")
    print("\nCV    = control-variate (lookahead) VR; CV+anti also pairs antithetic dice (half the")
    print("        independent trials). 'var drop' is 1 - Var(CV)/Var(plain); 'speed-up' is the")
    print("        trial-count factor a plain rollout needs to match the CV standard error.")


def _rollout_choice(state: GameState, agent: TDAgent, evaluator: RolloutEvaluator) -> tuple[int, float]:
    """
    Return the action a rollout-backed evaluation prefers at ``state`` and its rollout equity.

    Each legal action's afterstate is scored by the (slow, accurate) rollout evaluator from the side
    to move's perspective; the action maximising that rollout equity is returned.

    :param state: the decision-node state to choose a move for
    :param agent: the TD agent (unused policy holder kept for signature symmetry)
    :param evaluator: the rollout-backed evaluator scoring afterstates
    :return: the chosen action id and its rollout equity (from the side to move's perspective)
    """
    del agent
    mover = state.current_player()
    best_action = state.legal_actions()[0]
    best_value = -float("inf")
    for action in state.legal_actions():
        child = state.clone()
        child.apply_action(action)
        value = child.returns()[mover] if child.is_terminal() else evaluator.evaluate(child, mover)
        if value > best_value:
            best_value = value
            best_action = action
    return int(best_action), best_value


def _print_move_disagreement(agent: TDAgent, rng: np.random.Generator) -> None:
    """
    Find and print a position where the rollout-guided move differs from the 1-ply greedy choice.

    Self-play is walked until the 1-ply argmax move and the rollout-best move disagree (or a ply
    budget is exhausted); the two candidate moves and their rollout equities are then printed.

    :param agent: the TD agent supplying the value net and the 1-ply policy
    :param rng: the random number generator driving the self-play walk
    """
    leaf = ValueNetEvaluator(agent.get_model())
    evaluator = RolloutEvaluator(leaf, RolloutConfig(num_trials=MOVE_TRIALS, max_depth=MOVE_MAX_DEPTH,
                                                     seed=ROLLOUT_SEED, variance_reduction=True))
    game = create_game(PossibleEngine.OPEN_SPIEL)
    state = game.new_initial_state()
    print(RULE)
    print("(c) ROLLOUT-GUIDED MOVE CHOICE vs 1-PLY GREEDY CHOICE")
    print(RULE)
    for _ in range(DISAGREEMENT_PLIES):
        _advance_to_decision(state, rng)
        if state.is_terminal():
            break
        legal = state.legal_actions()
        greedy = agent.choose_move(legal, state)
        if len(legal) > 1:
            rollout_best, rollout_value = _rollout_choice(state, agent, evaluator)
            if rollout_best != greedy:
                greedy_child = state.clone()
                greedy_child.apply_action(greedy)
                greedy_value = evaluator.evaluate(greedy_child, state.current_player())
                print(f"side to move      : {'WHITE' if state.current_player() == WHITE else 'BLACK'}")
                print(f"legal moves       : {len(legal)}")
                print(f"1-ply greedy move : action {greedy:<5} rollout equity {greedy_value:+.4f}")
                print(f"rollout-best move : action {rollout_best:<5} rollout equity {rollout_value:+.4f}")
                print(f"rollout gain      : {rollout_value - greedy_value:+.4f} points")
                return
        state.apply_action(greedy)
    print("No disagreement found within the ply budget (the 1-ply net already matched the rollout).")


def main() -> None:
    """Parse arguments, load the calibrated model and print the three-part rollout report."""
    parser = argparse.ArgumentParser(description="Demonstrate truncated rollouts with variance reduction.")
    parser.add_argument("--positions", type=int, default=DEFAULT_NUM_POSITIONS,
                        help="number of decision-node positions to evaluate")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="rollout trials per position")
    parser.add_argument("--vr-trials", type=int, default=VR_TRIALS,
                        help="trials used for the variance-reduction comparison")
    args = parser.parse_args()

    agent = TDAgent(pre_made_model_file_name=CALIBRATED_MODEL)
    positions = _sample_positions(args.positions, np.random.default_rng(POSITION_SEED))

    _print_static_vs_rollout(positions, agent, args.trials)
    print()
    _print_variance_reduction(positions, agent, args.vr_trials)
    print()
    _print_move_disagreement(agent, np.random.default_rng(POSITION_SEED + 1))


if __name__ == "__main__":
    main()
