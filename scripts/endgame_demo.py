"""Demonstrate the endgame specialist's sharpness against a neural-net evaluator on real positions.

This script samples real OpenSpiel backgammon BEAROFF and RACE positions and, for each, compares the
side-to-move's **win probability** from three sources:

* **specialist** -- :func:`~rlgammon.endgame.bearoff.race_win_probability`, the exact two-sided
  bear-off database when both sides are home (and the effective-pip-count model for a longer race);
* **net** -- a :class:`~rlgammon.planning.leaf_evaluator.ValueNetEvaluator` wrapping a
  :class:`~rlgammon.models.value_model.TDGammonNet` (the contact-position fallback), whose equity is
  mapped to an implied win probability ``(equity + 1) / 2``;
* **rollout truth** -- the win rate from many random play-outs. Random play is near-optimal in a pure
  bear-off (the moves are largely forced), so for the headlined BEAROFF positions this is a faithful
  reference that the *exact* specialist tracks tightly while an untrained net does not.

Win probability is chosen as the headline metric because it is what the specialist computes exactly;
the gammon-aware point equity is also printed for context. The script additionally shows the
:class:`~rlgammon.endgame.composite_evaluator.CompositeEvaluator` routing (disengaged positions to the
specialist, the opening to the net). Run with ``python3 -m scripts.endgame_demo`` (OpenSpiel required).
"""

import argparse

import numpy as np

from rlgammon.endgame.bearoff import bearoff_equity, race_win_probability
from rlgammon.endgame.board_decode import decode_board, side_layout_for
from rlgammon.endgame.composite_evaluator import CompositeEvaluator
from rlgammon.endgame.endgame_types import Phase
from rlgammon.endgame.phase import detect_phase
from rlgammon.game import PossibleEngine, apply_sampled_chance, create_game
from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.game.openspiel_adapter import is_openspiel_available
from rlgammon.models.value_model import TDGammonNet
from rlgammon.planning.leaf_evaluator import ValueNetEvaluator

# Default number of disengaged positions to showcase.
DEFAULT_NUM_POSITIONS = 4
# Default number of random play-outs used to estimate the rollout-truth win rate per position.
DEFAULT_ROLLOUTS = 4000
# Safety bound on plies within a single random play-out.
MAX_PLAYOUT_PLIES = 4000
# Safety bound on the number of games scanned while hunting for disengaged positions.
MAX_SCAN_GAMES = 6000
# A strictly positive terminal return is a win for the scored side.
WIN_THRESHOLD = 0.0
# Largest equity magnitude the net's scalar can take (used to clamp its implied win probability).
NET_EQUITY_CLAMP = 1.0


def _rollout_win_rate(state: GameState, perspective: int, rng: np.random.Generator, rollouts: int) -> float:
    """
    Estimate ``perspective``'s win probability by random play-outs (the reference "truth").

    Each play-out resolves chance nodes by sampling and decision nodes uniformly at random; the win
    rate is the fraction of play-outs ending in a positive signed return for ``perspective``.

    :param state: the disengaged position to evaluate
    :param perspective: the side whose win probability to estimate (WHITE=0, BLACK=1)
    :param rng: the random number generator driving the play-outs
    :param rollouts: the number of random play-outs to average
    :return: the rollout estimate of ``perspective``'s win probability in ``[0, 1]``
    """
    wins = 0
    for _ in range(rollouts):
        playout = state.clone()
        plies = 0
        while not playout.is_terminal() and plies < MAX_PLAYOUT_PLIES:
            if playout.is_chance_node():
                apply_sampled_chance(playout, rng)
            else:
                legal = playout.legal_actions()
                playout.apply_action(int(legal[rng.integers(len(legal))]))
            plies += 1
        if playout.is_terminal() and playout.returns()[perspective] > WIN_THRESHOLD:
            wins += 1
    return wins / rollouts if rollouts else 0.0


def _specialist_win_probability(state: GameState, perspective: int) -> float:
    """
    Return the specialist's win probability for ``perspective`` (exact when both sides are home).

    :param state: the disengaged position to evaluate
    :param perspective: the side whose win probability to return (WHITE=0, BLACK=1)
    :return: the specialist win probability in ``[0, 1]``
    """
    layout = decode_board(state, perspective)
    me = side_layout_for(layout, perspective, decoded_from=perspective)
    opponent = side_layout_for(layout, 1 - perspective, decoded_from=perspective)
    on_roll = state.current_player() == perspective
    return race_win_probability(me, opponent, on_roll=on_roll)


def _net_win_probability(net_evaluator: ValueNetEvaluator, state: GameState, perspective: int) -> float:
    """
    Map the net's scalar equity for ``perspective`` to an implied win probability ``(equity + 1) / 2``.

    :param net_evaluator: the neural-net evaluator
    :param state: the position to evaluate
    :param perspective: the side whose win probability to imply (WHITE=0, BLACK=1)
    :return: the net's implied win probability in ``[0, 1]``
    """
    equity = net_evaluator.evaluate(state, perspective)
    clamped = min(max(equity, -NET_EQUITY_CLAMP), NET_EQUITY_CLAMP)
    return (clamped + 1.0) / 2.0


def _find_disengaged_positions(game: BackgammonGame, rng: np.random.Generator,
                               num_positions: int) -> list[GameState]:
    """
    Play random games and collect disengaged decision-node positions, preferring pure bear-offs.

    Two passes are run over the scan budget: the first keeps only BEAROFF positions (where the
    specialist is exact and random play is a faithful truth), and if too few are found a second pass
    fills the remainder with any RACE position, so the demo still produces output on unlucky seeds.

    :param game: the game factory producing fresh initial states
    :param rng: the random number generator driving the random games
    :param num_positions: the number of disengaged positions to collect
    :return: a list of cloned disengaged decision-node states (at most ``num_positions``)
    """
    for accepted_phases in ((Phase.BEAROFF,), (Phase.BEAROFF, Phase.RACE)):
        positions: list[GameState] = []
        for _ in range(MAX_SCAN_GAMES):
            if len(positions) >= num_positions:
                return positions
            positions.extend(_scan_one_game(game, rng, accepted_phases, num_positions - len(positions)))
        if len(positions) >= num_positions:
            return positions
    return positions


def _scan_one_game(game: BackgammonGame, rng: np.random.Generator,
                   accepted_phases: tuple[Phase, ...], wanted: int) -> list[GameState]:
    """
    Play one random game and return up to ``wanted`` decision-node states in the accepted phases.

    :param game: the game factory producing a fresh initial state
    :param rng: the random number generator driving the random game
    :param accepted_phases: the phases whose positions to collect
    :param wanted: the maximum number of positions to return from this game
    :return: the collected cloned decision-node states (at most ``wanted``)
    """
    collected: list[GameState] = []
    state = game.new_initial_state()
    plies = 0
    while not state.is_terminal() and plies < MAX_PLAYOUT_PLIES and len(collected) < wanted:
        if state.is_chance_node():
            apply_sampled_chance(state, rng)
        else:
            if detect_phase(state, state.current_player()) in accepted_phases:
                collected.append(state.clone())
            legal = state.legal_actions()
            state.apply_action(int(legal[rng.integers(len(legal))]))
        plies += 1
    return collected


def _print_position(index: int, state: GameState, composite: CompositeEvaluator,
                    net_evaluator: ValueNetEvaluator, rng: np.random.Generator, rollouts: int) -> tuple[float, float]:
    """
    Print one position's win-probability comparison and routing, returning the two absolute errors.

    :param index: the 1-based position number (for display)
    :param state: the disengaged decision-node state
    :param composite: the phase-aware composite evaluator (for the routing decision)
    :param net_evaluator: the neural-net evaluator used as the contact fallback
    :param rng: the random number generator driving the rollout truth
    :param rollouts: the number of play-outs behind the rollout-truth win rate
    :return: ``(specialist_error, net_error)`` -- absolute win-probability errors vs the rollout truth
    """
    mover = state.current_player()
    layout = decode_board(state, mover)
    phase = detect_phase(state, mover)
    specialist_win = _specialist_win_probability(state, mover)
    net_win = _net_win_probability(net_evaluator, state, mover)
    truth_win = _rollout_win_rate(state, mover, rng, rollouts)
    specialist_equity = bearoff_equity(state, mover)
    routed = "specialist" if composite.phase_of(state, mover) != Phase.CONTACT else "net"
    print(
        f"[{index}] phase={phase.value:<7} mover={mover} "
        f"pips(me/opp)={layout.mover.pip_count():>3}/{layout.opponent.pip_count():<3} "
        f"off(me/opp)={layout.mover.off:>2}/{layout.opponent.off:<2} -> routes to {routed}",
    )
    print(
        f"      P(win): specialist={specialist_win:.4f}  net={net_win:.4f}  rollout_truth={truth_win:.4f}  "
        f"|net-truth|={abs(net_win - truth_win):.4f}  |specialist-truth|={abs(specialist_win - truth_win):.4f}",
    )
    print(f"      specialist gammon-aware equity = {specialist_equity:+.4f} points")
    return abs(specialist_win - truth_win), abs(net_win - truth_win)


def run_demo(num_positions: int, rollouts: int, seed: int) -> None:
    """
    Sample disengaged positions and print the win-probability comparison and routing for each.

    :param num_positions: the number of BEAROFF/RACE positions to showcase
    :param rollouts: the number of random play-outs behind each rollout-truth win rate
    :param seed: the seed for the demo's random number generator
    """
    rng = np.random.default_rng(seed)
    game = create_game(PossibleEngine.OPEN_SPIEL)
    net_evaluator = ValueNetEvaluator(TDGammonNet())
    composite = CompositeEvaluator(net_evaluator)

    # Show the routing on the opening (a contact position) so the net path is exercised too.
    opening = game.new_initial_state()
    apply_sampled_chance(opening, rng)
    opening_routed = "specialist" if composite.phase_of(opening, opening.current_player()) != Phase.CONTACT else "net"
    print(f"opening position: phase={detect_phase(opening, opening.current_player()).value} "
          f"-> composite routes to {opening_routed}")
    print(f"sampling up to {num_positions} disengaged (bear-off/race) positions from random games...\n")

    positions = _find_disengaged_positions(game, rng, num_positions)
    if not positions:
        print("no disengaged position found in the scanned games (try a different seed).")
        return
    specialist_errors: list[float] = []
    net_errors: list[float] = []
    for index, state in enumerate(positions, start=1):
        specialist_error, net_error = _print_position(
            index, state, composite, net_evaluator, np.random.default_rng(seed + index), rollouts,
        )
        specialist_errors.append(specialist_error)
        net_errors.append(net_error)
    mean_specialist = float(np.mean(specialist_errors))
    mean_net = float(np.mean(net_errors))
    print(
        f"\nmean |win-prob error vs rollout truth|: specialist={mean_specialist:.4f}  "
        f"net (untrained)={mean_net:.4f}  "
        f"-> the specialist is {mean_net / max(mean_specialist, 1e-9):.1f}x sharper",
    )


def main() -> None:
    """Parse the command-line arguments and run the endgame specialist demo."""
    parser = argparse.ArgumentParser(description="Show the endgame specialist is sharper than the net.")
    parser.add_argument("--positions", type=int, default=DEFAULT_NUM_POSITIONS, help="disengaged positions to show")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS, help="play-outs per rollout-truth estimate")
    parser.add_argument("--seed", type=int, default=0, help="seed for the demo random number generator")
    args = parser.parse_args()
    if not is_openspiel_available():
        print("OpenSpiel (pyspiel) is not installed; the demo needs real backgammon states.")
        return
    run_demo(args.positions, args.rollouts, args.seed)


if __name__ == "__main__":
    main()
