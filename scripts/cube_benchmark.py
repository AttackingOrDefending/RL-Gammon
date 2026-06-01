"""Model-agnostic cube-quality and probability-calibration benchmark for a TD-Gammon value network.

This benchmark runs on *any* ``TDGammonNet`` checkpoint (a scalar-trained one with uncalibrated
win/loss components, or a calibrated one with meaningful components) and produces a single readable
report with three sections:

1. **Probability calibration of the net's components.** OpenSpiel self-play games are played; at
   sampled non-terminal decision nodes the WHITE-perspective raw 5-vector ``(o0..o4)`` is recorded
   together with the eventual terminal result. The report shows the mean predicted components, a
   reliability table (bin states by predicted ``o0`` and compare predicted vs empirical white-win
   frequency), the Brier score for ``P(win)``, and the empirical gammon-win rate versus the mean
   predicted gammon-win probability. A calibrated net produces a roughly diagonal reliability table
   and a positive gammon probability; an uncalibrated scalar net has ``o0 ~ 0`` and a badly-off
   table.

2. **Cube decisions are match-score-dependent (the headline).** For the opening position and a few
   sampled non-terminal positions the cube decision, the cubeful money equity / match-winning chance
   and the take decision are computed under four contexts (money; 2-away/2-away; trailer 0-6 in a
   first-to-7 Crawford game; the post-Crawford variant). The Crawford context is shown to yield
   ``NO_DOUBLE`` (the cube is dead) while the post-Crawford trailer doubles.

3. **Which probability path the cube layer used.** For each sampled position the report states
   whether ``cube_probs`` returned the raw 5-vector (a calibrated, monotone net) or fell back to the
   gammonless ``[p, 0, 0, 0, 0]`` vector (an uncalibrated net), making the scalar-vs-calibrated
   difference visible.

The benchmark only ever reads the model and the analytic cube layer; it never trains or mutates
anything. Without OpenSpiel installed the play-games sections are skipped and the cube-context table
falls back to a fixed probability vector, so the structure of the report is always present.
"""
import argparse
from dataclasses import dataclass, field

import numpy as np

from rlgammon.agents.td_agent import TDAgent
from rlgammon.cube.cube_equity import (
    CubeAction,
    TakeAction,
    cubeful_money_equity,
    double_decision,
    mwc_from_probs,
    take_decision,
)
from rlgammon.cube.cube_types import CubeOwner, CubeState, GameMode, MatchContext
from rlgammon.cube.met import WOOLSEY_HEINRICH
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    create_game,
    is_openspiel_available,
)
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.rlgammon_types import WHITE

# Default number of self-play games used to collect calibration samples.
DEFAULT_GAMES = 50
# Default random seed for self-play and position sampling.
DEFAULT_SEED = 0
# Number of equal-width bins over predicted P(win) in the reliability table.
N_RELIABILITY_BINS = 5
# Number of components in the cubeless probability 5-vector.
N_PROBABILITY_COMPONENTS = 5
# Probability of recording a decision node as a calibration sample (keeps samples decorrelated).
SAMPLE_PROBABILITY = 0.15
# Maximum number of non-terminal positions reused for the cube-context tables.
MAX_CUBE_POSITIONS = 4
# A WHITE win corresponds to a strictly positive terminal return for WHITE.
WIN_THRESHOLD = 0.0
# Terminal return magnitudes: a single game, a gammon and a backgammon.
GAMMON_MAGNITUDE = 2.0
BACKGAMMON_MAGNITUDE = 3.0
# A centred money 1-cube (Jacoby off, so gammons count) used for every cube-context comparison.
CENTERED_MONEY_CUBE = CubeState(value=1, owner=CubeOwner.CENTERED)
# A fixed, slightly-favoured gammonless vector used for the cube table when OpenSpiel is unavailable.
FALLBACK_CUBE_PROBS = [0.62, 0.0, 0.0, 0.0, 0.0]
# Match parameters for the two match contexts reported in the headline cube table.
MATCH_TO_SEVEN = 7
TWO_AWAY_SCORE = 5
TRAILER_MY_SCORE = 0
TRAILER_OPP_SCORE = 6
# Tolerance under which two cube probability vectors are considered identical (raw, not fallback).
PROBS_EQUAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CubeContext:
    """A named match context used in the headline cube-decision table.

    :param name: a short human-readable label for the context
    :param match_ctx: the match context (money or match) the decision is evaluated under
    """

    name: str
    match_ctx: MatchContext


@dataclass
class CalibrationStats:
    """Aggregated probability-calibration statistics over the sampled decision nodes.

    :param n_samples: the number of recorded (prediction, outcome) samples
    :param mean_predicted: the mean predicted 5-vector ``(o0..o4)``
    :param brier_win: the Brier score of the predicted ``P(win)`` against the white-win indicator
    :param empirical_win_rate: the empirical white-win frequency over the samples
    :param empirical_gammon_rate: the empirical white-gammon-or-better win frequency over the samples
    :param mean_predicted_gammon: the mean predicted gammon-or-better win probability ``o1``
    :param reliability: one ``(low, high, count, predicted, empirical)`` tuple per non-empty bin
    """

    n_samples: int = 0
    mean_predicted: list[float] = field(default_factory=lambda: [0.0] * N_PROBABILITY_COMPONENTS)
    brier_win: float = 0.0
    empirical_win_rate: float = 0.0
    empirical_gammon_rate: float = 0.0
    mean_predicted_gammon: float = 0.0
    reliability: list[tuple[float, float, int, float, float]] = field(default_factory=list)


@dataclass
class CubeRow:
    """One row of a cube-decision table: a context's action, equity/MWC and take decision.

    :param context_name: the label of the match context
    :param action: the doubler's cube action under the context
    :param value: the cubeful money equity (money) or match-winning chance (match) of no-double
    :param value_label: ``"equity"`` (money) or ``"mwc"`` (match), describing ``value``
    :param take: the taker's response to a double under the context
    """

    context_name: str
    action: CubeAction
    value: float
    value_label: str
    take: TakeAction


@dataclass
class PositionReport:
    """The cube-decision table and probability-path verdict for a single position.

    :param label: a short label identifying the position (e.g. ``"opening"``)
    :param raw_probs: the raw EQUITY_SIGMOID 5-vector from WHITE's perspective
    :param used_raw: whether ``cube_probs`` returned the raw vector (True) or the fallback (False)
    :param rows: one cube-decision row per context
    """

    label: str
    raw_probs: list[float]
    used_raw: bool
    rows: list[CubeRow]


@dataclass
class BenchmarkReport:
    """The full benchmark report returned by :func:`run_benchmark`.

    :param model_label: a short description of the loaded model
    :param openspiel: whether OpenSpiel was available (calibration was actually measured)
    :param games: the number of self-play games requested
    :param calibration: the probability-calibration statistics (``None`` without OpenSpiel)
    :param positions: the per-position cube-decision reports
    """

    model_label: str
    openspiel: bool
    games: int
    calibration: CalibrationStats | None
    positions: list[PositionReport]


def cube_contexts() -> list[CubeContext]:
    """Return the four match contexts compared in the headline cube-decision table.

    The contexts are money play, a 2-away/2-away match, a trailer 0-6 in a first-to-7 match (the
    Crawford game, in which the cube is dead), and the post-Crawford variant of that trailer
    position (Crawford already played, so the cube is live again).

    :return: the list of named cube contexts
    """
    return [
        CubeContext("money", MatchContext(mode=GameMode.MONEY)),
        CubeContext("match 2away-2away",
                    MatchContext(GameMode.MATCH, MATCH_TO_SEVEN, TWO_AWAY_SCORE, TWO_AWAY_SCORE)),
        CubeContext("match 0-6 Crawford",
                    MatchContext(GameMode.MATCH, MATCH_TO_SEVEN, TRAILER_MY_SCORE, TRAILER_OPP_SCORE)),
        CubeContext("match 0-6 post-Crawford",
                    MatchContext(GameMode.MATCH, MATCH_TO_SEVEN, TRAILER_MY_SCORE, TRAILER_OPP_SCORE,
                                 crawford_played=True)),
    ]


def _used_raw_probs(agent: TDAgent, state: GameState, perspective: int) -> bool:
    """Return whether ``cube_probs`` used the raw 5-vector rather than the gammonless fallback.

    The raw vector is used when it already forms a valid cumulative distribution; the agent's
    ``cube_probs`` returns it unchanged in that case and otherwise returns the gammonless fallback.
    Comparing the sanitized vector to the raw vector therefore reveals which path was taken.

    :param agent: the agent whose value network is queried
    :param state: the (non-terminal, non-chance) state to evaluate
    :param perspective: the player whose probabilities are computed (WHITE=0, BLACK=1)
    :return: ``True`` if the raw probabilities were used, ``False`` if the fallback was used
    """
    raw = agent.position_probs(state, perspective)
    sanitized = agent.cube_probs(state, perspective)
    return all(abs(a - b) <= PROBS_EQUAL_TOLERANCE for a, b in zip(raw, sanitized, strict=True))


def cube_row(probs: list[float], context: CubeContext) -> CubeRow:
    """Compute one cube-decision row (action, equity/MWC, take) for a position under one context.

    :param probs: the cubeless probability 5-vector from the on-roll (WHITE) player's perspective
    :param context: the named match context to evaluate the decision under
    :return: the cube-decision row for the context
    """
    match_arg = None if context.match_ctx.mode == GameMode.MONEY else context.match_ctx
    action = double_decision(probs, CENTERED_MONEY_CUBE, match_arg, met=WOOLSEY_HEINRICH)
    take = take_decision(probs, CENTERED_MONEY_CUBE, match_arg, met=WOOLSEY_HEINRICH)
    if context.match_ctx.mode == GameMode.MATCH:
        value = mwc_from_probs(probs, context.match_ctx, WOOLSEY_HEINRICH, CENTERED_MONEY_CUBE)
        value_label = "mwc"
    else:
        value = cubeful_money_equity(probs, CENTERED_MONEY_CUBE)
        value_label = "equity"
    return CubeRow(context.name, action, value, value_label, take)


def _position_report(label: str, probs: list[float], used_raw: bool) -> PositionReport:
    """Build the per-context cube-decision report for one position.

    :param label: a short label identifying the position
    :param probs: the cubeless probability 5-vector from WHITE's perspective
    :param used_raw: whether ``cube_probs`` used the raw vector for this position
    :return: the position report with one row per context
    """
    rows = [cube_row(probs, context) for context in cube_contexts()]
    return PositionReport(label=label, raw_probs=probs, used_raw=used_raw, rows=rows)


def _white_outcome(final_return: float) -> tuple[float, float]:
    """Return the (white-win, white-gammon-or-better) indicators from a terminal WHITE return.

    :param final_return: WHITE's signed terminal return (in ``{-3, -2, -1, +1, +2, +3}``)
    :return: a tuple of the white-win indicator and the white-gammon-or-better-win indicator
    """
    win = 1.0 if final_return > WIN_THRESHOLD else 0.0
    gammon_win = 1.0 if final_return >= GAMMON_MAGNITUDE else 0.0
    return win, gammon_win


def _reliability_table(predicted_win: list[float],
                       actual_win: list[float]) -> list[tuple[float, float, int, float, float]]:
    """Bin predictions by predicted ``P(win)`` and return the per-bin calibration summary.

    :param predicted_win: the predicted white-win probabilities (``o0``) at the sampled nodes
    :param actual_win: the white-win indicators at the sampled nodes (parallel to ``predicted_win``)
    :return: one ``(low, high, count, mean_predicted, empirical)`` tuple per non-empty bin
    """
    table: list[tuple[float, float, int, float, float]] = []
    for index in range(N_RELIABILITY_BINS):
        low = index / N_RELIABILITY_BINS
        high = (index + 1) / N_RELIABILITY_BINS
        in_bin = [(p, a) for p, a in zip(predicted_win, actual_win, strict=True)
                  if (low <= p < high) or (index == N_RELIABILITY_BINS - 1 and p == high)]
        if not in_bin:
            continue
        count = len(in_bin)
        mean_predicted = sum(p for p, _ in in_bin) / count
        empirical = sum(a for _, a in in_bin) / count
        table.append((low, high, count, mean_predicted, empirical))
    return table


def _aggregate_calibration(predictions: list[list[float]],
                           outcomes: list[tuple[float, float]]) -> CalibrationStats:
    """Aggregate raw per-node predictions and outcomes into calibration statistics.

    :param predictions: the recorded WHITE-perspective raw 5-vectors at the sampled nodes
    :param outcomes: the parallel ``(white-win, white-gammon-or-better)`` indicator tuples
    :return: the aggregated calibration statistics
    """
    if not predictions:
        return CalibrationStats()
    n_samples = len(predictions)
    array = np.asarray(predictions, dtype=float)
    mean_predicted = [float(value) for value in array.mean(axis=0)]
    predicted_win = [row[0] for row in predictions]
    actual_win = [win for win, _ in outcomes]
    actual_gammon = [gammon for _, gammon in outcomes]
    brier_win = float(np.mean([(p - a) ** 2 for p, a in zip(predicted_win, actual_win, strict=True)]))
    return CalibrationStats(
        n_samples=n_samples,
        mean_predicted=mean_predicted,
        brier_win=brier_win,
        empirical_win_rate=sum(actual_win) / n_samples,
        empirical_gammon_rate=sum(actual_gammon) / n_samples,
        mean_predicted_gammon=mean_predicted[1],
        reliability=_reliability_table(predicted_win, actual_win),
    )


def _play_and_collect(agent: TDAgent, games: int,
                      rng: np.random.Generator) -> tuple[CalibrationStats, list[GameState]]:
    """Play self-play games, collecting calibration samples and a few non-terminal positions.

    At each decision node the WHITE-perspective raw 5-vector is recorded with probability
    :data:`SAMPLE_PROBABILITY`, paired at game end with WHITE's actual outcome; a handful of the
    sampled decision states are cloned and returned for the cube-context tables. The agent's own
    1-ply move policy drives both seats (self-play).

    :param agent: the agent providing the move policy and the value network
    :param games: the number of self-play games to play
    :param rng: the random number generator driving chance sampling and node sampling
    :return: a tuple of the aggregated calibration statistics and the cloned sample positions
    """
    game = create_game(PossibleEngine.OPEN_SPIEL)
    predictions: list[list[float]] = []
    outcomes: list[tuple[float, float]] = []
    sample_positions: list[GameState] = []
    for _game_index in range(games):
        state = game.new_initial_state()
        game_sample_indices: list[int] = []
        while not state.is_terminal():
            if state.is_chance_node():
                apply_sampled_chance(state, rng)
                continue
            if rng.random() < SAMPLE_PROBABILITY:
                game_sample_indices.append(len(predictions))
                predictions.append(agent.position_probs(state, WHITE))
                outcomes.append((0.0, 0.0))
                if len(sample_positions) < MAX_CUBE_POSITIONS:
                    sample_positions.append(state.clone())
            action = agent.choose_move(state.legal_actions(), state)
            state.apply_action(action)
        outcome = _white_outcome(float(state.returns()[WHITE]))
        for sample_index in game_sample_indices:
            outcomes[sample_index] = outcome
    return _aggregate_calibration(predictions, outcomes), sample_positions


def _opening_position() -> GameState:
    """Return the opening backgammon position (the first decision node after the opening roll).

    :return: the post-opening-roll decision-node state
    """
    game = create_game(PossibleEngine.OPEN_SPIEL)
    state = game.new_initial_state()
    # Resolve the opening dice deterministically so the opening position is reproducible.
    actions, _probs = zip(*state.chance_outcomes(), strict=True)
    state.apply_action(int(actions[0]))
    return state


def _build_positions(agent: TDAgent, sample_positions: list[GameState]) -> list[PositionReport]:
    """Build the cube-decision reports for the opening position and the sampled positions.

    :param agent: the agent whose value network supplies the probabilities
    :param sample_positions: the non-terminal positions sampled during self-play
    :return: the per-position cube-decision reports
    """
    reports: list[PositionReport] = []
    opening = _opening_position()
    mover = opening.current_player()
    reports.append(_position_report(
        "opening", agent.cube_probs(opening, mover), _used_raw_probs(agent, opening, mover)))
    for index, state in enumerate(sample_positions):
        mover = state.current_player()
        reports.append(_position_report(
            f"sample-{index}", agent.cube_probs(state, mover), _used_raw_probs(agent, state, mover)))
    return reports


def _fixed_positions() -> list[PositionReport]:
    """Build a single cube-decision report from a fixed probability vector (no OpenSpiel).

    :return: a one-element list with the fixed-probability cube-decision report
    """
    return [_position_report("fixed-probs", FALLBACK_CUBE_PROBS, used_raw=True)]


def run_benchmark(*, model_path: str | None, fresh: bool, games: int, seed: int) -> BenchmarkReport:
    """Run the full cube-quality / calibration benchmark and return its report.

    With OpenSpiel available the calibration section plays ``games`` self-play games and the cube
    tables use the opening position plus sampled positions; without OpenSpiel the calibration section
    is skipped and the cube table uses a fixed probability vector, so the report structure is stable.

    :param model_path: a saved-model file name to load, or ``None`` to build a fresh network
    :param fresh: accepted for CLI symmetry; a fresh network is built whenever ``model_path`` is None
    :param games: the number of self-play games for the calibration section
    :param seed: the seed for the agent and the self-play random number generator
    :return: the assembled benchmark report
    """
    del fresh
    agent = TDAgent(pre_made_model_file_name=model_path, seed=seed) if model_path is not None \
        else TDAgent(seed=seed)
    model_label = f"loaded:{model_path}" if model_path is not None else "fresh (untrained)"
    if not is_openspiel_available():
        return BenchmarkReport(model_label=model_label, openspiel=False, games=games,
                               calibration=None, positions=_fixed_positions())
    rng = np.random.default_rng(seed)
    calibration, sample_positions = _play_and_collect(agent, games, rng)
    positions = _build_positions(agent, sample_positions)
    return BenchmarkReport(model_label=model_label, openspiel=True, games=games,
                           calibration=calibration, positions=positions)


def _format_calibration(calibration: CalibrationStats) -> list[str]:
    """Format the calibration section of the report into printable lines.

    :param calibration: the aggregated calibration statistics
    :return: the printable lines of the calibration section
    """
    mean = calibration.mean_predicted
    lines = [
        "== 1. Probability calibration (WHITE perspective) ==",
        f"samples: {calibration.n_samples}",
        "mean predicted o0..o4 (cumulative): "
        + ", ".join(f"{value:.3f}" for value in mean),
        f"  o0=P(win)={mean[0]:.3f}  o1=P(win>=gammon)={mean[1]:.3f}  o2=P(win bg)={mean[2]:.3f}",
        f"  o3=P(lose>=gammon)={mean[3]:.3f}  o4=P(lose bg)={mean[4]:.3f}",
        f"Brier score P(win): {calibration.brier_win:.4f}  (lower is better; 0.25 = always 0.5)",
        f"empirical white-win rate: {calibration.empirical_win_rate:.3f}  "
        f"(vs mean predicted o0 {mean[0]:.3f})",
        f"empirical white-gammon rate: {calibration.empirical_gammon_rate:.3f}  "
        f"(vs mean predicted o1 {calibration.mean_predicted_gammon:.3f})",
        "reliability table (bin by predicted P(win)):",
        "  bin            count   predicted   empirical",
    ]
    if not calibration.reliability:
        lines.append("  (no samples)")
    for low, high, count, predicted, empirical in calibration.reliability:
        lines.append(
            f"  [{low:.1f}, {high:.1f}]   {count:5d}      {predicted:.3f}       {empirical:.3f}")
    return lines


def _format_positions(positions: list[PositionReport]) -> list[str]:
    """Format the cube-decision and probability-path sections of the report into printable lines.

    :param positions: the per-position cube-decision reports
    :return: the printable lines of the cube-decision and probability-path sections
    """
    lines = ["== 2. Cube decisions are match-score-dependent =="]
    for position in positions:
        probs = ", ".join(f"{value:.3f}" for value in position.raw_probs)
        lines.append(f"position '{position.label}'  cube_probs=[{probs}]")
        lines.append("  context                   action        value            take")
        lines.extend(
            f"  {row.context_name:<24}  {row.action.name:<12}  "
            f"{row.value_label}={row.value:.4f}      {row.take.name}"
            for row in position.rows)
    lines.append("")
    lines.append("== 3. Probability path used by the cube layer ==")
    for position in positions:
        path = "RAW 5-vector (calibrated/monotone)" if position.used_raw \
            else "GAMMONLESS FALLBACK [p,0,0,0,0] (uncalibrated)"
        lines.append(f"  position '{position.label}': {path}")
    return lines


def format_report(report: BenchmarkReport) -> str:
    """Format a benchmark report as a single printable string.

    :param report: the benchmark report to format
    :return: the human-readable, multi-line report string
    """
    lines = [
        "=" * 72,
        f"CUBE-QUALITY / CALIBRATION BENCHMARK  (model: {report.model_label})",
        f"OpenSpiel available: {report.openspiel}   self-play games: {report.games}",
        "=" * 72,
    ]
    if report.calibration is not None:
        lines.extend(_format_calibration(report.calibration))
    else:
        lines.append("== 1. Probability calibration ==")
        lines.append("  (skipped: OpenSpiel not available; cube table uses fixed probs)")
    lines.append("")
    lines.extend(_format_positions(report.positions))
    return "\n".join(lines)


def main() -> None:
    """Parse the command-line arguments, run the benchmark and print the report."""
    parser = argparse.ArgumentParser(
        description="Model-agnostic cube-quality and probability-calibration benchmark.")
    parser.add_argument("--model", type=str, default=None,
                        help="saved-model file name (within rlgammon/agents/saved_agents) to load")
    parser.add_argument("--fresh", action="store_true",
                        help="use a fresh untrained network instead of loading a model")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="number of self-play games for the calibration section")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    args = parser.parse_args()

    report = run_benchmark(model_path=args.model, fresh=args.fresh, games=args.games, seed=args.seed)
    print(format_report(report))


if __name__ == "__main__":
    main()
