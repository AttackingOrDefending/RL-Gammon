"""Standalone harness to play full money cube games and matches between two cube-aware agents.

OpenSpiel backgammon has no doubling cube, so this harness layers the analytic cube on top of the
cubeless engine: at the start of every turn the on-roll agent may offer a double (queried through its
``should_double``), and the opponent decides whether to take (``should_take``); the cube value, owner
and the match score / Crawford flag are tracked outside the engine. Games are scored by
``cube_value * gammon_mult`` (the gammon multiplier read from the cubeless terminal return, clamped to
a single point under the live Jacoby rule), passes award the current cube value with no game played.
The harness is deliberately NOT wired into the JSON-validated trainer; it is a self-contained
evaluation tool. A ``use_cube=False`` flag disables all doubling (and Jacoby), reproducing the plain
cubeless result as a regression guard.
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from rlgammon.cube.cube_types import CubeOwner, CubeState, GameMode, MatchContext
from rlgammon.cube.met import MET, WOOLSEY_HEINRICH
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    create_game,
)
from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.rlgammon_types import BLACK, WHITE

# A strictly positive terminal return counts as a win for the scored side.
WIN_THRESHOLD = 0.0
# The gammon multiplier under the live Jacoby rule before the cube is ever turned.
JACOBY_CLAMP_MULT = 1.0
# Default cube-life index used by the queried agents.
DEFAULT_CUBE_EFFICIENCY = 0.68
# Maximum number of decision plies before a game is abandoned (defensive guard against stalls).
MAX_GAME_PLIES = 10_000


@runtime_checkable
class CubeAgent(Protocol):
    """Protocol for an agent that can both move and make doubling-cube decisions."""

    def choose_move(self, actions: list[int], state: GameState) -> int:
        """
        Return the chosen move for the side to move.

        :param actions: the legal action ids at the current decision node
        :param state: the current decision-node game state
        :return: the chosen action id
        """

    def should_double(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,
                      met: MET | None = ..., x: float = ...) -> bool:
        """
        Return whether the on-roll agent should offer a double.

        :param state: the decision-node state with the agent on roll
        :param cube: the current cube state from the agent's perspective
        :param match_ctx: the match context from the agent's perspective
        :param met: an optional match-equity table
        :param x: the cube-life index
        :return: whether the agent should double
        """

    def should_take(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,
                    met: MET | None = ..., x: float = ...) -> bool:
        """
        Return whether the agent (the taker) should take an offered double.

        :param state: the decision-node state with the agent (the taker) on roll
        :param cube: the pre-double cube state from the agent's perspective
        :param match_ctx: the match context from the agent's perspective
        :param met: an optional match-equity table
        :param x: the cube-life index
        :return: whether the agent should take
        """


class NoDoubleMoneyTaker:
    """A baseline cube policy that never doubles and always takes, wrapping a move policy."""

    def __init__(self, mover: CubeAgent) -> None:
        """
        Construct the baseline around an existing move policy.

        :param mover: the agent whose :meth:`choose_move` is reused for actual play
        """
        self._mover = mover

    def choose_move(self, actions: list[int], state: GameState) -> int:
        """
        Delegate the move choice to the wrapped agent.

        :param actions: the legal action ids at the current decision node
        :param state: the current decision-node game state
        :return: the chosen action id
        """
        return self._mover.choose_move(actions, state)

    def should_double(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,  # noqa: ARG002
                      met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY) -> bool:  # noqa: ARG002
        """
        Never offer a double.

        :param state: the decision-node state with the baseline on roll (unused)
        :param cube: the current cube state (unused)
        :param match_ctx: the match context (unused)
        :param met: an optional match-equity table (unused)
        :param x: the cube-life index (unused)
        :return: always ``False``
        """
        return False

    def should_take(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,  # noqa: ARG002
                    met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY) -> bool:  # noqa: ARG002
        """
        Always take an offered double (the money "always take" baseline).

        :param state: the decision-node state with the baseline on roll (unused)
        :param cube: the pre-double cube state (unused)
        :param match_ctx: the match context (unused)
        :param met: an optional match-equity table (unused)
        :param x: the cube-life index (unused)
        :return: always ``True``
        """
        return True


class _Tally:
    """Mutable accumulator of per-game outcomes across a cube test run."""

    def __init__(self) -> None:
        """Construct a zeroed tally."""
        self.games = 0
        self.wins = 0
        self.points = 0.0
        self.cube_turns = 0
        self.doubles = 0
        self.takes = 0
        self.passes = 0
        self.match_wins = 0
        self.matches = 0


class CubeTesting:
    """Harness playing money cube games and matches between two cube-aware agents (or a baseline)."""

    def __init__(self, met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY, *,
                 engine: PossibleEngine = PossibleEngine.OPEN_SPIEL) -> None:
        """
        Construct the cube-testing harness.

        :param met: the match-equity table to use for match play (defaults to the Woolsey-Heinrich table)
        :param x: the cube-life index passed to the queried agents
        :param engine: which game engine backend to play on (OpenSpiel for real backgammon)
        """
        self._met = met if met is not None else WOOLSEY_HEINRICH
        self._x = x
        self._engine = engine
        self._tally = _Tally()

    def _query_double(self, agent: CubeAgent, state: GameState, cube: CubeState,
                      match_ctx: MatchContext) -> bool:
        """
        Ask the on-roll agent whether to double, passing the cube/match from its own perspective.

        :param agent: the on-roll agent
        :param state: the decision-node state with ``agent`` on roll
        :param cube: the cube state from ``agent``'s perspective
        :param match_ctx: the match context from ``agent``'s perspective
        :return: whether the agent offers a double
        """
        return agent.should_double(state, cube, match_ctx, met=self._met, x=self._x)

    def _query_take(self, agent: CubeAgent, state: GameState, cube: CubeState,
                    match_ctx: MatchContext) -> bool:
        """
        Ask the taker whether to take, passing the (pre-double) cube/match from its own perspective.

        :param agent: the taker (the side that was doubled)
        :param state: the decision-node state with the taker on roll
        :param cube: the pre-double cube state from the taker's perspective
        :param match_ctx: the match context from the taker's perspective
        :return: whether the taker takes the double
        """
        return agent.should_take(state, cube, match_ctx, met=self._met, x=self._x)

    def _maybe_double(self, mover: int, agents: Mapping[int, CubeAgent], state: GameState,
                      cube: CubeState, match_ctx: MatchContext, tally: _Tally) -> tuple[CubeState, int | None]:
        """
        Run the start-of-turn cube interaction for the on-roll player.

        If the on-roll player may and wants to double, the opponent decides; a take updates the cube
        (value doubled, ownership to the doubler's opponent) while a pass ends the game immediately
        with the on-roll player winning the current cube value.

        :param mover: the on-roll player (WHITE=0, BLACK=1)
        :param agents: the per-colour cube agents
        :param state: the decision-node state with ``mover`` on roll
        :param cube: the current cube state (stored from WHITE's perspective)
        :param match_ctx: the match context (stored from WHITE's perspective)
        :param tally: the run tally to update with cube statistics
        :return: a tuple of the (possibly updated) cube and the winner if the double was passed
        """
        mover_cube = cube if mover == WHITE else cube.flip_perspective()
        if not mover_cube.can_double():
            return cube, None
        tally.cube_turns += 1
        mover_ctx = match_ctx if mover == WHITE else match_ctx.flip_perspective()
        if not self._query_double(agents[mover], state, mover_cube, mover_ctx):
            return cube, None
        tally.doubles += 1
        opponent = BLACK if mover == WHITE else WHITE
        opp_cube = cube if opponent == WHITE else cube.flip_perspective()
        opp_ctx = match_ctx if opponent == WHITE else match_ctx.flip_perspective()
        if self._query_take(agents[opponent], state, opp_cube, opp_ctx):
            tally.takes += 1
            doubled = mover_cube.after_double()
            # Re-express the doubled cube (owner is the opponent of the mover) from WHITE's view.
            return (doubled if mover == WHITE else doubled.flip_perspective()), None
        tally.passes += 1
        return cube, mover

    def _play_game(self, game: BackgammonGame, agents: Mapping[int, CubeAgent], cube: CubeState,
                   match_ctx: MatchContext, rng: np.random.Generator, *,
                   use_cube: bool) -> tuple[int, float]:
        """
        Play a single cube game to its end and return ``(winner, points_won)`` for the winner.

        :param game: the game factory producing a fresh initial state
        :param agents: the per-colour cube agents
        :param cube: the starting cube state for the game (from WHITE's perspective)
        :param match_ctx: the match context (from WHITE's perspective)
        :param rng: the random number generator driving chance sampling
        :param use_cube: whether doubling and the Jacoby clamp are active this game
        :return: a tuple of the winning colour and the (cube-scaled) points it won
        """
        state = game.new_initial_state()
        current_cube = cube
        tally = self._tally
        for _ply in range(MAX_GAME_PLIES):
            if state.is_terminal():
                break
            if state.is_chance_node():
                apply_sampled_chance(state, rng)
                continue
            mover = state.current_player()
            if use_cube:
                current_cube, passer = self._maybe_double(mover, agents, state, current_cube,
                                                          match_ctx, tally)
                if passer is not None:
                    return passer, float(current_cube.value)
            action = agents[mover].choose_move(state.legal_actions(), state)
            state.apply_action(action)
        return self._score_terminal(state, current_cube, use_cube=use_cube)

    def _score_terminal(self, state: GameState, cube: CubeState, *,
                        use_cube: bool) -> tuple[int, float]:
        """
        Score a terminal state into ``(winner, points_won)`` applying the cube and Jacoby clamp.

        :param state: the terminal game state
        :param cube: the final cube state (from WHITE's perspective)
        :param use_cube: whether the cube value and the Jacoby clamp apply
        :return: a tuple of the winning colour and the points it won
        """
        returns = state.returns()
        winner = WHITE if returns[WHITE] > WIN_THRESHOLD else BLACK
        gammon_mult = abs(returns[winner])
        if not use_cube:
            return winner, gammon_mult
        if cube.jacoby and cube.owner == CubeOwner.CENTERED and cube.value <= 1:
            # Live Jacoby, cube never turned: gammons/backgammons count as a single point.
            gammon_mult = JACOBY_CLAMP_MULT
        return winner, gammon_mult * float(cube.value)

    def _start_cube(self, base_cube: CubeState, match_ctx: MatchContext, *,
                    use_cube: bool) -> CubeState:
        """
        Return the per-game starting cube, freezing it dead for a Crawford game.

        :param base_cube: the configured starting cube (centred 1-cube by default)
        :param match_ctx: the match context for the upcoming game
        :param use_cube: whether the cube is active at all
        :return: the starting cube for the game
        """
        if not use_cube or match_ctx.cube_dead_this_game:
            # A dead cube (Crawford) is represented by an un-doublable max-value cube held centred.
            return CubeState(value=base_cube.value, owner=CubeOwner.CENTERED, jacoby=base_cube.jacoby,
                             beavers=base_cube.beavers, max_cube=base_cube.value)
        return base_cube

    def play_money_games(self, agents: Mapping[int, CubeAgent], n_games: int, rng: np.random.Generator,
                         *, cube: CubeState | None = None, use_cube: bool = True) -> dict[str, float]:
        """
        Play ``n_games`` money cube games (alternating which colour the first agent controls).

        :param agents: the per-colour cube agents (keys ``WHITE`` and ``BLACK``)
        :param n_games: the number of games to play
        :param rng: the random number generator driving chance sampling
        :param cube: the starting cube (defaults to a centred 1-cube; Jacoby off, beavers off)
        :param use_cube: whether doubling and the Jacoby clamp are active
        :return: a result dict (win_rate, ppg, mwc, mean_cube_turns, doubles, takes, passes, games)
        """
        self._tally = _Tally()
        base_cube = cube if cube is not None else CubeState()
        match_ctx = MatchContext(mode=GameMode.MONEY)
        game = create_game(self._engine)
        for game_index in range(n_games):
            # The scored agent (key WHITE in the result) alternates colour to remove first-move bias.
            colour_agents = agents if game_index % 2 == 0 else {WHITE: agents[BLACK], BLACK: agents[WHITE]}
            start_cube = self._start_cube(base_cube, match_ctx, use_cube=use_cube)
            winner, points = self._play_game(game, colour_agents, start_cube, match_ctx, rng,
                                             use_cube=use_cube)
            # Translate the colour winner back to "did the scored agent win" accounting for the swap.
            scored_won = (winner == WHITE) if game_index % 2 == 0 else (winner == BLACK)
            self._tally.games += 1
            if scored_won:
                self._tally.wins += 1
                self._tally.points += points
            else:
                self._tally.points -= points
        return self._summarize(match_aware=False)

    def play_match(self, agents: Mapping[int, CubeAgent], match_length: int, rng: np.random.Generator,
                   *, cube: CubeState | None = None, use_cube: bool = True) -> int:
        """
        Play a single match to ``match_length`` points and return the winning colour.

        The cube/Crawford state is tracked across games; the scored agent keeps its colour for the
        whole match (matches are symmetric over many runs, unlike single money games).

        :param agents: the per-colour cube agents (keys ``WHITE`` and ``BLACK``)
        :param match_length: the number of points needed to win the match
        :param rng: the random number generator driving chance sampling
        :param cube: the per-game starting cube (defaults to a centred 1-cube)
        :param use_cube: whether doubling is active
        :return: the colour that won the match (WHITE=0, BLACK=1)
        """
        base_cube = cube if cube is not None else CubeState()
        scores = {WHITE: 0, BLACK: 0}
        crawford_played = False
        game = create_game(self._engine)
        while scores[WHITE] < match_length and scores[BLACK] < match_length:
            match_ctx = MatchContext(mode=GameMode.MATCH, match_length=match_length,
                                     my_score=scores[WHITE], opp_score=scores[BLACK],
                                     crawford_played=crawford_played)
            is_crawford = match_ctx.is_crawford
            start_cube = self._start_cube(base_cube, match_ctx, use_cube=use_cube)
            winner, points = self._play_game(game, agents, start_cube, match_ctx, rng, use_cube=use_cube)
            scores[winner] = min(scores[winner] + int(points), match_length)
            self._tally.games += 1
            if is_crawford:
                crawford_played = True
        return WHITE if scores[WHITE] >= match_length else BLACK

    def play_matches(self, agents: Mapping[int, CubeAgent], match_length: int, n_matches: int,
                     rng: np.random.Generator, *, cube: CubeState | None = None,
                     use_cube: bool = True) -> dict[str, float]:
        """
        Play ``n_matches`` matches and aggregate the scored agent's match-winning chance.

        :param agents: the per-colour cube agents (keys ``WHITE`` and ``BLACK``)
        :param match_length: the number of points needed to win each match
        :param n_matches: the number of matches to play
        :param rng: the random number generator driving chance sampling
        :param cube: the per-game starting cube (defaults to a centred 1-cube)
        :param use_cube: whether doubling is active
        :return: a result dict (win_rate, mwc, doubles, takes, passes, mean_cube_turns, games/matches)
        """
        self._tally = _Tally()
        for match_index in range(n_matches):
            colour_agents = agents if match_index % 2 == 0 else {WHITE: agents[BLACK], BLACK: agents[WHITE]}
            winner = self.play_match(colour_agents, match_length, rng, cube=cube, use_cube=use_cube)
            scored_won = (winner == WHITE) if match_index % 2 == 0 else (winner == BLACK)
            self._tally.matches += 1
            if scored_won:
                self._tally.match_wins += 1
        return self._summarize(match_aware=True)

    def _summarize(self, *, match_aware: bool) -> dict[str, float]:
        """
        Reduce the current tally into a float result dictionary.

        :param match_aware: whether to report match-winning chance (matches) or per-game stats (money)
        :return: the aggregated result dictionary
        """
        tally = self._tally
        games = tally.games if tally.games else 1
        matches = tally.matches if tally.matches else 1
        win_rate = (tally.match_wins / matches) if match_aware else (tally.wins / games)
        mwc = (tally.match_wins / matches) if match_aware else win_rate
        decisions = tally.doubles if tally.doubles else 1
        return {
            "win_rate": win_rate,
            "ppg": tally.points / games,
            "mwc": mwc,
            "mean_cube_turns": tally.cube_turns / games,
            "doubles": float(tally.doubles),
            "takes": tally.takes / decisions,
            "passes": tally.passes / decisions,
            "games": float(tally.games),
            "matches": float(tally.matches),
        }
