"""Fully autonomous DCOI player for backgammon.

Plugs our TD-Gammon bot into the Discord Computer Olympiad Interface
(https://github.com/CohenSolalQuentin/Discord_Computer_Olympiad_Interface) so it
plays games on Discord with **no human in the loop**.

It subclasses the framework's :class:`BasicPlayer` and implements ``my_plays``:

1. ``game_history`` (the full ICGA transcript so far, dice + moves for both players)
   is replayed onto a fresh pyspiel state with :mod:`icga.icga_format`.
2. the TD bot chooses a move on that state (handling doubles as a full turn);
3. we return the *referee engine's own* legal-move string whose checker-move
   multiset equals the bot's choice. Returning the engine's native string means it
   is guaranteed to satisfy the framework's ``assert move in valid_actions()``.

Time management: ``my_plays`` receives the remaining clock (``time_left``). A
:class:`~rlgammon.planning.time_manager.TimeManager` turns it into a per-move
monotonic deadline, which is handed to the bot's anytime expectiminimax search so
the bot thinks deeper when it has time and never overruns its budget. An infinite
clock (untimed game) yields no deadline and keeps the fast 1-ply default.

Any failure (reconstruction mismatch, model error, ...) falls back to a legal move,
so the bot never crashes or forfeits a game.

Launch it with :mod:`icga.run_dcoi_bot` (see that module / icga/README.md).

NOTE: this module imports the DCOI framework (``discord_interface``), which requires
`discord.py` to be installed. It is only imported when running the Discord bot; the
offline tools in :mod:`icga.auto_play` do not depend on it.
"""
from __future__ import annotations

from math import inf
import os
import time
import traceback

import pyspiel  # type: ignore[import-not-found]

from discord_interface.player.model.basic_player import BasicPlayer
from icga.bot import DEFAULT_MODEL, ICGABot
from icga.icga_format import (
    apply_icga_move,
    dice_to_chance_action,
    parse_dice,
    parse_icga_move,
    turn_to_icga,
)
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.planning.time_manager import TimeManager

GAME_NAME = "backgammon(scoring_type=full_scoring)"
# Below this per-move budget (seconds) the anytime search cannot complete even one useful ply,
# so we fall back to the instant 1-ply agent rather than risk overrunning the clock.
MIN_SEARCH_BUDGET = 2.0


def _move_multiset(token: str) -> tuple[tuple[int, int], ...]:
    """Normalise an ICGA move token to a sorted multiset of (point, die)."""
    return tuple(sorted(parse_icga_move(token)))


class TDGammonAI(BasicPlayer):
    """Autonomous DCOI player driven by the TD-Gammon agent."""

    def __init__(self, game: object = None, model: str | None = None,
                 engine: str = "td", ply: int = 1) -> None:
        """Construct the player. The agent/model is loaded once and reused across games.

        Reads optional overrides from the environment so the launcher needs no edits:
        ``ICGA_MODEL`` (path), ``ICGA_ENGINE`` (``td``/``search``), ``ICGA_PLY`` (int).
        """
        super().__init__(game=game)
        # Player.reset() re-invokes __init__ between games; only load the model once.
        if getattr(self, "_bot", None) is None:
            model = os.environ.get("ICGA_MODEL", model) or str(DEFAULT_MODEL)
            engine = os.environ.get("ICGA_ENGINE", engine)
            ply = int(os.environ.get("ICGA_PLY", ply))
            print(f"[TDGammonAI] loading agent: engine={engine} ply={ply} model={model}")
            self._bot = ICGABot(model, engine=engine, search_ply=ply)
            self._pyspiel_game = pyspiel.load_game(GAME_NAME)
            self._time_manager = TimeManager()

    # -- the only method the framework requires us to implement -------------- #
    def my_plays(self, game_history: list[str], time_left: float = inf,
                 opponent_time_left: float = inf) -> str:
        """Return our move (a string from ``self.game.textual_legal_moves()``)."""
        del opponent_time_left  # the budget only depends on our own remaining clock
        try:
            move = self._choose(game_history, time_left)
            if move is not None:
                return move
            print("[TDGammonAI] no multiset match for bot move; using fallback")
        except Exception:  # noqa: BLE001 - never crash a live game
            print("[TDGammonAI] error while choosing a move; using fallback:")
            traceback.print_exc()
        return self._fallback()

    # -- internals ----------------------------------------------------------- #
    def _reconstruct(self, game_history: list[str]) -> tuple[GameState, tuple[int, int] | None]:
        """Replay the ICGA transcript onto a fresh pyspiel state.

        Returns ``(state, dice)`` with ``state`` at our decision node and ``dice``
        the roll currently in effect.
        """
        state: GameState = self._pyspiel_game.new_initial_state()
        dice: tuple[int, int] | None = None
        for token in game_history:
            if state.is_terminal():
                break
            if state.is_chance_node():
                dice = parse_dice(token)
                state.apply_action(dice_to_chance_action(state, dice))
            elif dice is not None:
                apply_icga_move(state, token, dice)
        return state, dice

    def _deadline(self, game_history: list[str], time_left: float) -> float | None:
        """Turn the remaining clock into a per-move monotonic deadline (``None`` = fast path).

        ``move_number`` is approximated by the length of the transcript so far (it
        counts both players' half-moves and the dice tokens between them); the time
        manager only uses it to spread the clock across the moves still to come, so a
        coarse estimate is sufficient.

        Returns ``None`` (the instant 1-ply path) for an unlimited clock *or* when the
        allotted budget is too small for the anytime search to complete a useful ply,
        so the bot stays fast and never overruns the clock under time pressure.
        """
        budget = self._time_manager.budget_for_move(float(time_left), len(game_history))
        if budget is None or budget < MIN_SEARCH_BUDGET:
            return None
        return time.monotonic() + budget

    def _choose(self, game_history: list[str], time_left: float = inf) -> str | None:
        """Pick the bot's move and map it to the engine's matching legal-move string."""
        state, dice = self._reconstruct(game_history)
        if state.is_chance_node() or state.is_terminal() or dice is None:
            return None  # not our decision node -> let the fallback handle it

        player = state.current_player()
        deadline = self._deadline(game_history, time_left)
        actions, _white_eval = self._bot.play_turn(state, deadline=deadline)
        target = _move_multiset(turn_to_icga(actions, dice, player))

        for legal in self.game.textual_legal_moves():  # type: ignore[no-untyped-call]
            if _move_multiset(legal) == target:
                return str(legal)
        return None

    def _fallback(self) -> str:
        """Return some legal move so the game always proceeds."""
        legal = self.game.textual_legal_moves()  # type: ignore[no-untyped-call]
        return str(legal[0]) if legal else "pass"
